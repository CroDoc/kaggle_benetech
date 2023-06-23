import pytorch_lightning as pl

from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from code.score import CustomMetric
import torch
from code.score import val_score

class MatchaModel(pl.LightningModule):
    def __init__(self, processor, model, cfg, gt_df):

        super().__init__()
        self.processor = processor
        self.model = model
        self.cfg = cfg

        self.pad_token_id = processor.tokenizer.pad_token_id

        self.metrics = [CustomMetric(gt_df, metric='score')]

        for chart in sorted(set(gt_df['chart_type'])):
            self.metrics.append(CustomMetric(gt_df, metric=chart))
        
        self.gt_df = gt_df
        self.chart_type_loss = torch.nn.CrossEntropyLoss()

        if 'skip_validation' in self.cfg:
            self.skip_validation = self.cfg['skip_validation']

    def training_step(self, batch, batch_idx):

        _, images, labels = batch
        
        labels[labels == self.pad_token_id] = -100

        outputs = self.model(**images, labels=labels)
        
        chart_type_loss = self.chart_type_loss(outputs.logits[:, 0], labels[:, 0])

        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("chart_loss", chart_type_loss, prog_bar=True)
        self.log("lr", self.optimizer.param_groups[-1]['lr'], prog_bar=True)

        loss += chart_type_loss
        
        return loss

    def on_validation_epoch_end(self):
        if self.skip_validation > 0:
            self.skip_validation -= 1

    def validation_step(self, batch, batch_idx):

        if self.skip_validation > 0:
            for metric in self.metrics:
                metric([], [])
                self.log(metric.metric, metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, metric_attribute=metric)
            return
        
        ids, images, _ = batch

        outputs = self.model.generate(**images, max_new_tokens=self.cfg['max_length'])
        
        preds = self.processor.batch_decode(outputs, skip_special_tokens=True)

        for metric in self.metrics:
            metric(preds, ids)
            self.log(metric.metric, metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, metric_attribute=metric)

    def predict_step(self, batch, batch_idx):
        ids, images = batch

        outputs = self.model.generate(**images, max_new_tokens=self.cfg['max_length'])
        preds = self.processor.batch_decode(outputs, skip_special_tokens=True)

        sc = val_score(preds, ids, self.gt_df)

        return ids, preds, sc[0], sc[1]

    def configure_optimizers(self):

        weight_decay = self.cfg['optimizer']['weight_decay']

        param_optimizer = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg['optimizer']['params']['lr'],
            },
        ]

        optimizer = eval(self.cfg['optimizer']['name'])(
            optimizer_parameters, **self.cfg['optimizer']['params']
        )

        self.optimizer = optimizer

        if 'scheduler' in self.cfg:
            scheduler_name = self.cfg['scheduler']['name']
            params = self.cfg['scheduler']['params']

            if scheduler_name in ['poly', 'cosine']:
                epoch_steps = self.cfg['dataset_size']
                batch_size = self.cfg['train_loader']['batch_size']
                acc_steps = self.cfg['trainer']['accumulate_grad_batches']
                num_gpus = self.cfg['trainer']['devices']
                
                print(f'- Dataset size : {epoch_steps} - Effective batch size :  {batch_size * acc_steps * num_gpus}')
            
                warmup_steps = self.cfg['scheduler']['warmup'] * epoch_steps// (
                    batch_size * acc_steps * num_gpus
                )
                training_steps = self.cfg['epochs'] * epoch_steps // (
                    batch_size * acc_steps * num_gpus
                )

                print(f"Total warmup steps : {warmup_steps} - Total training steps : {training_steps}")

                if scheduler_name == 'poly':
                    power = params['power']
                    lr_end = params['lr_end']
                    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)
                elif scheduler_name == 'cosine':
                    print(params)
                    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, **params)
                else:
                    raise NotImplemented('not implemented!')
            else:
                scheduler = eval(scheduler_name)(
                    optimizer, **params
                )

            lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': self.cfg['scheduler']['interval'],
                'frequency': 1,
            }

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

        return optimizer
    
class PredMatchaModel(pl.LightningModule):
    def __init__(self, processor, model, cfg):

        super().__init__()
        self.processor = processor
        self.model = model
        self.cfg = cfg

        self.pad_token_id = processor.tokenizer.pad_token_id

    def predict_step(self, batch, batch_idx):
        ids, images = batch

        outputs = self.model.generate(**images, max_new_tokens=self.cfg['max_length'])
        preds = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return ids, preds