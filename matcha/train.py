# 256 0.8938063323318697
# 384 0.9822212684472911
# 512 0.9960051503846281
# 1024 0.9999174617848063
# 1536 0.9999834923569613

import argparse
import os
import yaml
import shutil

import torch

from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, Pix2StructProcessor
from code.utils import NEW_TOKENS
from code.data import generate_train_dataset, add_generated_data
from code.dataset import ImageDataModule
from code.model import MatchaModel
import pytorch_lightning as pl
import pandas as pd
from collections import Counter

import warnings

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

# or to ignore all warnings that could be false positives
#from pytorch_lightning.utilities.warnings import PossibleUserWarning
#warnings.filterwarnings("ignore", category=PossibleUserWarning)

torch.set_float32_matmul_precision('medium')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', default=None, action='store', required=True
    )

    parser.add_argument(
        '--yaml', default=None, action='store', required=True
    )

    parser.add_argument(
        '--ckpt', default=None, action='store', required=False
    )

    parser.add_argument(
        '--no_wandb', default=False, action='store_true'
    )

    parser.add_argument(
        '--project', default='benetech-fixed', action='store', required=False
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

with open(opt.yaml, 'r') as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg['model']

if 'processor' in cfg:
    PROCESSOR_NAME = cfg['processor']
else:
    PROCESSOR_NAME = cfg['model']

config = Pix2StructConfig.from_pretrained(MODEL_NAME)
config.text_config.use_cache = True

processor = Pix2StructProcessor.from_pretrained(PROCESSOR_NAME, is_vqa=False)

processor.tokenizer.add_tokens(NEW_TOKENS)

model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

train_df, valid_df, gt_df = generate_train_dataset()
train_df = add_generated_data(train_df)

cfg['dataset_size'] = len(train_df)
print('TRAIN DATASET SIZE:', cfg['dataset_size'])

print('TRAIN:', Counter(train_df['chart-type']))
print('VALID:', Counter(valid_df['chart-type']))

matcha = MatchaModel(processor, model, cfg, gt_df)

root_dir = 'runs/' + opt.name

os.makedirs(root_dir, exist_ok=True)
shutil.copyfile(opt.yaml, root_dir + '/' + opt.yaml.split('/')[-1])
shutil.copyfile('train.py', root_dir + '/train.py')

matcha.model.save_pretrained(root_dir + '/model')
matcha.processor.save_pretrained(root_dir + '/processor')

datamodule = ImageDataModule(train_df=train_df, valid_df=valid_df, processor=processor, cfg=cfg)

if opt.ckpt:
    state_dict = torch.load(opt.ckpt)['state_dict']

    for key in list(state_dict.keys()):
        state_dict[key[6:]] = state_dict.pop(key)

    matcha.model.load_state_dict(state_dict, strict=True)

loss_weights = pl.callbacks.ModelCheckpoint(
    dirpath=root_dir + '/weights',
    filename='ws-{epoch}-{score:.4f}',
    monitor='score',
    save_weights_only=True,
    save_top_k=100,
    mode='max',
    save_last=False,
)

loggers = False

if not opt.no_wandb:
    wandb_logger = pl.loggers.WandbLogger(project=opt.project, name=opt.name)
    try:
        wandb_logger.experiment.config.update(cfg)
    except:
        pass
    loggers = [wandb_logger]

trainer = pl.Trainer(
    logger=loggers,
    accelerator="gpu",
    max_epochs=cfg['epochs'],
    callbacks=[loss_weights],
    **cfg['trainer'],
)

trainer.fit(matcha, datamodule=datamodule)
