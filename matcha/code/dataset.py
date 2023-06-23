from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import torch
import numpy as np
from code.generator import GraphGenerator

from code.transforms import get_transfos

class ImageDataset(Dataset):
    def __init__(self, df, processor, cfg, validation):

        self.processor = processor
        self.cfg = cfg
        self.validation = validation

        self.transforms = None

        if not self.validation:
            self.graph_generator = GraphGenerator(df['id'].values)

            if cfg['aug_strength']:
                self.transforms = get_transfos(cfg['aug_strength'])
                print('-> Aug strength :', cfg['aug_strength'])

        self.ids = df['id'].values
        self.image_paths = df['image_path'].values
        self.ground_truths = df['ground_truth'].values
        self.sources = df['source'].values
        self.chart_types = df['chart-type'].values

    def __len__(self):
        return len(self.ids)
    
    def make_label(self, text):
        return self.processor.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.cfg['max_length'],
            padding=True,
            truncation=True,
        ).input_ids

    def __getitem__(self, idx):

        source = self.sources[idx]

        if source == 'my_generated':
            chart_type = self.chart_types[idx]
            img, ground_truth = self.graph_generator.generate(chart_type)
        else:
            img = Image.open(self.image_paths[idx])
            ground_truth = self.ground_truths[idx]
        
        if self.transforms is not None:
            img = self.transforms(image=np.array(img))["image"]

        img = self.processor(img, return_tensors='pt')
        label = self.make_label(ground_truth)

        return self.ids[idx], img, label

class CustomCollator():

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        ids, images, labels = [], [], []

        for item in batch:
            ids.append(item[0])
            images.append(item[1])
            labels.append(item[2])
        
        images = {
            "flattened_patches": torch.tensor(np.vstack([img['flattened_patches'] for img in images])),
            "attention_mask": torch.tensor(np.vstack([img['attention_mask'] for img in images])),
        }

        max_length = max(len(label) for label in labels)
        labels = torch.tensor([label + [self.pad_token_id] * (max_length - len(label)) for label in labels])

        return ids, images, labels

class PredictImageDataset(Dataset):
    def __init__(self, df, processor, cfg):

        self.processor = processor
        self.cfg = cfg

        self.ids = df['id'].values
        self.image_paths = df['image_path'].values

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img = Image.open(self.image_paths[idx])
        img = self.processor(img, return_tensors='pt')

        return self.ids[idx], img

class PredictCustomCollator():

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        ids, images = [], []

        for item in batch:
            ids.append(item[0])
            images.append(item[1])
        
        images = {
            "flattened_patches": torch.tensor(np.vstack([img['flattened_patches'] for img in images])),
            "attention_mask": torch.tensor(np.vstack([img['attention_mask'] for img in images])),
        }

        return ids, images

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df = None,
        valid_df = None,
        predict_df = None,
        processor = None,
        cfg = None,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.predict_df = predict_df

        self.processor = processor
        self.cfg = cfg

    def setup(self, stage):

        if stage in ['fit', 'validate']:
            self.train_dataset = ImageDataset(self.train_df, self.processor, self.cfg, validation=False)
            self.valid_dataset = ImageDataset(self.valid_df, self.processor, self.cfg, validation=True)
        elif stage == 'predict':
            self.predict_dataset = PredictImageDataset(self.predict_df, self.processor, self.cfg)
        else:
            raise Exception()

    def train_dataloader(self):
        custom_collator = CustomCollator(self.processor.tokenizer.pad_token_id)
        return DataLoader(self.train_dataset, **self.cfg["train_loader"], collate_fn=custom_collator)

    def val_dataloader(self):
        custom_collator = CustomCollator(self.processor.tokenizer.pad_token_id)
        return DataLoader(self.valid_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)

    def predict_dataloader(self):
        custom_collator = PredictCustomCollator(self.processor.tokenizer.pad_token_id)
        return DataLoader(self.predict_dataset, **self.cfg["val_loader"], collate_fn=custom_collator)
