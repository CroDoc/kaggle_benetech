import argparse
import yaml
import torch

from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, Pix2StructProcessor
from code.utils import NEW_TOKENS
from code.data import generate_train_dataset
from code.dataset import ImageDataModule
from code.model import MatchaModel
import pytorch_lightning as pl

torch.set_float32_matmul_precision('medium')

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--yaml', default=None, action='store', required=True
    )

    parser.add_argument(
        '--ckpt', default=None, action='store', required=False
    )

    return parser.parse_known_args()[0]

opt = parse_opt()

with open(opt.yaml, 'r') as f:
    cfg = yaml.safe_load(f)

cfg['skip_validation'] = 0

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
#valid_df = valid_df[valid_df['chart-type'] == 'line']

matcha = MatchaModel(processor, model, cfg, gt_df)

cfg['dataset_size'] = len(train_df)

datamodule = ImageDataModule(train_df=train_df, valid_df=valid_df, processor=processor, cfg=cfg)

if opt.ckpt:
    state_dict = torch.load(opt.ckpt)['state_dict']

    for key in list(state_dict.keys()):
        state_dict[key[6:]] = state_dict.pop(key)

    matcha.model.load_state_dict(state_dict, strict=True)

trainer = pl.Trainer(
    logger=False,
    accelerator="gpu",
    max_epochs=cfg['epochs'],
    **cfg['trainer'],
)

trainer.validate(matcha, datamodule=datamodule)
