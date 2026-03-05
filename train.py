import argparse
from pathlib import Path
import numpy as np
import glob
import torch

torch.use_deterministic_algorithms(True, warn_only=True)

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *
import addict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from models.model_interface import *

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str, help="Stage to run: train/test")
    parser.add_argument('--config', default='config/TransMIL.yaml', type=str, help="Path to configuration file")
    parser.add_argument('--gpus', default=0, type=int, help="GPU index to use")
    parser.add_argument('--fold', default=0, type=int, help="Data fold to use")
    return parser.parse_args()

def main(cfg):
    # Convert addict.Dict to dict for compatibility
    if isinstance(cfg, addict.Dict):
        cfg = cfg.to_dict()

    # Debugging: Print YAML representation

    # Set global seed for reproducibility
    pl.seed_everything(cfg['General']['seed'])

    # Load loggers and callbacks
    cfg['load_loggers'] = load_loggers(cfg)
    cfg['callbacks'] = load_callbacks(cfg)

    DataInterface_dict = {
        'train_batch_size': cfg['Data']['train_dataloader']['batch_size'],
        'train_num_workers': cfg['Data']['train_dataloader']['num_workers'],
        'test_batch_size': cfg['Data']['test_dataloader']['batch_size'],
        'test_num_workers': cfg['Data']['test_dataloader']['num_workers'],
        'dataset_name': cfg['Data']['dataset_name'],
        'dataset_cfg': cfg['Data'],
    }
    dm = DataInterface(**DataInterface_dict)

    ModelInterface_dict = {
        'model': cfg['Model'],
        'loss': cfg['Loss'],
        'optimizer': cfg['Optimizer'],
        'data': cfg['Data'],
        'log': cfg['log_path'],
    }
    model = ModelInterface(**ModelInterface_dict)
    
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg['load_loggers'],
        callbacks=cfg['callbacks'],
        max_epochs= 200,
        gpus=[0],
        amp_level="O2",  
        precision=16,  
        accumulate_grad_batches=2,
        deterministic=True,
        check_val_every_n_epoch=1,
    )

    if cfg['General']['server'] == 'train':
        trainer.fit(model=model, datamodule=dm)
        save_plot()
    else:
        model_paths = list(cfg['log_path'].glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            new_model =  ModelInterface.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)

    # Convert addict.Dict to plain dict if necessary
    if isinstance(cfg, addict.Dict):
        cfg = cfg.to_dict()

    cfg['config'] = args.config
    cfg['General']['gpus'] = args.gpus
    cfg['General']['server'] = args.stage
    cfg['Data']['fold'] = args.fold
    main(cfg)