from pathlib import Path
import yaml
from addict import Dict
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn.functional as F

# ----> Read YAML
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

# ----> Load Loggers
def load_loggers(cfg):
    log_path = cfg['General']['log_path']
    Path(log_path).mkdir(exist_ok=True, parents=True)
    log_name = Path(cfg['config']).parent
    version_name = Path(cfg['config']).name[:-5]
    cfg['log_path'] = Path(log_path) / log_name / version_name / f"fold{cfg['Data']['fold']}"
    print(f"---->Log dir: {cfg['log_path']}")

    #---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=str(cfg['log_path']),
        name=None,
        log_graph=True,
        default_hp_metric=False  # Disable hyperparameter logging
    )
    #---->CSV Logger (optional, can be disabled completely)
    csv_logger = pl_loggers.CSVLogger(
        save_dir=str(cfg['log_path']),
        name=None
    )

    return [tb_logger, csv_logger]

# ----> Load Callbacks
def load_callbacks(cfg):
    my_callbacks = []
    # Make output path
    output_path = cfg['log_path']
    output_path.mkdir(exist_ok=True, parents=True)

    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg['General']['patience'],
        verbose=True,
        mode='min',
    )
    my_callbacks.append(early_stop_callback)

    # Model Checkpoint
    if cfg['General']['server'] == 'train':
        my_callbacks.append(
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=str(cfg['log_path']),
                filename='{epoch:02d}-{val_loss:.4f}',
                verbose=True,
                save_last=True,
                save_top_k=1,
                mode='min',
                save_weights_only=True,
            )
        )
    return my_callbacks

# ----> Custom Loss Function
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i]) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    loss = -torch.sum(x_log) / len(y)
    return loss
