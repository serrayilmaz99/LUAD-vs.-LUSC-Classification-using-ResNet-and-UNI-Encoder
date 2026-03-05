import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom Imports
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

# PyTorch and Lightning
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

import matplotlib.pyplot as plt

fold = 4
train_accuracies = []
val_accuracies = []

def save_plot():
    """Saves the accuracy plot for a given fold and seed."""
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='s')
    #plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')

    plt.title(f'Fold {fold} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid()

    filename = f'fold_{fold}.png'
    plt.savefig(filename)
    plt.close()


class  ModelInterface(pl.LightningModule):

    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.log_path = kargs['log']
        self.n_classes = model['n_classes']

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # Initialize attributes
        
        #---->Metrics
        if self.n_classes > 2: 
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes, average='micro'),
                torchmetrics.CohenKappa(task="multiclass", num_classes=self.n_classes),
                torchmetrics.F1Score(task="multiclass", num_classes=self.n_classes, average='macro'),
                torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.n_classes),
                torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.n_classes),
                # Remove Specificity if not required or if your version doesn't support it
            ])
        else:
            metrics = torchmetrics.MetricCollection([
                torchmetrics.Accuracy(task="binary", average='micro'),
                torchmetrics.CohenKappa(task="binary"),
                torchmetrics.F1Score(task="binary", average='macro'),
                torchmetrics.Recall(task="binary", average='macro'),
                torchmetrics.Precision(task="binary", average='macro'),
            ])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data']["data_shuffle"]
        self.count = 0


    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):

        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(logits, label)

        # Accuracy Logging
        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss}

    def on_train_epoch_end(self, outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print(f'class {c}: acc {acc}, correct {correct}/{count}')
        
        t_correct = self.data[0]["correct"] + self.data[1]["correct"]
        t_count = self.data[0]["count"] + self.data[1]["count"]
        train_accuracies.append(100*(t_correct/t_count))
        

        # Reset data tracking for the next epoch
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]
        

    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def validation_epoch_end(self, validation_outputs):
        logits = torch.cat([x['logits'] for x in validation_outputs], dim=0)
        probs = torch.cat([x['Y_prob'] for x in validation_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in validation_outputs])
        target = torch.stack([x['label'] for x in validation_outputs], dim=0)

        # Log metrics
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        #self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
                          on_epoch = True, logger = True)

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))

        t_correct = self.data[0]["correct"] + self.data[1]["correct"]
        t_count = self.data[0]["count"] + self.data[1]["count"]
        val_accuracies.append(100*(t_correct/t_count))
        
        
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

     

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
            data, label = batch
            results_dict = self.model(data=data, label=label)
            logits = results_dict['logits']
            Y_prob = results_dict['Y_prob']
            Y_hat = results_dict['Y_hat']

            #---->acc log
            Y = int(label)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat.item() == Y)

            return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
            probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
            max_probs = torch.stack([x['Y_hat'] for x in output_results])
            target = torch.stack([x['label'] for x in output_results], dim = 0)
            
            #---->
            #auc = self.AUROC(probs, target.squeeze())
            metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
            #metrics['auc'] = auc
            for keys, values in metrics.items():
                print(f'{keys} = {values}')
                metrics[keys] = values.cpu().numpy()
            print()
            #---->acc log
            for c in range(self.n_classes):
                count = self.data[c]["count"]
                correct = self.data[c]["correct"]
                if count == 0: 
                    acc = None
                else:
                    acc = float(correct) / count
                print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
            self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
            #---->
            result = pd.DataFrame([metrics])
            result.to_csv(self.log_path / 'result.csv')

    def load_model(self):
        name = "TransMIL"  # self.hparams.model.name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])  # Generates 'Transmil'

        try:
            module = importlib.import_module(f'models.{name}')
            Model = getattr(module, camel_name)
        except ModuleNotFoundError:
            raise ValueError(f"Module 'models.{name}' not found. Ensure the file exists.")
        except AttributeError:
            raise ValueError(f"Class '{camel_name}' not found in module 'models.{name}'. Ensure the class name matches.")
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()

        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams.model[arg]  # Fixed access here
        args1.update(other_args)
        return Model(**args1)
