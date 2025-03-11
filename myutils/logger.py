import os
import numpy as np
import torch
import wandb


class BaseLogger:
    """BaseLogger that can handle most of the logging
    logging convention
    ------------------
    'loss' has to be exist in all training settings
    endswith('_') : scalar
    endswith('@') : image
    """
    def __init__(self, tb_writer, use_wandb=False):
        """tb_writer: tensorboard SummaryWriter"""
        self.writer = tb_writer
        self.train_loss_meter = averageMeter()
        self.val_loss_meter = averageMeter()
        self.d_train = {}
        self.d_val = {}
        self.has_val_loss = True  # If True, we assume that validation loss is available
        self.has_train_loss = True  # If True, we assume that validation loss is available
        self.use_wandb = use_wandb

    def log(self, d_result, i):
        """
        a simple method for writing data into tensorboard.
        automatically logs based on the name of the key
        """
        for key, val in d_result.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                if self.use_wandb:
                    wandb.log({key:val}, step=i)
            if key.endswith('@'):
                # key = key.strip('@')
                if val is not None:
                    self.writer.add_image(key, val, i)
                    if self.use_wandb:
                        img = wandb.Image(val)
                        wandb.log({key: img}, step=i)


    def process_iter_train(self, d_result):
        if self.has_train_loss:
            self.train_loss_meter.update(d_result['loss'])
        self.d_train = d_result

    def summary_train(self, i):
        if self.has_train_loss:
            self.d_train['loss/train_loss_'] = self.train_loss_meter.avg 
        for key, val in self.d_train.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                # wandb.log({key:val}, step=i)
            if key.endswith('@'):
                # key = key.strip('@')
                if val is not None:
                    self.writer.add_image(key, val, i)
                    # img = wandb.Image(val)
                    # wandb.log({key: img}, step=i)

        result = self.d_train
        self.d_train = {}
        self.train_loss_meter.reset()
        return result

    def process_iter_val(self, d_result):
        if self.has_val_loss:
            self.val_loss_meter.update(d_result['loss'])
        self.d_val = d_result

    def summary_val(self, i):
        if self.has_val_loss:
            self.d_val['loss/val_loss_'] = self.val_loss_meter.avg 
        l_print_str = [f'Iter [{i:d}]']
        for key, val in self.d_val.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                # wandb.log({key:val}, step=i)
                l_print_str.append(f'{key}: {val:.4f}')
            if key.endswith('@'):
                if val is not None:
                    self.writer.add_image(key, val, i)
                    # img = wandb.Image(val)
                    # wandb.log({key: img}, step=i)
            if key.endswith('#'):  # save file
                if val is not None:
                    path = os.path.join(self.writer.file_writer.get_logdir(), key.strip('#'))
                    torch.save(val, path)

        print_str = ' '.join(l_print_str)

        result = self.d_val
        result['print_str'] = print_str
        self.d_val = {}
        self.val_loss_meter.reset()
        return result
    


# From metrics.py of latent-ood
class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

