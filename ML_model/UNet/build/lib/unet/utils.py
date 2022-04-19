
import os
import logging

import numpy as np

import torch
from torch.utils.data import Dataset

__all__ = ["TransformedDataset", "build_sampler",
           "config_logger", "invfreq_weights", "BinCounter",
           "arraymap"]


class TransformedDataset(Dataset):
    
    def __init__(self, source_dataset, transform):
        self.source_dataset = source_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.source_dataset)
    
    def __getitem__(self, idx):
        return self.transform(*self.source_dataset[idx])


def _cycle(iterable):
    while True:
        for i in iterable:
            yield i


def build_sampler(loader):
    
    def sampler(_aux=_cycle(loader)):
        return next(_aux)
    
    return sampler


def labels_to_probabilities(labels, num_labels):
    
    probs = np.zeros((len(labels), num_labels) + labels.shape[1:], dtype=np.float32)
    a = np.arange(len(labels))
    probs[(a[:, None, None], labels) + np.ix_(*map(range, labels.shape[1:]))] = 1
    return probs


def pad_for_unet(array, unet_config, mode='reflect', padding_for='blocks'):
    """
    Pads an array for U-Net training and prediction. Detects if the array has
    more than one channel and does not pad over those dimensions.
    
    `padding_for` can be one of ['blocks', 'input', 'ouput'].
    """
    
    ndims = unet_config.ndims
    array_shape = array.shape[:ndims] # Array shape ignoring channels
    
    if padding_for == 'blocks':
        margin = unet_config.margin()
        pad_width = [(margin, margin)] * ndims
    elif padding_for == 'input':
        pad_width, _ = unet_config.in_out_pad_widths(array_shape)
    elif padding_for == 'output':
        _, pad_width = unet_config.in_out_pad_widths(array_shape)
    else:
        raise ValueError("Unknown `padding_for` value '{}'".format(padding_for))
    
    if array.ndim > ndims:
        pad_width = pad_width + [(0, 0)] * (array.ndim - ndims)
    
    return np.pad(array, pad_width, mode)


class BinCounter:
    """Counter of elements in NumPy arrays."""
    
    def __init__(self, minlength=0, x=None, weights=None):
        
        self.minlength = minlength
        self.counts = np.zeros(minlength, dtype=np.int_)
        
        if x is not None and len(x) > 0:
            self.update(x, weights)
    
    def update(self, x, weights=None):
        if weights is not None:
            weights = weights.flatten()
        
        minlength = max(len(self.counts), self.minlength)
        current_counts = np.bincount(np.ravel(x), weights=weights, minlength=minlength)
        current_counts[:len(self.counts)] += self.counts
        
        self.counts = current_counts
    
    @property
    def frequencies(self):
        return self.counts / np.float_(np.sum(self.counts))
    

def invfreq_weights(bin_counter, num_classes=None):
    if num_classes is None:
        num_classes = len(bin_counter.counts)
    class_weight = 1.0 / (num_classes * bin_counter.frequencies)[:num_classes]
    return class_weight


def arraymap(f, arr):
    return np.array(list(map(f, arr)))


def config_logger(log_file):
    """
    Basic configuration of the logging system. Support logging to a file.
    """
    
    class MyFormatter(logging.Formatter):
        
        info_format = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        error_format = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"
        
        def format(self, record):
            
            if record.levelno > logging.INFO:
                self._style._fmt = self.error_format
            else:
                self._style._fmt = self.info_format
            
            return super(MyFormatter, self).format(record)
    
    root_logger = logging.getLogger()
    
    dirname = os.path.dirname(log_file)
    os.makedirs(dirname, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_formatter = MyFormatter()
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    root_logger.setLevel(logging.INFO)
