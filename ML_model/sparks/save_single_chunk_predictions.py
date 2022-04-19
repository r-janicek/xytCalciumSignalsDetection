# 19.10.2020
#
# Salva le predictions singole di ogni chunk per il video 130918 C ET-1 (con i margini ignorati dalla loss function) di modo da capire perché c'è una pred che da puff diventa improvvisamente una wave.

import os
import sys

import glob
import logging
import argparse

import numpy as np
import imageio

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import unet

from datasets import MaskDataset, MaskTestDataset
from training_tools import training_step, test_function, sampler

from tensorboardX import SummaryWriter


basepath = os.getcwd()


### Select run and video to load ###

dataset_path = os.path.join(basepath,"..","..","data")
sample_name = '130918_C_ET-1'

run_name = 'new_masks_spark_centres_new_data'
load_epoch = 80000
batch_size = 8 # same as for selected training

# ignored index
ignore_index = 4

# remove background
remove_background = True

# step and chunks duration
step = 4
chunks_duration = 16 # power of 2
ignore_frames_loss = (chunks_duration-step)//2 # frames ignored by loss fct

### Configure cuda and logger ###

unet.config_logger("/dev/null")
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
print("Using ", device, "; number of gpus: ", n_gpus)


### Create test dataset for selected video ###
test_dataset = MaskTestDataset(base_path=dataset_path,
                                   video_name=sample_name, smoothing='2d',
                                   step=step, duration=chunks_duration,
                                   remove_background = remove_background)

print("Number of chunks: ", len(test_dataset))

testing_dataset_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)

### Configure network ###
unet_config = unet.UNetConfig(
    steps=4,
    num_classes=4,
    ndims=3,
    border_mode='same',
    batch_normalization=False
)

# U-net layers
unet_config.feature_map_shapes((chunks_duration, 64, 512))

network = unet.UNetClassifier(unet_config)

network = nn.DataParallel(network)
network = network.to(device)

# uncomment to use weights initialization
#network.apply(weights_init)

optimizer = optim.Adam(network.parameters(), lr=1e-4)

output_path = "runs/"+run_name
os.makedirs(output_path, exist_ok=True)
summary_writer = SummaryWriter(os.path.join(output_path, "summary"),
                               purge_step=0)

trainer = unet.TrainingManager(
    lambda _: training_step(sampler, network, optimizer, device, class_weights,
                            dataset_loader, ignore_frames=ignore_frames_loss,
                            ignore_ind=ignore_index),
    save_every=5000,
    save_path=output_path,
    managed_objects=unet.managed_objects({'network': network,
                                          'optimizer': optimizer}),
    test_function=lambda _: test_function(network,device,class_weights,
                                          testing_datasets,logger,
                                          ignore_ind=ignore_index),
    test_every=1000,
    plot_every=1000,
    summary_writer=summary_writer # comment to use normal plots
)


### Load network parameters ###
trainer.load(load_epoch)


### Save chunks ###

chunks_folder = "saved_sample_chunks"
os.makedirs(chunks_folder, exist_ok=True)

def save_chunks(network, device, test_dataset):
    network.eval()

    preds = []

    with torch.no_grad():
        x,y = test_dataset[0]
        x = torch.Tensor(x).to(device)
        pred = network(x[None, None])[0].cpu().numpy()
        preds.append(pred)

        for i in range(1,len(test_dataset)-1):
            x,y = test_dataset[i]
            x = torch.Tensor(x).to(device)
            pred = network(x[None, None])[0].cpu().numpy()
            preds.append(pred)

        x,y = test_dataset[-1]
        x = torch.Tensor(x).to(device)
        pred = network(x[None, None])[0].cpu().numpy()
        preds.append(pred)

    # predictions have logarithmic values

    # save chunks as videos
    for i,chunk in enumerate(preds):
        imageio.volwrite(os.path.join(chunks_folder,
                                      test_dataset.video_name
                                      + "_sparks_" + str(i) + ".tif"),
                                      np.exp(chunk[1]))
        imageio.volwrite(os.path.join(chunks_folder,
                                      test_dataset.video_name
                                      + "_waves_" + str(i) + ".tif"),
                                      np.exp(chunk[2]))
        imageio.volwrite(os.path.join(chunks_folder,
                                      test_dataset.video_name
                                      + "_puffs_" + str(i) + ".tif"),
                                      np.exp(chunk[3]))

print("Saving sample video chunks...")
save_chunks(network,device,test_dataset)
