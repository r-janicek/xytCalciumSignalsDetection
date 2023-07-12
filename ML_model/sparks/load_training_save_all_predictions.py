# 19.02.2021
#
# Load a stored trained network at a given epoch and save all predictions as
# .tif videos (for both trainining and testing datasets).

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

from datasets import IDMaskTestDataset
from training_tools import training_step, test_function, sampler

from tensorboardX import SummaryWriter


basepath = os.getcwd()


### Select run and dataset to load ###

dataset_path = os.path.join(basepath,"..","data")
dataset_name = "temp_annotation_masks"
#sample_name = '130918_C_ET-1'

# get all videos names
train_files = sorted([".".join(f.split(".")[:-1]) for f in
             os.listdir(os.path.join(dataset_path, dataset_name,"videos"))])
test_files = sorted([".".join(f.split(".")[:-1]) for f in
             os.listdir(os.path.join(dataset_path, dataset_name,"videos_test"))])



# run parameters
run_name = "focal_loss_w_weights"
load_epoch = 100000
batch_size = 2 # same as for selected training

# ignored index
ignore_index = 4

# remove background
remove_background = True

# step and chunks duration
step = 4
chunks_duration = 64 # power of 2
half_overlap = (chunks_duration-step)//2 # to re-build videos from chunks
ignore_frames_loss = 4 # frames ignored by loss fct

### Configure cuda and logger ###

unet.config_logger("/dev/null")
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()
print("Using ", device, "; number of gpus: ", n_gpus)



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

# Run every video on unet and store preds
save_folder = "SAVED_PREDS"
os.makedirs(save_folder, exist_ok=True)

def save_preds(network, device, test_dataset):
    network.eval()

    xs = []
    ys = []
    preds = []
    sparks = []
    puffs = []
    waves = []

    with torch.no_grad():
        x,y = test_dataset[0]
        xs.append(x[:-half_overlap])
        ys.append(y[:-half_overlap])

        x = torch.Tensor(x).to(device)
        pred = network(x[None, None])[0].cpu().numpy()
        preds.append(pred[:,:-half_overlap])
        sparks.append(pred[1,:-half_overlap])
        puffs.append(pred[3,:-half_overlap])
        waves.append(pred[2,:-half_overlap])

        for i in range(1,len(test_dataset)-1):
            x,y = test_dataset[i]
            xs.append(x[half_overlap:-half_overlap])
            ys.append(y[half_overlap:-half_overlap])

            x = torch.Tensor(x).to(device)
            pred = network(x[None, None])[0].cpu().numpy()
            preds.append(pred[:,half_overlap:-half_overlap])
            sparks.append(pred[1,half_overlap:-half_overlap])
            puffs.append(pred[3,half_overlap:-half_overlap])
            waves.append(pred[2,half_overlap:-half_overlap])

        x,y = test_dataset[-1]
        xs.append(x[half_overlap:])
        ys.append(y[half_overlap:])

        x = torch.Tensor(x).to(device)
        pred = network(x[None, None])[0].cpu().numpy()
        preds.append(pred[:,half_overlap:])
        sparks.append(pred[1,half_overlap:])
        puffs.append(pred[3,half_overlap:])
        waves.append(pred[2,half_overlap:])

    # concatenated frames and predictions for a single video:
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    sparks = np.concatenate(sparks, axis=0)
    waves = np.concatenate(waves, axis=0)
    puffs = np.concatenate(puffs, axis=0)
    preds = np.concatenate(preds, axis=1)

    if test_dataset.pad != 0:
        xs = xs[:-test_dataset.pad]
        ys = ys[:-test_dataset.pad]
        sparks = sparks[:-test_dataset.pad]
        waves = waves[:-test_dataset.pad]
        puffs = puffs[:-test_dataset.pad]
        preds = preds[:,:-test_dataset.pad]

    # predictions have logarithmic values

    # save preds as videos
    imageio.volwrite(os.path.join(save_folder,
                                  run_name+"_"+str(load_epoch)+"_"
                                  +video_name + "_xs.tif"),
                                  xs)
    imageio.volwrite(os.path.join(save_folder,
                                  run_name+"_"+str(load_epoch)+"_"
                                  +video_name + "_ys.tif"),
                                  np.uint8(ys))
    imageio.volwrite(os.path.join(save_folder,
                                  run_name+"_"+str(load_epoch)+"_"
                                  +video_name + "_sparks.tif"),
                                  np.exp(sparks))
    imageio.volwrite(os.path.join(save_folder,
                                  run_name+"_"+str(load_epoch)+"_"
                                  +video_name + "_waves.tif"),
                                  np.exp(waves))
    imageio.volwrite(os.path.join(save_folder,
                                  run_name+"_"+str(load_epoch)+"_"
                                  +video_name + "_puffs.tif"),
                                  np.exp(puffs))


print("TRAIN VIDEOS")

for video_name in train_files:
    print("Filename: ", video_name)

    ### Create test dataset ###
    dataset = IDMaskTestDataset(base_path=os.path.join(dataset_path,dataset_name),
                                     video_name=video_name, smoothing='2d',
                                     step=step, duration=chunks_duration,
                                     #radius_event = radius_event,
                                     remove_background = remove_background,
                                     test=False)

    dataset_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=1)

    print("Saving unet preds...")
    save_preds(network,device,dataset)


print("TEST VIDEOS")

for video_name in test_files:
    print("Filename: ", video_name)

    ### Create test dataset ###
    dataset = IDMaskTestDataset(base_path=os.path.join(dataset_path,dataset_name),
                                     video_name=video_name, smoothing='2d',
                                     step=step, duration=chunks_duration,
                                     #radius_event = radius_event,
                                     remove_background = remove_background,
                                     test=True)

    dataset_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    print("Saving unet preds...")
    save_preds(network,device,dataset)
