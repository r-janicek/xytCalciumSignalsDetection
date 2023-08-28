"""
14.12.2021 (last update: 21.10.2022)

Load a saved UNet model at given epochs and save its predictions in the
folder `trainings_validation`.

Predictions are saved as:
`{training_name}_{epoch}_{video_id}_{class}.tif`

**Idea**: Use predictions to produce plots and tables to visualize the results.
"""

import configparser
import logging
import os
import sys

import numpy as np
import torch
from architectures import TempRedUNet
from datasets import SparkDataset
from in_out_tools import write_videos_on_disk
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training_inference_tools import get_preds

import unet

BASEDIR = os.path.dirname(os.path.realpath(__file__))


################################ Set parameters ################################

training_name = "final_model"
config_file = "config_final_model.ini"
use_train_data = False


########################### Configure output folder ############################

output_folder = "trainings_validation"  # same folder for train and test preds
os.makedirs(output_folder, exist_ok=True)

# subdirectory of output_folder where predictions are saved
# change this to save results for same model with different inference approaches
#output_name = training_name + "_step=2"
output_name = training_name

save_folder = os.path.join(output_folder, output_name)
os.makedirs(save_folder, exist_ok=True)


############################### Configure logger ###############################

logger = logging.getLogger(__name__)

log_level = logging.DEBUG
log_handlers = (logging.StreamHandler(sys.stdout),)

logging.basicConfig(
    level=log_level,
    format="[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}",
    style="{",
    datefmt="%H:%M:%S",
    handlers=log_handlers,
)


########################### Detect GPU, if available ###########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
logger.info(f"Using device '{device}' with {n_gpus} GPUs")


############################### Load config file ###############################

config_folder = "config_files"
CONFIG_FILE = os.path.join(BASEDIR, config_folder, config_file)
c = configparser.ConfigParser()
if os.path.isfile(CONFIG_FILE):
    logger.info(f"Loading {CONFIG_FILE}")
    c.read(CONFIG_FILE)
else:
    logger.info(
        f"No config file found at {CONFIG_FILE}, trying to use fallback values."
    )


######################## Config dataset and UNet model #########################

logger.info(f"Processing training '{training_name}'...")

### Params ###
load_epoch = c.getint("testing", "load_epoch")

batch_size = c.getint("testing", "batch_size", fallback="1")
ignore_frames = c.getint("training", "ignore_frames_loss")

temporal_reduction = c.getboolean(
    "network", "temporal_reduction", fallback=False)
num_channels = (
    c.getint("network", "num_channels",
             fallback=1) if temporal_reduction else 1
)

### Configure dataset/inference method ###
dataset_size = c.get("testing", "dataset_size")
data_step = c.getint("testing", "data_step")
data_duration = c.getint("testing", "data_duration")
inference = c.get("testing", "inference")

if use_train_data:
    logger.info("Predict outputs for training data")
    if dataset_size == "full":
        sample_ids = [
            "01",
            "02",
            "03",
            "04",
            "06",
            "07",
            "08",
            "09",
            "11",
            "12",
            "13",
            "14",
            "16",
            "17",
            "18",
            "19",
            "21",
            "22",
            "23",
            "24",
            "27",
            "28",
            "29",
            "30",
            "33",
            "35",
            "36",
            "38",
            "39",
            "41",
            "42",
            "43",
            "44",
            "46",
        ]
    elif dataset_size == "minimal":
        sample_ids = ["01"]
else:
    logger.info("Predict outputs for testing data")
    if dataset_size == "full":
        sample_ids = ["05", "10", "15", "20", "25", "32", "34", "40", "45"]
    elif dataset_size == "minimal":
        sample_ids = ["34"]

relative_path = c.get("dataset", "relative_path")
dataset_path = os.path.realpath(f"{BASEDIR}/{relative_path}")
assert os.path.isdir(dataset_path), f'"{dataset_path}" is not a directory'
logger.info(f"Using {dataset_path} as dataset root path")
logger.info(f"Annotations and predictions will be saved on '{save_folder}'")

### Configure UNet ###

batch_norm = {"batch": True, "none": False}

unet_config = unet.UNetConfig(
    steps=c.getint("network", "unet_steps"),
    first_layer_channels=c.getint("network", "first_layer_channels"),
    num_classes=4,
    ndims=3,
    dilation=c.getint("network", "dilation", fallback=1),
    border_mode=c.get("network", "border_mode"),
    batch_normalization=batch_norm[c.get("network", "batch_normalization")],
    num_input_channels=num_channels,
)
if not temporal_reduction:
    network = unet.UNetClassifier(unet_config)
else:
    assert (
        c.getint("dataset", "data_duration") % num_channels == 0
    ), "using temporal reduction chunks_duration must be a multiple of num_channels"
    network = TempRedUNet(unet_config)

network = nn.DataParallel(network).to(device)

### Load UNet model ###
models_relative_path = "runs/"
model_path = os.path.join(models_relative_path, training_name)
# logger.info(f"Saved model path: {model_path}")
summary_writer = SummaryWriter(
    os.path.join(model_path, "summary"), purge_step=0)

trainer = unet.TrainingManager(
    # training items
    training_step=None,
    save_path=model_path,
    managed_objects=unet.managed_objects({"network": network}),
    summary_writer=summary_writer,
)

logger.info(
    f"Loading trained model '{training_name}' at epoch {load_epoch}...")
trainer.load(load_epoch)
# logger.info(f"Loaded trained model located in '{model_path}'")


############################# Run samples in UNet ##############################

for sample_id in sample_ids:
    ### Create dataset ###
    testing_dataset = SparkDataset(
        base_path=dataset_path,
        sample_ids=[sample_id],
        testing=False,  # we just do inference, without metrics computation
        smoothing=c.get("dataset", "data_smoothing"),
        step=data_step,
        duration=data_duration,
        remove_background=c.get("dataset", "remove_background"),
        temporal_reduction=c.getboolean(
            "network", "temporal_reduction", fallback=False
        ),
        num_channels=num_channels,
        normalize_video=c.get("dataset", "norm_video"),
        only_sparks=c.getboolean("dataset", "only_sparks", fallback=False),
        sparks_type=c.get("dataset", "sparks_type"),
        ignore_frames=c.get("training", "ignore_frames_loss"),
        ignore_index=4,
        gt_available=True,
        inference=inference,
    )

    logger.info(
        f"\tTesting dataset of movie {testing_dataset.video_name} "
        f"\tcontains {len(testing_dataset)} samples."
    )

    logger.info(f"\tProcessing samples in UNet...")
    # ys and preds are numpy arrays
    _, ys, preds = get_preds(
        network=network, test_dataset=testing_dataset, compute_loss=False, device=device
    )

    ### Save preds on disk ###
    logger.info(f"\tSaving annotations and predictions...")

    video_name = f"{str(load_epoch)}_{testing_dataset.video_name}"

    # preds are in logarithmic scale, compute exp
    preds = np.exp(preds)

    write_videos_on_disk(
        training_name=output_name,
        video_name=video_name,
        path=save_folder,
        preds=preds,
        ys=ys,
    )

logger.info(f"DONE")
