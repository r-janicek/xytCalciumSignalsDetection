import napari
from visualization_tools import get_annotations_contour, get_discrete_cmap, get_labels_cmap
import imageio
import os
import math
import numpy as np
import torch
from datasets import SparkDatasetPath
from torch import nn
from training_inference_tools import get_preds, myTrainingManager
from data_processing_tools import get_processed_result, preds_dict_to_mask
from in_out_tools import write_videos_on_disk
from training_script_utils import init_model
from torch.utils.tensorboard import SummaryWriter
import unet
import configparser

### Set parameters ###
# Parameters that are necessary to configure the dataset and the UNet model (can be eventually hard-coded in the function)

training_name = "final_model"
config_file = os.path.join("config_files", "config_final_model.ini")

c = configparser.ConfigParser()
c.read(config_file)

params = {}

# training params

params['load_epoch'] = c.getint("testing", "load_epoch")
# params['load_epoch'] = 100000
params['batch_size'] = c.getint("testing", "batch_size", fallback="1")
params["ignore_frames_loss"] = c.getint(
    "training", "ignore_frames_loss", fallback=0)

# dataset params

params["data_duration"] = c.getint("dataset", "data_duration")
params["data_step"] = c.getint("dataset", "data_step", fallback=1)
params["data_smoothing"] = c.get("dataset", "data_smoothing", fallback="2d")
params["norm_video"] = c.get("dataset", "norm_video", fallback="chunk")
params["remove_background"] = c.get(
    "dataset", "remove_background", fallback="average")
params["only_sparks"] = c.getboolean("dataset", "only_sparks", fallback=False)
params["sparks_type"] = c.get("dataset", "sparks_type", fallback="peaks")

# UNet params

params["nn_architecture"] = c.get(
    "network", "nn_architecture", fallback="pablos_unet")
if params["nn_architecture"] == "unet_lstm":
    params["bidirectional"] = c.getboolean("network", "bidirectional")
params["unet_steps"] = c.getint("network", "unet_steps")
params["first_layer_channels"] = c.getint("network", "first_layer_channels")
params["num_channels"] = c.getint("network", "num_channels", fallback=1)
params["dilation"] = c.getboolean("network", "dilation", fallback=1)
params["border_mode"] = c.get("network", "border_mode")
params["batch_normalization"] = c.get(
    "network", "batch_normalization", fallback="none")
params["temporal_reduction"] = c.getboolean(
    "network", "temporal_reduction", fallback=False)
params["initialize_weights"] = c.getboolean(
    "network", "initialize_weights", fallback=False)
if params["nn_architecture"] == "github_unet":
    params["attention"] = c.getboolean("network", "attention")
    params["up_mode"] = c.get("network", "up_mode")
if params["nn_architecture"] == "openai_unet":
    params["num_res_blocks"] = c.getint("network", "num_res_blocks")

assert params['nn_architecture'] in ['pablos_unet', 'github_unet', 'openai_unet'], \
    f"nn_architecture must be one of 'pablos_unet', 'github_unet', 'openai_unet'"


# Load UNet model

### Configure UNet ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = init_model(
    params=params,
    num_classes=4,
    ndims=3
)
network = nn.DataParallel(network).to(device)
network.eval()

### Load UNet model ###
models_relative_path = "runs/"
model_path = os.path.join(models_relative_path, training_name)
summary_writer = SummaryWriter(
    os.path.join(model_path, "summary"), purge_step=0
)

trainer = myTrainingManager(
    # training items
    training_step=None,
    save_path=model_path,
    managed_objects=unet.managed_objects({"network": network}),
    summary_writer=summary_writer,
)

trainer.load(params['load_epoch'])


# Function definition
torch.no_grad()
def get_preds_from_path(
    model,
    params,
    movie_path,
    return_dict=False,
    output_dir=None
):
    """
    Function to get predictions from a movie path
    :param model: model to use for prediction
    :param params: parameters for prediction
    :param movie_path: path to movie
    :param return_dict: if True, return dictionary else return tuple of numpy arrays
    :param output_dir: if not None, save raw predictions on disk
    :return: if return_dict is True, return dictionary with keys 'sparks',
     'puffs', 'waves' else return tuple of numpy arrays with integral values
     for classes and instances
    """

    ### Get sample as dataset ###
    sample_dataset = SparkDatasetPath(
        sample_path=movie_path,
        step=params["data_step"],
        duration=params["data_duration"],
        smoothing=params["data_smoothing"],
        remove_background=params["remove_background"],
        temporal_reduction=params["temporal_reduction"],
        num_channels=params["num_channels"],
        normalize_video=params["norm_video"],
        only_sparks=params["only_sparks"],
        sparks_type=params["sparks_type"],
        ignore_index=4,
        ignore_frames=params["ignore_frames_loss"],
        # resampling=False, # could be implemented later
        # resampling_rate=150,
    )

    ### Set physiological parameters ###

    # min distance in space between two sparks
    min_dist_xy = sample_dataset.min_dist_xy  # = 9 pixels
    # min distance in time between two sparks
    min_dist_t = sample_dataset.min_dist_t  # = 3 frames

    # spark instances detection parameters
    radius = math.ceil(min_dist_xy / 2)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    disk = x**2 + y**2 <= radius**2
    conn_mask = np.stack([disk] * (min_dist_t), axis=0)

    # parameters for removing small events
    # TODO: use better parameters !!!
    spark_min_width = 3
    spark_min_t = 3
    puff_min_t = 5
    wave_min_width = round(15 / sample_dataset.pixel_size)

    # connectivity for event instances detection
    connectivity = 26

    # maximal gap between two predicted puffs or waves that belong together
    max_gap = 2  # i.e., 2 empty frames

    sigma = 3  # for gaussian smoothing

    ### Run samples in UNet ###
    xs, preds = get_preds(
        network=model,
        test_dataset=sample_dataset,
        compute_loss=False,
        device=next(model.parameters()).device,  # same device as model
        batch_size=1,
        inference_types=None,
    )  # ys and preds are numpy arrays

    ### Get processed output ###

    # get predicted segmentation and event instances
    preds_instances, preds_segmentation, _ = get_processed_result(
        sparks=preds[1],
        puffs=preds[3],
        waves=preds[2],
        xs=xs,
        conn_mask=conn_mask,
        connectivity=connectivity,
        max_gap=max_gap,
        sigma=sigma,
        wave_min_width=wave_min_width,
        puff_min_t=puff_min_t,
        spark_min_t=spark_min_t,
        spark_min_width=spark_min_width,
        training_mode=False,
        debug=False
    )
    # preds_instances and preds_segmentations are dictionaries
    # with keys 'sparks', 'puffs', 'waves'

    # Save raw preds on disk ### I don't know if this is necessary
    # if output_dir is not None:
    # # create output directory if it does not exist
    # os.makedirs(output_dir, exist_ok=True)
    # write_videos_on_disk(
    #     training_name=None,
    #     video_name=sample_dataset.video_name,
    #     path=output_dir,
    #     preds=preds,
    #     ys=None,
    # )

    if return_dict:
        return preds_segmentation, preds_instances

    else:
        # get integral values for classes and instances
        preds_segmentation = preds_dict_to_mask(preds_segmentation)
        preds_instances = sum(preds_instances.values())
        # instances already have different ids

        return preds_segmentation, preds_instances


segmentation, instances = get_preds_from_path(
    model=network,
    params=params,
    movie_path=movie_path,
    return_dict=False,
)
