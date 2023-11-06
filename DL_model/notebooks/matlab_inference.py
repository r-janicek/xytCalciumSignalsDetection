# To import modules from parent directory in Jupyter Notebook
import sys
sys.path.append("..")

import os

import imageio
import numpy as np
import torch
from torch import nn

from DL_model.config import TrainingConfig, config

from DL_model.utils.training_inference_tools import get_final_preds
from DL_model.utils.training_script_utils import init_model
from DL_model.utils.visualization_tools import (
    get_annotations_contour,
    get_discrete_cmap,
    get_labels_cmap,
)

### Set training-specific parameters ###

# Initialize training-specific parameters
config_path = os.path.join("config_files", "config_final_model.ini")
params = TrainingConfig(training_config_file=config_path)
params.run_name = "final_model"
model_filename = f"network_100000.pth"

### Configure UNet ###
# params.set_device(device="auto")
params.set_device(device="cpu")  # temporary

network = init_model(params=params)

# Move the model to the GPU if available
if params.device.type != "cpu":
    network = nn.DataParallel(network).to(params.device, non_blocking=True)
    # cudnn.benchmark = True

### Load UNet model ###

# Path to the saved model checkpoint
models_relative_path = os.path.join(
    "models", "saved_models", params.run_name, model_filename
)
model_dir = os.path.realpath(os.path.join(config.basedir, models_relative_path))

# Load the model state dictionary
try:
    network.load_state_dict(torch.load(model_dir, map_location=params.device))
except RuntimeError as e:
    if "module" in str(e):
        # The error message contains "module," so handle the DataParallel loading
        print(
            "Failed to load the model, as it was trained with DataParallel. Wrapping it in DataParallel and retrying..."
        )
        # Get current device of the object (model)
        temp_device = next(iter(network.parameters())).device

        network = nn.DataParallel(network)
        network.load_state_dict(torch.load(model_dir, map_location=params.device))

        print("Network should be on CPU, removing DataParallel wrapper...")
        network = network.module.to(temp_device)
    else:
        # Handle other exceptions or re-raise the exception if it's unrelated
        raise


network.eval()
segmentation, instances = get_final_preds(
    model=network,
    params=params,
    movie=movie_matrix,
)

