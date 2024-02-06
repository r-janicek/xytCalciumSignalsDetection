"""
Create classes to manage the project.

Classes:
    ProjectConfig: stores all the global variables for the project.
    TrainingConfig: loads settings from a configuration file, initializes
                    parameters, and configures WandB (Weights and Biases)
                    logging.

Author: Prisca Dotti
Last modified: 21.10.2023
"""


import logging
import math
import os
import sys
import warnings
from configparser import ConfigParser
from logging.handlers import RotatingFileHandler

import numpy as np
import torch

import wandb

__all__ = ["config", "TrainingConfig"]

logger = logging.getLogger(__name__)


class ProjectConfig:
    def __init__(self):
        """
        Initialize the configuration object.
        The configuration object stores all the global variables for the project.

        Comment for Matlab GUI: to change the values of the attributes, you can
        access them using dot notation, e.g.
        config.min_size["sparks"] = [1, 2, 3].
        """
        # Get basedir of the project
        self.basedir = os.path.dirname(os.path.realpath(__file__))

        ### General parameters ###

        self.logfile = (
            None  # Change this when publishing the finished project on GitHub
        )
        self.verbosity = 2
        self.debug_mode = False
        # wandb_project_name = "TEST"
        self.wandb_project_name = "sparks2"
        # Directory where output, saved parameters, and testing results are saved
        self._output_relative_dir = os.path.join("models", "saved_models")

        ### Dataset parameters ###

        self.ndims = 3  # Using 3D data
        self.pixel_size = 0.2  # 1 pixel = 0.2 um x 0.2 um
        self.time_frame = 6.8  # 1 frame = 6.8 ms

        self.classes_dict: dict[str, int] = {
            "background": 0,
            "sparks": 1,
            "waves": 2,
            "puffs": 3,
            # "transient": 4,
            # "undefined": 5,  type of local signal (not spark of puff)
        }
        # note: the class values have to be consecutive
        self.event_types = ["sparks", "waves", "puffs"]

        self.ignore_index = 4  # Label ignored during training

        # Include ingore index in the classes dictionary
        self.classes_dict["ignore"] = self.ignore_index

        ### Physiological parameters ###

        # Minimal dimensions to remove small events in UNet detections
        self.min_size = {
            "sparks": [2, 3, 3],
            "waves": [None, None, round(15 / self.pixel_size)],
            "puffs": [5, None, None],
        }  # [duration, height, width]

        ## Sparks (1) parameters ##

        # To get sparks locations
        # Connectivity mask of sparks
        self._set_sparks_connectivity_mask()
        # Sigma value used for sample smoothing in sparks peaks detection
        self.sparks_sigma = 3
        # Sigma values used for sample smoothing in sparks peaks detection in
        # dataset creation
        self.sparks_sigma_dataset = 2

        ## Waves (2) parameters ##
        # can insert other parameters here...

        ## Puffs (3) parameters ##
        # can insert other parameters here...

        ### Events detection parameters ###

        # Connectivity for event instances detection
        self.connectivity = 26
        # Maximal gap between two predicted puffs or waves that belong together
        # (in frames)
        self.max_gap = {
            "sparks": None,
            "waves": 2,
            "puffs": 2,
        }

        # Parameters for correspondence computation
        # (threshold for considering annotated and pred ROIs a match)
        self.iomin_t = 0.5

    def _set_sparks_connectivity_mask(self):
        """
        Create a mask defining the minimum distance between two spark peaks.

        Returns:
        - Connectivity mask representing the minimum distance.
        """
        radius_xy = math.ceil(self.min_dist_xy / 2)
        y, x = np.ogrid[-radius_xy : radius_xy + 1, -radius_xy : radius_xy + 1]
        disk_xy = x**2 + y**2 <= radius_xy**2
        self._conn_mask = np.stack([disk_xy] * self.min_dist_t, axis=0)

    @property
    def conn_mask(self):
        return self._conn_mask

    @property
    def min_dist_xy(self):
        # Minimum XY distance between spark peaks (to get sparks locations).
        return round(1.8 / self.pixel_size)

    @property
    def min_dist_t(self):
        # Minimum T (time) distance between spark peaks (to get sparks locations).
        return round(20 / self.time_frame)

    @property
    def num_classes(self):
        return len(self.event_types) + 1  # +1 for background

    @property
    def output_dir(self):
        return os.path.realpath(os.path.join(self.basedir, self._output_relative_dir))


# Initialize the configuration object
config = ProjectConfig()


class TrainingConfig:
    def __init__(self, training_config_file: str = ""):
        """
        Initialize the training configuration object.
        A configuration manager for loading settings from a configuration file,
        initializing parameters, and configuring WandB (Weights and Biases)
        logging, if wandb_project_name is not None.

        If training_config_file is not provided, create a TrainingConfig object
        where default values are used. These corresponds to the training
        configuration of config_final_model.ini.

        WandB logging is disabled by default. To enable it, provide
        training_config_file and set wandb_enable to True in the file.


        Parameters:
        training_config_file : str
            Path to the training configuration file.

        Attributes:
        TODO: add explanation of all attributes...

        params.norm_video:
            "chunk": Normalizing each chunk using min and max
            "movie": Normalizing whole video using min and max
            "abs_max": "Normalizing whole video using 16-bit absolute max"

        params.nn_architecture:
            "pablos_unet": classical 3D U-Net
            "github_unet": other 3D U-Net implementation from GitHub that has
                more options, such as batch normalization, attention, etc.
            "openai_unet": 3D U-Net implementation from OpenAI that has more
                options, such as residual blocks, attention, etc.

        params.temporal_reduction:
            If True, apply convolutional layers with stride 2 to reduce the
            temporal dimension prior to the U-Net (using TempRedUNet). Tried
            this to fit longer sequences in memory, but it didn't help much.

        params.num_channels:
            >0 if using temporal_reduction, value depends on temporal
            reduction configuration

        params.data_stride:
            Step between two chunks extracted from the sample

        params.data_duration:
            Duration of a chunk

        params.data_smoothing:
            If '2d' or '3d', preprocess movie with simple convolution
            (probably useless)

        params.remove_background:
            If 'moving' or 'average', remove background from input movies
            accordingly

        # params.only_sparks: # not used anymore
        #     If True, train using only sparks annotations

        params.sparks_type:
            Can be 'raw', 'peaks' (use smaller annotated ROIs), or 'dilated'
            (add ignore regions around events)

        params.ignore_frames_loss:
            If testing, used to ignore events in first and last frames

        """

        # Configure logging
        self.configure_logging()

        # Load configuration file
        self.training_config_file = self.dataset_dir = os.path.realpath(
            os.path.join(config.basedir, training_config_file)
        )
        self.load_configuration_file()

        # Load configuration parameters here...
        self.load_training_params()
        self.load_dataset_params()
        self.load_inference_params()
        self.load_unet_params()

        # Configure WandB
        self.configure_wandb()

        # Set the device to use for training
        self.set_device(device="params")

    def configure_logging(self):
        # Define a mapping of verbosity levels
        level_map = {
            3: logging.DEBUG,
            2: logging.INFO,
            1: logging.WARNING,
            0: logging.ERROR,
        }

        # Get the log level based on the configured verbosity
        log_level = level_map[config.verbosity]

        # Create a list of log handlers starting with stdout
        log_handlers = (logging.StreamHandler(sys.stdout),)

        # Configure file logging if a logfile is provided
        # (use this when project is finished)
        if config.logfile:
            self.configure_file_logging(log_handlers)

        # Configure the basic logging settings
        logging.basicConfig(
            level=log_level,
            format="[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}",
            style="{",
            datefmt="%H:%M:%S",
            handlers=log_handlers,
        )

    def configure_file_logging(self, log_handlers):
        if config.logfile:
            # Add file logging handler if a logfile is provided
            log_dir = os.path.basename(config.logfile)
            if not os.path.isdir(log_dir):
                logger.info("Creating parent directory for logs")
                os.mkdir(log_dir)

            if os.path.isdir(config.logfile):
                logfile_path = os.path.abspath(
                    os.path.join(config.logfile, f"{__name__}.log")
                )
            else:
                logfile_path = os.path.abspath(config.logfile)

            logger.info(f"Storing logs in {logfile_path}")
            file_handler = RotatingFileHandler(
                filename=logfile_path,
                maxBytes=(1024 * 1024 * 8),  # 8 MB
                backupCount=4,
            )
            log_handlers += (file_handler,)

    def load_configuration_file(self):
        # Initialize the ConfigParser
        self.c = ConfigParser()

        # Read the configuration file if it exists
        if os.path.isfile(self.training_config_file):
            logger.info(f"Loading {self.training_config_file}")
            self.c.read(self.training_config_file)
        else:
            logger.warning(
                f"No config file found at {self.training_config_file}, trying to use fallback values."
            )

    def load_training_params(self):
        # Create dict of fallback values
        fallback_training_section = {
            "run_name": "final_model",
            "load_run_name": "",
            "load_epoch": "0",
            "train_epochs": "5000",
            "criterion": "nll_loss",
            "lr_start": "1e-4",
            "ignore_frames_loss": "6",
            "gamma": "2.0",
            "w": "0.5",
            "cuda": "True",
            "scheduler": None,
            "scheduler_step_size": "0",
            "scheduler_gamma": "0.1",
            "optimizer": "adam",
        }

        # Load training parameters
        training_section = (
            self.c["training"] if "training" in self.c else fallback_training_section
        )

        self.run_name = training_section.get("run_name", "final_model")
        self.load_run_name = training_section.get("load_run_name", "")
        self.load_epoch = int(training_section.get("load_epoch", "0"))
        self.train_epochs = int(training_section.get("train_epochs", "5000"))
        self.criterion = training_section.get("criterion", "nll_loss")
        self.lr_start = float(training_section.get("lr_start", "1e-4"))
        self.ignore_frames_loss = int(training_section.get("ignore_frames_loss", "6"))
        if (self.criterion == "focal_loss") or (self.criterion == "sum_losses"):
            self.gamma = float(training_section.get("gamma", "2.0"))
        if self.criterion == "sum_losses":
            self.w = float(training_section.get("w", "0.5"))
        self.cuda = bool(training_section.get("cuda", "True"))
        self.scheduler = training_section.get("scheduler")
        if self.scheduler == "step":
            self.scheduler_step_size = int(training_section.get("step_size", "0"))
            self.scheduler_gamma = float(training_section.get("gamma", "0.1"))
        self.optimizer = training_section.get("optimizer", "adam")

    def load_dataset_params(self):
        # Create dict of fallback values
        fallback_dataset_section = {
            "relative_path": "data/sparks_dataset",
            "dataset_size": "full",
            "batch_size": "4",
            "num_workers": "1",
            "data_duration": "256",
            "data_stride": "32",
            "data_smoothing": "no",
            "norm_video": "abs_max",
            "remove_background": "no",
            # "only_sparks": "", # not used anymore
            "noise_data_augmentation": "",
            "sparks_type": "raw",
            "new_fps": "0",
        }

        # Load dataset parameters
        dataset_section = (
            self.c["dataset"] if "dataset" in self.c else fallback_dataset_section
        )

        self.dataset_dir = dataset_section.get("relative_path", "data/sparks_dataset")
        # get dataset's absolute path
        self.dataset_dir = os.path.realpath(
            os.path.join(config.basedir, self.dataset_dir)
        )
        if not os.path.isdir(self.dataset_dir):
            warnings.warn(
                f'Specified dataset path "{self.dataset_dir}" is not a directory, only inference with a provided movie path or an array is possible.'
            )
        self.dataset_size = dataset_section.get("dataset_size", "full")
        self.batch_size = int(dataset_section.get("batch_size", "4"))
        # self.num_workers = dataset_section.getint("num_workers", 1)
        self.num_workers = 0
        self.data_duration = int(dataset_section.get("data_duration", "256"))
        self.data_stride = int(dataset_section.get("data_stride", "32"))
        self.data_smoothing = dataset_section.get("data_smoothing", "no")
        self.norm_video = dataset_section.get("norm_video", "abs_max")
        self.remove_background = dataset_section.get("remove_background", "no")
        # self.only_sparks = dataset_section.getboolean(
        #     "only_sparks", ) # not used anymore
        self.noise_data_augmentation = bool(
            dataset_section.get("noise_data_augmentation", "")
        )
        self.sparks_type = dataset_section.get("sparks_type", "raw")
        self.new_fps = int(
            dataset_section.get("new_fps", "0")
        )  # can be implemented later

    def load_inference_params(self):
        fallback_inference_section = {
            "inference_data_duration": self.data_duration,
            "inference_data_stride": self.data_stride,
            "inference": "overlap",
            "inference_load_epoch": "100000",
            "inference_batch_size": "4",
            "inference_dataset_size": "full",
        }

        # Load inference parameters
        inference_section = (
            self.c["inference"] if "inference" in self.c else fallback_inference_section
        )

        self.inference_data_duration = int(
            inference_section.get("inference_data_duration", "256")
        )
        self.inference_data_stride = int(
            inference_section.get("inference_data_stride", "32")
        )
        self.inference = inference_section.get("inference", "overlap")
        assert self.inference in [
            "overlap",
            "average",
            "gaussian",
            "max",
        ], f"Inference type '{self.inference}' not supported yet."
        self.inference_load_epoch = int(
            inference_section.get("inference_load_epoch", "100000")
        )
        self.inference_batch_size = int(
            inference_section.get("inference_batch_size", "4")
        )
        self.inference_dataset_size = inference_section.get(
            "inference_dataset_size", "full"
        )

    def load_unet_params(self):
        # Create dict of fallback values
        fallback_network_section = {
            "nn_architecture": "pablos_unet",
            "unet_steps": "6",
            "first_layer_channels": "8",
            "num_channels": "1",
            "dilation": "1",
            "border_mode": "same",
            "batch_normalization": "none",
            "temporal_reduction": "",
            "initialize_weights": "",
            "attention": "",
            "up_mode": "transpose",
            "num_res_blocks": "1",
        }

        # Load UNet parameters
        network_section = (
            self.c["network"] if "network" in self.c else fallback_network_section
        )

        self.nn_architecture = network_section.get("nn_architecture", "pablos_unet")
        assert self.nn_architecture in [
            "pablos_unet",
            "github_unet",
            "openai_unet",
        ], f"nn_architecture must be one of 'pablos_unet', 'github_unet', 'openai_unet'"

        if self.nn_architecture == "unet_lstm":
            self.bidirectional = bool(network_section.get("bidirectional"))
        self.unet_steps = int(network_section.get("unet_steps", "6"))
        self.first_layer_channels = int(
            network_section.get("first_layer_channels", "8")
        )
        self.num_channels = int(network_section.get("num_channels", "1"))
        self.dilation = int(network_section.get("dilation", "1"))
        self.border_mode = network_section.get("border_mode", "same")
        self.batch_normalization = network_section.get("batch_normalization", "none")
        self.temporal_reduction = bool(network_section.get("temporal_reduction", ""))
        self.initialize_weights = bool(network_section.get("initialize_weights", ""))
        if self.nn_architecture == "github_unet":
            self.attention = bool(network_section.get("attention", ""))
            self.up_mode = network_section.get("up_mode", "transpose")
        if self.nn_architecture == "openai_unet":
            self.num_res_blocks = int(network_section.get("num_res_blocks", "1"))

    def initialize_wandb(self):
        # Only resume when loading the same saved model
        if self.load_epoch > 0 and self.load_run_name is None:
            resume = "must"
        else:
            resume = None

        wandb.init(
            project=config.wandb_project_name,
            notes=self.c.get("general", "wandb_notes", fallback=None),
            id=self.run_name,
            resume=resume,
            allow_val_change=True,
        )
        logging.getLogger("wandb").setLevel(logging.DEBUG)
        # wandb.save(CONFIG_FILE)

    def configure_wandb(self):
        if config.wandb_project_name is not None:
            self.wandb_log = self.c.getboolean(
                "general", "wandb_enable", fallback=False
            )
        else:
            self.wandb_log = False

        if self.wandb_log:
            self.initialize_wandb()

    def set_device(self, device: str = "auto"):
        # Set the device to use for training
        if device == "cuda":
            self.device = torch.device("cuda")
            self.pin_memory = True
        elif device == "cpu":
            self.device = torch.device("cpu")
            self.pin_memory = False
        elif device == "params":
            self.device = torch.device("cuda" if self.cuda else "cpu")
            self.pin_memory = True if self.device else False
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pin_memory = True if torch.cuda.is_available() else False
        else:
            raise ValueError(
                f"Device is {device} but is should be 'cuda', 'cpu', or 'auto."
            )

        self.n_gpus = torch.cuda.device_count()

    def display_device_info(self):
        if self.n_gpus > 1:
            logger.info(f"Using device '{self.device}' with {self.n_gpus} GPUs")
        else:
            logger.info(f"Using {self.device}")

    def print_params(self):
        for attribute, value in vars(self).items():
            logger.info(f"{attribute:>24s}: {value}")

            # Load parameters to wandb
            if self.wandb_log:
                if self.load_epoch == 0:
                    wandb.config[attribute] = value
                else:
                    wandb.config.update({attribute: value}, allow_val_change=True)

            # TODO: add all parameters that have to be printed here...
