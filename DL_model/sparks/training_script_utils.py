
from training_inference_tools import compute_class_weights
from torch import nn
from custom_losses import (
    FocalLoss,
    LovaszSoftmax3d,
    SumFocalLovasz,
    mySoftDiceLoss,
    LovaszSoftmax
)
from architectures import TempRedUNet, UNetConvLSTM
import unet_openai
from new_unet import UNet
from datasets import SparkDataset, SparkDatasetLSTM
import unet
from training_inference_tools import random_flip, random_flip_noise
import logging
import argparse
import configparser
import os
import sys

import wandb

logger = logging.getLogger(__name__)


def init_param_config_logs(
        basedir, logfile=None, verbosity=2, wandb_project_name=None,
        config_file=None, print_params=True):
    '''
    Load configuration file.
    Initialize parameters.
    Configure WandB as well. wandb_log is False if wandb_project_name is None.
    Print all parameters.

    Returns:
        c (configparser.ConfigParser): Configuration file.
        params (dict): Parameters.
        wandb_log (bool): Whether to log to WandB or not.
    '''

    # Configure logging

    level_map = {
        3: logging.DEBUG,
        2: logging.INFO,
        1: logging.WARNING,
        0: logging.ERROR,
    }
    log_level = level_map[verbosity]
    log_handlers = (logging.StreamHandler(sys.stdout),)

    # use this when project is finished:
    if logfile:
        if not os.path.isdir(os.path.basename(logfile)):
            logger.info("Creating parent directory for logs")
            os.mkdir(os.path.basename(logfile))

        if os.path.isdir(logfile):
            logfile_path = os.path.abspath(
                os.path.join(logfile, f"{__name__}.log"))
        else:
            logfile_path = os.path.abspath(logfile)

        logger.info(f"Storing logs in {logfile_path}")
        file_handler = logging.RotatingFileHandler(
            filename=logfile_path,
            maxBytes=(1024 * 1024 * 8),  # 8 MB
            backupCount=4,
        )
        log_handlers += (file_handler, )

    logging.basicConfig(
        level=log_level,
        format="[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}",
        style="{",
        datefmt="%H:%M:%S",
        handlers=log_handlers,
    )

    # Load configuration file
    if basedir is not None:
        parser = argparse.ArgumentParser("Spark & Puff detector using U-Net.")
        parser.add_argument(
            "config", type=str, help="Input config file, used to configure training"
        )
        args = parser.parse_args()

        CONFIG_FILE = os.path.join(basedir, "config_files", args.config)
    else:
        CONFIG_FILE = os.path.join("config_files", config_file)

    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"Loading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.warning(
            f"No config file found at {CONFIG_FILE}, trying to use fallback values."
        )

    params = {}

    # training params
    params["run_name"] = c.get(
        "training", "run_name", fallback="TEST")  # Run name
    params["load_run_name"] = c.get("training", "load_run_name", fallback=None)
    params["load_epoch"] = c.getint("training", "load_epoch", fallback=0)
    params["train_epochs"] = c.getint(
        "training", "train_epochs", fallback=5000)
    params["criterion"] = c.get("training", "criterion", fallback="nll_loss")
    params["lr_start"] = c.getfloat("training", "lr_start", fallback=1e-4)
    params["ignore_frames_loss"] = c.getint(
        "training", "ignore_frames_loss", fallback=0)
    if (params["criterion"] == "focal_loss") or (params["criterion"] == "sum_losses"):
        params["gamma"] = c.getfloat("training", "gamma", fallback=2.0)
    if params["criterion"] == "sum_losses":
        params["w"] = c.getfloat("training", "w", fallback=0.5)
    params["cuda"] = c.getboolean("training", "cuda")
    params["scheduler"] = c.get("training", "scheduler", fallback=None)
    if params["scheduler"] == "step":
        params["scheduler_step_size"] = c.getint("training", "step_size")
        params["scheduler_gamma"] = c.getfloat("training", "gamma")
    params["optimizer"] = c.get("training", "optimizer", fallback="adam")

    # dataset params
    params["relative_path"] = c.get("dataset", "relative_path")
    params["dataset_size"] = c.get("dataset", "dataset_size", fallback="full")
    params["batch_size"] = c.getint("dataset", "batch_size", fallback=1)
    params["num_workers"] = 0  # c.getint("dataset", "num_workers", fallback=1)
    params["data_duration"] = c.getint("dataset", "data_duration")
    params["data_step"] = c.getint("dataset", "data_step", fallback=1)
    params["testing_data_step"] = c.getint("testing", "data_step")
    params["data_smoothing"] = c.get(
        "dataset", "data_smoothing", fallback="2d")
    params["norm_video"] = c.get("dataset", "norm_video", fallback="chunk")
    params["remove_background"] = c.get(
        "dataset", "remove_background", fallback="average"
    )
    params["only_sparks"] = c.getboolean(
        "dataset", "only_sparks", fallback=False)
    params["noise_data_augmentation"] = c.getboolean(
        "dataset", "noise_data_augmentation", fallback=False
    )
    params["sparks_type"] = c.get("dataset", "sparks_type", fallback="peaks")
    params["inference"] = c.get("dataset", "inference", fallback="overlap")

    # UNet params
    params["nn_architecture"] = c.get(
        "network", "nn_architecture", fallback="pablos_unet"
    )
    if params["nn_architecture"] == "unet_lstm":
        params["bidirectional"] = c.getboolean("network", "bidirectional")
    params["unet_steps"] = c.getint("network", "unet_steps")
    params["first_layer_channels"] = c.getint(
        "network", "first_layer_channels")
    params["num_channels"] = c.getint("network", "num_channels", fallback=1)
    params["dilation"] = c.getboolean("network", "dilation", fallback=1)
    params["border_mode"] = c.get("network", "border_mode")
    params["batch_normalization"] = c.get(
        "network", "batch_normalization", fallback="none"
    )
    params["temporal_reduction"] = c.getboolean(
        "network", "temporal_reduction", fallback=False
    )
    params["initialize_weights"] = c.getboolean(
        "network", "initialize_weights", fallback=False
    )
    if params["nn_architecture"] == "github_unet":
        params["attention"] = c.getboolean("network", "attention")
        params["up_mode"] = c.get("network", "up_mode")
    if params["nn_architecture"] == "openai_unet":
        params["num_res_blocks"] = c.getint("network", "num_res_blocks")

    # config wandb
    if wandb_project_name is not None:
        wandb_log = c.getboolean("general", "wandb_enable", fallback=False)
    else:
        wandb_log = False

    if wandb_log:
        # only resume when loading the same saved model
        if params["load_epoch"] > 0 and params["load_run_name"] is None:
            resume = "must"
        else:
            resume = None

        wandb.init(
            project=wandb_project_name,
            # name=params["run_name"],
            notes=c.get("general", "wandb_notes", fallback=None),
            id=params["run_name"],
            resume=resume,
            allow_val_change=True
        )
        logging.getLogger("wandb").setLevel(logging.DEBUG)
        # wandb.save(CONFIG_FILE)

    # print all parameters
    if print_params:
        logger.info("Command parameters:")
        for k, v in params.items():
            logger.info(f"{k:>24s}: {v}")
            # load parameters to wandb
            if wandb_log:
                if params["load_epoch"] == 0:
                    wandb.config[k] = v
                else:
                    wandb.config.update({k: v}, allow_val_change=True)

            # TODO: AGGIUNGERE TUTTI I PARAMS NECESSARI DA PRINTARE

    return c, params, wandb_log


def init_training_dataset(params, train_sample_ids, ignore_index, dataset_path):
    # initialize training dataset
    assert os.path.isdir(dataset_path), f'"{dataset_path}" is not a directory'
    if params["nn_architecture"] in ['pablos_unet', 'github_unet', 'openai_unet']:
        dataset = SparkDataset(
            base_path=dataset_path,
            sample_ids=train_sample_ids,
            testing=False,
            smoothing=params["data_smoothing"],
            step=params["data_step"],
            duration=params["data_duration"],
            remove_background=params["remove_background"],
            temporal_reduction=params["temporal_reduction"],
            num_channels=params["num_channels"],
            normalize_video=params["norm_video"],
            only_sparks=params["only_sparks"],
            sparks_type=params["sparks_type"],
            ignore_index=ignore_index,
            inference=None,
        )
    elif params["nn_architecture"] == 'unet_lstm':
        dataset = SparkDatasetLSTM(
            base_path=dataset_path,
            sample_ids=train_sample_ids,
            testing=False,
            duration=params["data_duration"],
            smoothing=params["data_smoothing"],
            remove_background=params["remove_background"],
            temporal_reduction=params["temporal_reduction"],
            num_channels=params["num_channels"],
            normalize_video=params["norm_video"],
            only_sparks=params["only_sparks"],
            sparks_type=params["sparks_type"],
            ignore_index=ignore_index,
            inference=None
        )
    else:
        logger.error(
            f"{params['nn_architecture']} is not a valid nn architecture.")
        exit()

    # transforms are applied when getting a sample from the dataset
    if params["noise_data_augmentation"]:
        dataset = unet.TransformedDataset(dataset, random_flip_noise)
    else:
        dataset = unet.TransformedDataset(dataset, random_flip)

    return dataset


def init_testing_dataset(params, test_sample_ids, ignore_index, dataset_path):
    # initialize testing dataset

    # pattern_test_filenames = os.path.join(
    #     f"{dataset_path}", "videos_test", "[0-9][0-9]_video.tif"
    # ) # NON SERVE???

    if params["nn_architecture"] in ['pablos_unet', 'github_unet', 'openai_unet']:
        testing_datasets = [
            SparkDataset(
                base_path=dataset_path,
                sample_ids=[sample_id],
                testing=True,
                smoothing=params["data_smoothing"],
                step=params["testing_data_step"],
                duration=params["data_duration"],
                remove_background=params["remove_background"],
                temporal_reduction=params["temporal_reduction"],
                num_channels=params["num_channels"],
                normalize_video=params["norm_video"],
                only_sparks=params["only_sparks"],
                sparks_type=params["sparks_type"],
                ignore_frames=params["ignore_frames_loss"],
                ignore_index=ignore_index,
                inference=params["inference"],
            )
            for sample_id in test_sample_ids
        ]
    elif params["nn_architecture"] == 'unet_lstm':
        testing_datasets = [
            SparkDatasetLSTM(
                base_path=dataset_path,
                sample_ids=[sample_id],
                testing=True,
                duration=params["data_duration"],
                smoothing=params["data_smoothing"],
                remove_background=params["remove_background"],
                temporal_reduction=params["temporal_reduction"],
                num_channels=params["num_channels"],
                normalize_video=params["norm_video"],
                only_sparks=params["only_sparks"],
                sparks_type=params["sparks_type"],
                ignore_index=ignore_index,
                inference=params["inference"]
            )
            for sample_id in test_sample_ids
        ]
    else:
        logger.error(
            f"{params['nn_architecture']} is not a valid nn architecture.")
        exit()

    return testing_datasets


def init_model(params, num_classes, ndims):
    # initialize deep learning model
    if params["nn_architecture"] == "pablos_unet":

        batch_norm = {"batch": True, "none": False}

        unet_config = unet.UNetConfig(
            steps=params["unet_steps"],
            first_layer_channels=params["first_layer_channels"],
            num_classes=num_classes,
            ndims=ndims,
            dilation=params["dilation"],
            border_mode=params["border_mode"],
            batch_normalization=batch_norm[params["batch_normalization"]],
            num_input_channels=params["num_channels"],
        )

        if not params["temporal_reduction"]:
            network = unet.UNetClassifier(unet_config)
        else:
            assert (
                params["data_duration"] % params["num_channels"] == 0
            ), "using temporal reduction chunks_duration must be a multiple of num_channels"
            network = TempRedUNet(unet_config)

    elif params["nn_architecture"] == "github_unet":

        network = UNet(
            in_channels=params["num_channels"],
            out_channels=num_classes,
            n_blocks=params["unet_steps"] + 1,
            start_filts=params["first_layer_channels"],
            up_mode=params["up_mode"],
            # up_mode = 'transpose', # TESTARE DIVERSE POSSIBILTÀ
            # up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
            merge_mode="concat",  # Default, dicono che funziona meglio
            # planar_blocks=(0,), # magari capire cos'è e testarlo ??
            activation="relu",
            normalization=params[
                "batch_normalization"
            ],  # Penso che nell'implementazione di Pablo è 'none'
            attention=params["attention"],  # magari da testare con 'True' ??
            # full_norm=False,  # Uncomment to restore old sparse normalization scheme
            dim=ndims,
            # 'valid' ha dei vantaggi a quanto pare...
            conv_mode=params["border_mode"],
        )

    elif params["nn_architecture"] == "unet_lstm":

        batch_norm = {"batch": True, "none": False}
        ndims = 2  # convolutions applied to single frames

        unet_config = unet.UNetConfig(
            steps=params["unet_steps"],
            first_layer_channels=params["first_layer_channels"],
            num_classes=num_classes,
            ndims=ndims,
            dilation=params["dilation"],
            border_mode=params["border_mode"],
            batch_normalization=batch_norm[params["batch_normalization"]],
            # num_input_channels=params["data_duration"], # frames seen as modalities
            num_input_channels=params["num_channels"]
        )

        network = UNetConvLSTM(
            unet_config, bidirectional=params["bidirectional"])

    elif params["nn_architecture"] == "openai_unet":
        if params["unet_steps"] == 4:
            channel_mult = (1, 2, 4, 8)
        elif params["unet_steps"] == 3:
            channel_mult = (1, 2, 4)
        elif params["unet_steps"] == 2:
            channel_mult = (1, 2)
        else:
            logger.error(
                f"{params['unet_steps']} is not a valid number of unet steps.")
            exit()

        network = unet_openai.UNetModel(
            # image_size=x.shape[1:],
            in_channels=params["num_channels"],
            model_channels=params["first_layer_channels"],
            out_channels=num_classes,
            channel_mult=channel_mult,
            num_res_blocks=params["num_res_blocks"],
            attention_resolutions=[],
            # attention_resolutions=[8], # bottom layer
            dropout=0.0,
            dims=ndims,
            # use_checkpoint=True
        )

    return network


def init_criterion(params, dataset, ignore_index, device):
    # initialize loss function

    # class weights
    if params["criterion"] in ["nll_loss", "focal_loss", "sum_losses"]:
        class_weights = compute_class_weights(dataset)
        logger.info(
            "Using class weights: {}".format(
                ", ".join(str(w.item()) for w in class_weights)
            )
        )

    if params["criterion"] == "nll_loss":
        criterion = nn.NLLLoss(
            ignore_index=ignore_index, weight=class_weights.to(
                device, non_blocking=True)
        )
    elif params["criterion"] == "focal_loss":
        criterion = FocalLoss(
            reduction="mean",
            ignore_index=ignore_index,
            alpha=class_weights,
            gamma=params["gamma"],
        )
    elif params["criterion"] == "lovasz_softmax":
        if params["nn_architecture"] in ['pablos_unet', 'github_unet', 'openai_unet']:
            criterion = LovaszSoftmax3d(
                classes="present", per_image=False, ignore=ignore_index
            )
        elif params["nn_architecture"] == 'unet_lstm':
            criterion = LovaszSoftmax(
                classes="present", per_image=True, ignore=ignore_index
            )
    elif params["criterion"] == "sum_losses":
        criterion = SumFocalLovasz(
            classes="present",
            per_image=False,
            ignore=ignore_index,
            alpha=class_weights,
            gamma=params["gamma"],
            reduction="mean",
            w=params["w"],
        )

    elif params["criterion"] == "dice_loss":
        softmax = nn.Softmax(dim=1)
        criterion = mySoftDiceLoss(apply_nonlin=softmax,
                                   batch_dice=True,
                                   do_bg=False)

    else:
        logger.error(
            f"{params['criterion']} is not a valid criterion.")
        exit()

    return criterion
