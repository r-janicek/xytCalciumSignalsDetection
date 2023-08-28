import logging
import os
import random

import numpy as np
import torch
import wandb
from torch import nn, optim
# from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training_inference_tools import (
    sampler,
    test_function,
    training_step,
    weights_init,
    myTrainingManager
)
from training_script_utils import (
    init_param_config_logs,
    init_training_dataset,
    init_testing_dataset,
    init_model,
    init_criterion
)

import unet

if __name__ == "__main__":
    BASEDIR = os.path.dirname(os.path.realpath(__file__))
    logger = logging.getLogger(__name__)
    torch.set_float32_matmul_precision('high')

    ############################# fixed parameters #############################

    # General params
    logfile = None  # change this when publishing finished project on github
    verbosity = 2
    debug_mode = False
    wandb_project_name = "sparks2"  # use new wandb project name with new test_function
    # wandb_project_name = "TEST"  # use new wandb project name with new test_function
    output_relative_path = "runs"  # directory where output, saved params and
    # testing results are saved

    # Dataset parameters
    ignore_index = 4  # label ignored during training
    num_classes = 4  # i.e., BG, sparks, waves, puffs
    ndims = 3  # using 3D data

    ############################## get parameters ##############################

    c, params, wandb_log = init_param_config_logs(
        BASEDIR, logfile, verbosity, wandb_project_name)

    ############################ init random seeds #############################

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    ############################ configure datasets ############################

    # select samples that are used for training and testing
    if params["dataset_size"] == "full":
        train_sample_ids = ["01", "02", "03", "04", "06", "07", "08", "09",
                            "11", "12", "13", "14", "16", "17", "18", "19",
                            "21", "22", "23", "24", "27", "28", "29", "30",
                            "33", "35", "36", "38", "39", "41", "42", "43",
                            "44", "46"]
        test_sample_ids = ["05", "10", "15",
                           "20", "25", "32", "34", "40", "45"]
    elif params["dataset_size"] == "minimal":
        train_sample_ids = ["01"]
        test_sample_ids = ["34"]
    else:
        logger.error(f"{params['dataset_size']} is not a valid dataset size.")
        exit()

    # detect CUDA devices
    if params["cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pin_memory = True
    else:
        device = "cpu"
        pin_memory = False
    n_gpus = torch.cuda.device_count()
    logger.info(f"Using torch device {device}, with {n_gpus} GPUs")

    # set if temporal reduction is used
    if params["temporal_reduction"]:
        logger.info(
            f"Using temporal reduction with {params['num_channels']} channels")

    # normalize whole videos or chunks individually
    if params["norm_video"] == "chunk":
        logger.info("Normalizing each chunk using min and max")
    elif params["norm_video"] == "movie":
        logger.info("Normalizing whole video using min and max")
    elif params["norm_video"] == "abs_max":
        logger.info("Normalizing whole video using 16-bit absolute max")

    dataset_path = os.path.realpath(f"{BASEDIR}/{params['relative_path']}")

    # initialize training dataset
    dataset = init_training_dataset(
        params=params,
        train_sample_ids=train_sample_ids,
        ignore_index=ignore_index,
        dataset_path=dataset_path
    )

    logger.info(f"Samples in training dataset: {len(dataset)}")

    # initialize testing dataset
    testing_datasets = init_testing_dataset(
        params=params,
        test_sample_ids=test_sample_ids,
        ignore_index=ignore_index,
        dataset_path=dataset_path
    )

    for i, tds in enumerate(testing_datasets):
        logger.info(f"Testing dataset {i} contains {len(tds)} samples")

    # initialize data loaders
    dataset_loader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=pin_memory,
    )

    ############################## configure UNet ##############################

    network = init_model(params=params, num_classes=num_classes, ndims=ndims)

    if device != "cpu":
        network = nn.DataParallel(network).to(device, non_blocking=True)
        torch.backends.cudnn.benchmark = True

    if wandb_log:
        wandb.watch(network)

    if params["initialize_weights"]:
        logger.info("Initializing UNet weights...")
        network.apply(weights_init)

    # torch.compile(network, mode="default", backend="inductor")
    # does not work on windows

    ########################### initialize training ############################

    if params["optimizer"] == "adam":
        optimizer = optim.Adam(network.parameters(), lr=params["lr_start"])
    elif params["optimizer"] == "adadelta":
        optimizer = optim.Adadelta(network.parameters(), lr=params["lr_start"])
    else:
        logger.error(f"{params['optimizer']} is not a valid optimizer.")
        exit()

    if params["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params["scheduler_step_size"],
            gamma=params["scheduler_gamma"],
        )
    else:
        scheduler = None

    network.train()

    output_path = os.path.join(output_relative_path, params["run_name"])
    logger.info(f"Output directory: {output_path}")

    summary_writer = SummaryWriter(os.path.join(
        output_path, "summary"), purge_step=0)

    if params["load_run_name"] != None:
        load_path = os.path.join(output_relative_path, params["load_run_name"])
        logger.info(f"Model loaded from directory: {load_path}")
    else:
        load_path = None

    # initialize loss function
    criterion = init_criterion(
        params=params,
        dataset=dataset,
        ignore_index=ignore_index,
        device=device
    )

    # directory where predicted class movies are saved
    preds_output_dir = os.path.join(output_path, "predictions")
    os.makedirs(preds_output_dir, exist_ok=True)

    # generate dict of managed objects
    managed_objects = {"network": network, "optimizer": optimizer}
    if scheduler is not None:
        managed_objects["scheduler"] = scheduler

    trainer = myTrainingManager(
        # training items
        training_step=lambda _: training_step(
            sampler=sampler,
            network=network,
            optimizer=optimizer,
            # scaler=GradScaler(),
            scheduler=scheduler,
            device=device,
            criterion=criterion,
            dataset_loader=dataset_loader,
            ignore_frames=params["ignore_frames_loss"],
        ),
        save_every=c.getint("training", "save_every", fallback=5000),
        load_path=load_path,
        save_path=output_path,
        managed_objects=unet.managed_objects(managed_objects),
        # testing items
        test_function=lambda _: test_function(
            network=network,
            device=device,
            criterion=criterion,
            testing_datasets=testing_datasets,
            ignore_frames=params["ignore_frames_loss"],
            training_name=params["run_name"],
            output_dir=preds_output_dir,
            batch_size=params["batch_size"],
            training_mode=True,
            debug=debug_mode,
        ),
        test_every=c.getint("training", "test_every", fallback=1000),
        plot_every=c.getint("training", "test_every", fallback=1000),
        summary_writer=summary_writer,
    )

    ############################## start training ##############################

    if params["load_epoch"] != 0:
        trainer.load(params["load_epoch"])

    # resume wandb run
    # if wandb.run.resumed:
    #     checkpoint = torch.load(wandb.restore(checkpoint_path))
    #     network.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']

    if c.getboolean("general", "training", fallback=False):  # Run training procedure on data
        if params["load_epoch"] > 0:
            logger.info("Validate network before training")
            trainer.run_validation(wandb_log=wandb_log)
        logger.info("Starting training")
        trainer.train(
            params["train_epochs"],
            print_every=c.getint("training", "print_every", fallback=100),
            wandb_log=wandb_log
        )

    if c.getboolean("general", "testing", fallback=False):  # Run final validation
        logger.info("Starting final validation")
        trainer.run_validation(wandb_log=wandb_log)
