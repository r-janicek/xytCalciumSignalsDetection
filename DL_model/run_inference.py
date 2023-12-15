"""
Load a saved UNet model at given epochs and save its predictions in the
folder `C:project_basedir/evaluation/inference_script/run_name/`.

Predictions are saved as:
`{training_name}_{epoch}_{video_id}_{class}.tif`

**Idea**: Use predictions to produce plots and tables to visualize the
          results.

Author: Prisca Dotti
Last modified: 21.10.2023
"""

import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import TrainingConfig, config
from data.data_processing_tools import masks_to_instances_dict, process_raw_predictions
from utils.in_out_tools import write_videos_on_disk
from utils.training_inference_tools import do_inference
from utils.training_script_utils import get_sample_ids, init_dataset, init_model

logger = logging.getLogger(__name__)


def main():
    config.verbosity = 3  # To get debug messages

    ##################### Get training-specific parameters #####################

    run_name = "final_model"
    # run_name = "TEMP_new_annotated_peaks_physio"  # (if run on laptop)
    config_filename = "config_final_model.ini"
    load_epoch = 100000

    use_train_data = False
    get_final_pred = True  # Set to False to only compute raw predictions
    custom_ids = []
    # custom_ids = ["05"] # Override sample_ids if needed

    testing = True  # Set to False to only generate U-Net predictions
    # Set to True to also compute processed outputs and metrics
    # inference_types = ['overlap', 'average', 'gaussian', 'max']
    inference_types = []  # Set to empty to use the default inference type from
    # the config file

    # Initialize general parameters
    params = TrainingConfig(
        training_config_file=os.path.join("config_files", config_filename)
    )

    if run_name:
        params.run_name = run_name
    model_filename = f"network_{load_epoch:06d}.pth"

    # Print parameters to console if needed
    # params.print_params()

    if testing:
        get_final_pred = True

    debug = True if config.verbosity == 3 else False

    ######################### Configure output folder ##########################

    output_folder = os.path.join(
        "evaluation", "inference_script"
    )  # Same folder for train and test preds
    os.makedirs(output_folder, exist_ok=True)

    # Subdirectory of output_folder where predictions are saved.
    # Change this to save results for same model with different inference
    # approaches.
    # output_name = training_name + "_step=2"
    output_name = params.run_name

    save_folder = os.path.join(config.basedir, output_folder, output_name)
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"Annotations and predictions will be saved on '{save_folder}'.")

    ######################### Detect GPU, if available #########################

    params.set_device(device="auto")
    params.display_device_info()

    ###################### Config dataset and UNet model #######################

    logger.info(f"Processing training '{params.run_name}'...")

    # Define the sample IDs based on dataset size and usage
    sample_ids = get_sample_ids(
        train_data=use_train_data,
        dataset_size=params.dataset_size,
        custom_ids=custom_ids,
    )
    logger.info(f"Predicting outputs for samples {sample_ids}")

    logger.info(f"Using {params.dataset_dir} as dataset root path")

    # Create dataset
    dataset = init_dataset(
        params=params,
        sample_ids=sample_ids,
        apply_data_augmentation=False,
        print_dataset_info=True,
        load_instances=testing,
    )

    # Create a dataloader
    dataset_loader = DataLoader(
        dataset,
        batch_size=params.inference_batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
    )
    ### Configure UNet ###

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
    logger.info(f"Loading trained model '{run_name}' at epoch {load_epoch}...")
    try:
        network.load_state_dict(torch.load(model_dir, map_location=params.device))
    except RuntimeError as e:
        if "module" in str(e):
            # The error message contains "module," so handle the DataParallel loading
            logger.warning(
                "Failed to load the model, as it was trained with DataParallel. Wrapping it in DataParallel and retrying..."
            )
            # Get current device of the object (model)
            temp_device = next(iter(network.parameters())).device

            network = nn.DataParallel(network)
            network.load_state_dict(torch.load(model_dir, map_location=params.device))

            logger.info("Network should be on CPU, removing DataParallel wrapper...")
            network = network.module.to(temp_device)
        else:
            # Handle other exceptions or re-raise the exception if it's unrelated
            raise

    ########################### Run samples in UNet ############################

    if len(inference_types) == 0:
        inference_types = [params.inference]

    # get U-Net's raw predictions
    network.eval()
    raw_preds = do_inference(
        network=network,
        params=params,
        dataloader=dataset_loader,
        device=params.device,
        compute_loss=False,
        inference_types=inference_types,
    )

    ############# Get movies and labels (and instances if testing) #############

    xs = dataset.get_movies()
    ys = dataset.get_labels()

    if testing:
        ys_instances = dataset.get_instances()

        # convert instance masks to dictionaries
        ys_instances = {
            i: masks_to_instances_dict(
                instances_mask=instances_mask,
                labels_mask=ys[i],
                shift_ids=True,
            )
            for i, instances_mask in ys_instances.items()
        }

        # remove ignored events entry from ys_instances
        for inference in ys_instances:
            ys_instances[inference].pop("ignore", None)

    #################### Get processed output (if required) ####################

    if get_final_pred:
        logger.debug("Getting processed output (segmentation and instances)")

        final_segmentation_dict = {}
        final_instances_dict = {}
        for i in range(len(sample_ids)):
            movie_segmentation = {}
            movie_instances = {}

            for inference in inference_types:
                # transform raw predictions into a dictionary
                raw_preds_dict = {
                    event_type: raw_preds[i][inference][event_label]
                    for event_type, event_label in config.classes_dict.items()
                    if event_type in config.event_types
                }

                preds_instances, preds_segmentation, _ = process_raw_predictions(
                    raw_preds_dict=raw_preds_dict,
                    input_movie=xs[i],
                    training_mode=False,
                    debug=debug,
                )

                movie_segmentation[inference] = preds_segmentation
                movie_instances[inference] = preds_instances

            final_segmentation_dict[sample_ids[i]] = movie_segmentation
            final_instances_dict[sample_ids[i]] = movie_instances

    else:
        final_segmentation_dict = {}
        final_instances_dict = {}

    ############################ Save preds on disk ############################

    logger.info(f"\tSaving annotations and predictions...")

    for i, sample_id in enumerate(sample_ids):
        for inference in inference_types:
            video_name = f"{str(params.load_epoch)}_{sample_id}_{inference}"

            raw_preds_movie = raw_preds[i][inference]
            if get_final_pred:
                segmented_preds_movie = final_segmentation_dict[sample_id][inference]
                instances_preds_movie = final_instances_dict[sample_id][inference]
            else:
                segmented_preds_movie = None
                instances_preds_movie = None

            write_videos_on_disk(
                training_name=output_name,
                video_name=video_name,
                out_dir=os.path.join(save_folder, "inference_" + inference),
                # xs=xs[i], # xs is available elsewhere
                # ys=ys[i], # ys is available elsewhere
                raw_preds=raw_preds_movie,
                segmented_preds=segmented_preds_movie,
                instances_preds=instances_preds_movie,
            )

    logger.info(f"DONE")


if __name__ == "__main__":
    main()
