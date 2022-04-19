#!/usr/bin/env python3

import sys
import os
import logging
import logging.handlers
import argparse
import pathlib
import configparser
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import imageio

# append whole ML folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
###

from UNet import unet
from sparks.datasets import SparkTestDataset

BASEDIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(BASEDIR, "config.ini")

logger = logging.getLogger(__name__)
# disable most wandb logging, since we don't care about it here
logging.getLogger('wandb').setLevel(logging.ERROR)


def get_dataset(input_files: List[str], gt_available: bool) -> Dataset:
    """Returns a fully configured dataset for spark detection for a given input video."""
    assert len(input_files) == 1, f"More than one input provided: {len(input_files)}"
    input_file = pathlib.Path(input_files[0])
    name_for_ds, ext = os.path.splitext(input_file.name)

    dataset = SparkTestDataset(
        video_path=input_file,
        smoothing='2d',
        step=c.getint("data", "step"),
        duration=c.getint("data", "chunks_duration"),
        remove_background=c.getboolean("data", "remove_background"),
        gt_available=gt_available
    )

    logger.info(f"Loaded dataset with {len(dataset)} samples")

    return dataset


def get_network(device: torch.device, state_file: str = None) -> nn.Module:
    """Returns a fully configured UNet, optionally with loaded weights from a state."""
    unet_config = unet.UNetConfig(
        steps=c.getint("data", "step"),
        num_classes=c.getint("network", "num_classes"),
        ndims=c.getint("network", "ndims"),
        border_mode=c.get("network", "border_mode"),
        batch_normalization=c.getboolean("network", "batch_normalization")
    )
    unet_config.feature_map_shapes((c.getint("data", "chunks_duration"), 64, 512))
    network = nn.DataParallel(unet.UNetClassifier(unet_config)).to(device)

    if state_file:
        # load model state
        if not os.path.isfile(state_file):
            raise RuntimeError(f"State is not a file: {state_file}")

        logger.info(f"Loading state from {state_file}")
        network.load_state_dict(torch.load(state_file, map_location=device))
    else:
        logger.warning("Not loading network state")
    return network


# we don't care about gradients, and retaining them explodes the memory usage
@torch.no_grad()
def predict(network: nn.Module, dataset: Dataset, gt_available: bool,
            quick: bool = False) -> torch.Tensor:
    """Run a network against a dataset and produce the predictions, optionally massively shortened by the quick parameter."""

    # Since the dataset acts more like a generator than a list, we can't yield the first item
    # without breaking access to other elements. Thus, the container must be initialized later
    predictions = None

    # retain some contextual information (in the form of previous and following frames)
    # by slicing results according to a frame overlap
    overlap = (dataset.duration - dataset.step) // 2
    assert overlap % 2 == 0, f"Frame overlap must be an even number; is {overlap}"

    # TODO: have this use a dataloader
    # dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    for i, chunk in enumerate(dataset):

        if gt_available:
            chunk, labels = chunk

        if args.quick and 0 < i < (len(dataset) - 1):
            # skipping all but the first and last frame
            continue

        logger.debug(f"Sample {i+1:>3}/{len(dataset):>3}")
        if predictions is None:
            # initialize mask container with the size from the first chunk
            predictions = torch.zeros(
                (
                    (1 + 1 + 4) if gt_available else (1 + 4),       # input (+ labels) + number of classes
                    len(dataset) * dataset.step + 2 * overlap,      # time, i.e. every video frame
                    *(chunk.shape[1:])                              # the actual frames, height x width
                ), dtype=torch.float32, device=device)
            logger.debug(f"Created prediction container of shape {predictions.shape}")

        # for every frame, cut the predictions into the range [-overlap, overlap], except for
        # - the first item (start at the first frame)
        # - the last item (end at the last frame)
        # additionally, keep track of the absolute locations of the data in the input video
        if i == 0:
            logger.debug(f"First sample, adding extra {overlap} frames")
            slice_start = 0
            slice_end = -overlap
            abs_start = 0
            abs_end = dataset.step + overlap
        else:
            slice_start = overlap
            slice_end = -overlap
            abs_start = i * dataset.step + overlap
            abs_end = abs_start + dataset.step
            if i == len(dataset) - 1:
                logger.debug(f"Last sample, adding extra {overlap} frames")
                slice_end = None  # slice to and including the end
                abs_end += overlap

        logger.debug(
            f"Slicing results: [{slice_start}:{slice_end}] "
            f"mapping to time frames [{abs_start:>5}:{abs_end:<5}]")

        chunk_t = torch.Tensor(chunk).to(device)
        if gt_available:
            labels_t = torch.Tensor(labels).to(device)

        prediction = network(chunk_t[None, None])

        # remove empty first dim
        # note that predictions still have one more dim than data & labels!
        prediction_squeeze = torch.squeeze(prediction, dim=0)
        data_squeeze = torch.squeeze(chunk_t, dim=0)
        if gt_available:
            label_squeeze = torch.squeeze(labels_t, dim=0)

        # this line is slightly different from ...[:, start:end] - I couldn't find a way to ask for a slice
        # which would have actually included the last element in the standard [start:end] notation.
        # instead, slice(start, end) does what I need by accepting None
        prediction_slice = prediction_squeeze[:, slice(slice_start, slice_end)]
        data_slice = data_squeeze[slice(slice_start, slice_end)]
        if gt_available:
            label_slice = label_squeeze[slice(slice_start, slice_end)]

        time_length = prediction_slice.shape[1]

        logger.debug(
            f"Mangling prediction {prediction.shape} into sliced {prediction_slice.shape}, "
            f"spanning {time_length} time units")

        # poor man's unittest to assert correct slicing of video chunks
        assert time_length == (abs_end - abs_start)
        if i == 0 or i == len(dataset) - 1:
            # first and last entries are longer
            assert time_length == dataset.step + overlap
        else:
            assert time_length == dataset.step

        predictions[0, abs_start:abs_end, :, :] = data_slice
        if gt_available:
            predictions[1, abs_start:abs_end, :, :] = label_slice
            predictions[2:, abs_start:abs_end, :, :] = prediction_slice
        else:
            predictions[1:, abs_start:abs_end, :, :] = prediction_slice

    return predictions


def parse_predictions(predictions: torch.Tensor, desc: str = "", output: str = "./output") -> None:
    """Work on the predictions and do whatever is required to make them human-parseable."""

    logger.info(f"Generating output files in {os.path.abspath(output)}")

    assert predictions.size()[0] == (5 or 6), "Prediction shape is wrong"

    predictions_exp = torch.exp(predictions).detach().cpu().numpy()

    # produce a singleton output, containing all outputs stacked, i.e. stacked along the vertical pixel axis
    # since .cat() requires a collection of tensors, we create that using torch's views along the first dim;
    # and finally, we have to remove the empty (one-dimensional) first dimension by squeeze()
    predictions_stacked = np.concatenate(predictions_exp, axis=1).squeeze()
    # fn = os.path.join(output, f"{desc}_stacked.tif")
    fn = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        output,
        f"{desc}_stacked.tif"
        )
    
    logger.debug(f"Prediction has shape {predictions.shape}, stack along output axis has shape {predictions_stacked.shape}")
 
    imageio.volwrite(fn, predictions_stacked)

    # produce an output file per output
    categories = (
        ("input", "labels", "background", "sparks", "waves", "puffs")
        if predictions.size()[0] == 6
        else ("input", "background", "sparks", "waves", "puffs"))

    for class_label, data in zip(categories, predictions_exp):
        # fn = os.path.join(output, f"{desc}_{class_label}.tif")
        fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), output, f"{desc}_{class_label}.tif")
        logger.debug(f"Building {fn} from data in shape {data.shape}")
        imageio.volwrite(fn, data)


if __name__ == "__main__":

    c = configparser.ConfigParser()
    if os.path.isfile(CONFIG_FILE):
        logger.info(f"Loading {CONFIG_FILE}")
        c.read(CONFIG_FILE)
    else:
        logger.warning(f"No config file found at {CONFIG_FILE}, trying to use fallback values.")

    parser = argparse.ArgumentParser("Spark & Puff detector using U-Net.")

    parser.add_argument(
        '-v-', '--verbosity',
        type=int,
        default=c.getint("general", "verbosity", fallback="0"),
        help="Set verbosity level ([0, 3])"
    )

    parser.add_argument(
        '--logfile',
        type=str,
        default=None,
        help="In addition to stdout, store all logs in this file"
    )

    parser.add_argument(
        '-s', '--state',
        type=str,
        default=os.path.join(
                    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
                    c.get("state", "file", fallback=None)
                    ),
        help="Load pretrained model from this file"
    )
    """
    # parser.add_argument(
        '-s', '--state',
        type=str,
        default=c.get("state", "file", fallback=None),
        help="Load pretrained model from this file"
    )
    """
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=c.get("general", "output", fallback="output"),
        help="Move output files to this directory"
    )

    """
    # TODO: Determine how this affects output
    parser.add_argument(
        '--fps',
        type=float,
        default=145.0,
        help=""
    )
    """

    parser.add_argument(
        'input',
        type=str,
        nargs=1,
        help="Input video file, used as input for the model to produce predictions"
    )

    parser.add_argument(
        '-q', '--quick',
        action="store_true",
        default=False,
        help="Run on as little data as possible while still touching as much functionality as possible for debugging."
    )

    parser.add_argument(
        '-gt', '--gt_available',
        action="store_true",
        default=c.get("predict", "gt_available", fallback=False),
        help="Compare predictions with given gt_available"
    )

    args = parser.parse_args()

    level_map = {3: logging.DEBUG, 2: logging.INFO, 1: logging.WARNING, 0: logging.ERROR}
    log_level = level_map[args.verbosity]
    log_handlers = (logging.StreamHandler(sys.stdout), )

    if args.logfile:
        if not os.path.isdir(os.path.basename(args.logfile)):
            logger.info("Creating parent directory for logs")
            os.mkdir(os.path.basename(args.logfile))

        if os.path.isdir(args.logfile):
            logfile_path = os.path.abspath(os.path.join(args.logfile, f"{__name__}.log"))
        else:
            logfile_path = os.path.abspath(args.logfile)

        logger.info(f"Storing logs in {logfile_path}")
        file_handler = logging.handlers.RotatingFileHandler(
            filename=logfile_path,
            maxBytes=(1024 * 1024 * 8),  # 8 MB
            backupCount=4,
        )
        log_handlers += (file_handler, )

    logging.basicConfig(
        level=log_level,
        format='[{asctime}] [{levelname:^8s}] [{name:^12s}] <{lineno:^4d}> -- {message:s}',
        style='{',
        datefmt="%H:%M:%S",
        handlers=log_handlers)

    logger.info("Command parameters:")
    for k, v in vars(args).items():
        logger.info(f"{k:>18s}: {v}")

    # detect CUDA devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    logger.info(f"Using torch device {device}, with {n_gpus} GPUs")

    # extract a part of the input file name to better describe the output.
    input_desc, _ = os.path.splitext(os.path.basename(args.input[0]))

    dataset = get_dataset(args.input, args.gt_available)
    network = get_network(device, args.state)

    # feed data to the network
    network.eval()
    predictions = predict(network, dataset, args.gt_available, args.quick)
    print(args.output)
    parse_predictions(predictions, desc=input_desc, output=args.output)
