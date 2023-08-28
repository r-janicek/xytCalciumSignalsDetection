"""
Functions that are used during the training of the neural network
"""

import logging
import math
import os
import time

import numpy as np
import torch
import wandb
from data_processing_tools import (
    class_to_nb,
    empty_marginal_frames,
    get_event_instances_class,
    get_processed_result
)
from in_out_tools import write_videos_on_disk
from metrics_tools import (
    compute_iou,
    get_matches_summary,
    get_score_matrix,
)
from scipy import ndimage as ndi
from torch import nn
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from metrics_tools import get_metrics_from_summary
import unet
from unet.trainer import _write_results

logger = logging.getLogger(__name__)


########################### custom training manager ############################


class myTrainingManager(unet.TrainingManager):

    def run_validation(self, wandb_log=False):

        if self.test_function is None:
            return

        logger.info("Validating network at iteration {}...".format(self.iter))

        test_output = self.test_function(self.iter)

        if wandb_log:
            wandb.log({m: val for m, val in test_output.items()
                       if 'confusion_matrix' not in m}, step=self.iter)
            # wandb.log({'Step_val': self.iter}, step=self.iter)

        logger.info("Metrics:")
        for metric_name, metric_value in test_output.items():
            if 'confusion_matrix' in metric_name:
                with np.printoptions(precision=0, suppress=True):
                    logger.info(f"\t{metric_name}:\n{metric_value:}")
            else:
                logger.info(f"\t{metric_name}: {metric_value:.4g}")

        # if "loss" in test_output:
        #     logger.info("\tValidation loss: {:.4g}".format(
        #         test_output["loss"]))

        _write_results(self.summary, "testing", test_output, self.iter)

    def train(self, num_iters, print_every=0, maxtime=np.inf, wandb_log=False):

        tic = time.process_time()
        time_elapsed = 0

        if wandb_log:
            loss_sum = 0

        for _ in range(num_iters):

            step_output = self.training_step(self.iter)

            if wandb_log:
                loss_sum += step_output["loss"]

            time_elapsed = time.process_time() - tic

            # logger.info(info)
            if print_every and self.iter % print_every == 0:
                # Move loss value to cpu
                step_output["loss"] = step_output["loss"].item()

                # Register data
                _write_results(self.summary, "training",
                               step_output, self.iter)

                # Check for nan
                loss = step_output["loss"]
                if np.any(np.isnan(loss)):
                    logger.error("Last loss is nan! Training diverged!")
                    break

                # Log training loss
                logger.info("Iteration {}...".format(self.iter))
                logger.info("\tTraining loss: {:.4g}".format(loss))
                logger.info("\tTime elapsed: {:.2f}s".format(time_elapsed))

                # Log to wandb
                if wandb_log:
                    wandb.log({"U-Net training loss": (loss_sum.item() / print_every if self.iter > 0 else loss_sum.item()),
                               # "Step_": self.iter
                               },
                              step=self.iter)
                    loss_sum = 0

            self.iter += 1

            # Validation
            if self.test_every and self.iter % self.test_every == 0:
                self.run_validation(wandb_log=wandb_log)

            # Plot
            if self.plot_function and self.plot_every and self.iter % self.plot_every == 0:
                self.plot_function(self.iter, self.summary)

            # Save model and solver
            if self.save_every and self.iter % self.save_every == 0:
                self.save()

            if time_elapsed > maxtime:
                logger.info("Maximum time reached!")
                break


################################ training step #################################

# Make one step of the training (update parameters and compute loss)


def training_step(
    sampler,
    network,
    optimizer,
    # scaler,
    scheduler,
    device,
    criterion,
    dataset_loader,
    ignore_frames,
):
    network.train()

    x, y = sampler(dataset_loader)
    x = x.to(device, non_blocking=True)  # [b, d, 64, 512]
    y = y.to(device, non_blocking=True)  # [b, d, 64, 512] or [b, 64, 512]

    # detect nan in tensors
    # if (torch.isnan(x).any() or torch.isnan(y).any()):
    #    logger.info(f"Detect nan in network input: {torch.isnan(x).any()}")
    #    logger.info(f"Detect nan in network annotation: {torch.isnan(y).any()}")

    # Forward pass with mixed precision
    # with torch.cuda.amp.autocast():  # autocast as a context manager
    y_pred = network(x[:, None])  # [b, 4, d, 64, 512] or [b, 4, 64, 512]

    # Compute loss
    if ignore_frames != 0:
        pred = y_pred[:, :, ignore_frames:-ignore_frames]
        target = y[:, ignore_frames:-ignore_frames]
    else:
        pred = y_pred
        target = y
    # if isinstance(criterion, SoftDiceLoss):
    if type(criterion).__name__ == "mySoftDiceLoss":
        # set regions in pred where label is ignored to 0
        pred = pred * (target != 4)
        target = target * (target != 4)
    else:
        target = target.long()

    loss = criterion(pred, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    if scheduler is not None:
        lr = scheduler.get_last_lr()[0]
        logger.debug(f"Current learning rate: {lr}")
        scheduler.step()

        new_lr = scheduler.get_last_lr()[0]
        if new_lr != lr:
            logger.info(f"Learning rate changed to {new_lr}")

    return {"loss": loss}


# Iterator (?) over the dataset
def mycycle(dataset_loader):
    while True:
        for x in dataset_loader:
            yield x


# _cycle = mycycle(dataset_loader)


def sampler(dataset_loader):
    return next(mycycle(dataset_loader))  # (_cycle)


### functions for data augmentation ###


def random_flip(x, y):
    # flip movie and annotation mask
    # if np.random.uniform() > 0.5:
    rand = torch.rand(1).item()

    if torch.rand(1).item() > 0.5:
        x = x.flip(-1)
        y = y.flip(-1)

    if torch.rand(1).item() > 0.5:
        x = x.flip(-2)
        y = y.flip(-2)

    return x, y


def random_flip_noise(x, y):
    # flip movie and annotation mask
    x, y = random_flip(x, y)

    # add noise to movie with a 50% chance
    if torch.rand(1).item() > 0.5:
        # 50/50 of being normal or Poisson noise
        if torch.rand(1).item() > 0.5:
            noise = torch.normal(mean=0.0, std=1.0, size=x.shape)
            x = x + noise
        else:
            x = torch.poisson(x)  # non so se funziona !!!

        # x = x.astype('float32')

    # denoise input with a 50% chance
    if torch.rand(1).item() > 0.5:
        # 50/50 of gaussian filtering or median filtering
        if torch.rand(1).item() > 0.5:
            x = ndi.gaussian_filter(x, sigma=1)
        else:
            x = ndi.median_filter(x, size=2)

    return torch.as_tensor(x), torch.as_tensor(y)


### functions related to UNet and loss weights ###


def compute_class_weights(dataset, w0=1, w1=1, w2=1, w3=1):
    # For 4 classes
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    with torch.no_grad():
        for _, y in dataset:
            count0 += torch.count_nonzero(y == 0)
            count1 += torch.count_nonzero(y == 1)
            count2 += torch.count_nonzero(y == 2)
            count3 += torch.count_nonzero(y == 3)

    total = count0 + count1 + count2 + count3

    w0_new = w0 * total / (4 * count0) if count0 != 0 else 0
    w1_new = w1 * total / (4 * count1) if count1 != 0 else 0
    w2_new = w2 * total / (4 * count2) if count2 != 0 else 0
    w3_new = w3 * total / (4 * count3) if count3 != 0 else 0

    weights = torch.as_tensor([w0_new, w1_new, w2_new, w3_new])
    return weights


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        stdv = np.sqrt(2 / m.weight.size(1))
        m.weight.data.normal_(m.weight, std=stdv)


############################### inference tools ################################


def detect_nan_sample(x, y):
    # detect nan in tensors
    if torch.isnan(x).any() or torch.isnan(y).any():
        logger.warning(
            f"Detect nan in network input (test): " "{torch.isnan(x).any()}"
        )
        logger.warning(
            f"Detect nan in network annotation "
            "(test): {torch.isnan(y).any()}"
        )


def gaussian(n_chunks):
    """ Return the normalized Gaussian with standard deviation sigma = 2. """

    sigma = 3
    x = np.linspace(-10, 10, n_chunks)

    c = np.sqrt(2 * np.pi)
    res = np.exp(-0.5 * (x / sigma)**2) / sigma / c
    return torch.as_tensor(res / res.sum())


@torch.no_grad()
def do_inference(
    network, test_dataset, test_dataloader, device,
    detect_nan=False, compute_loss=False, inference_types=None
):
    '''
    From a given test sample (i.e., a test dataset) and a network model,
    combine all outputs from chunks for all movie frames.
    If inference_types is None, use the inference type defined in the test
    dataset. Otherwise, use the inference types provided in the list.

    Returns:
        preds:      predictions (4 x movie duration x 64 x 512) if
                    if inference_types is None, otherwise returns a dictionary
                    with inference type as key and predictions as value
    '''

    chunk_idx = 0

    if inference_types is None:
        inference_types = [test_dataset.inference]

    # initialize dictionary with predictions
    preds = {i: [] for i in inference_types}

    if 'overlap' in inference_types:
        # this is the default inference type used in training,
        # if one of the others works better, I will probably remove
        # this one

        assert (
            test_dataset.duration - test_dataset.step
        ) % 2 == 0, "(duration-step) is not even in overlap inference"
        half_overlap = (test_dataset.duration - test_dataset.step) // 2
        # to re-build videos from chunks

        # adapt half_overlap duration if using temporal reduction
        if test_dataset.temporal_reduction:
            assert half_overlap % test_dataset.num_channels == 0, (
                "with temporal reduction half_overlap must be "
                "a multiple of num_channels"
            )

            half_overlap_mask = half_overlap // test_dataset.num_channels
        else:
            half_overlap_mask = half_overlap

        preds['overlap'] = []
        n_chunks = len(test_dataset)

    if 'average' in inference_types or 'max' in inference_types or 'gaussian' in inference_types:
        movie_duration = test_dataset.data[0].shape[0]

        # initialize dict with list of predictions for each frame
        output_frames = {idx: [] for idx in range(movie_duration)}
        chunks = test_dataset.get_chunks(test_dataset.lengths[0],
                                         test_dataset.step,
                                         test_dataset.duration
                                         )

    for x, y in test_dataloader:

        if detect_nan:
            detect_nan_sample(x, y)

        # torch.cuda.FloatTensor, b x d x 64 x 512
        x = x.to(device, non_blocking=True)
        batch_preds = (network(x[:, None]).cpu())
        # b x 4 x d x 64 x 512 with 3D-UNet
        # b x 4 x 64 x 512 with LSTM-UNet -> not implemented yet

        if not compute_loss:
            # if computing loss, need logit values, otherwise compute
            # probabilities because it reduces the errors due to floating
            # point precision
            batch_preds = torch.exp(batch_preds)  # convert to probabilities

        for pred in batch_preds:

            if 'overlap' in inference_types:
                # define start and end of used frames in chunks
                # inference type does not matter for x and y
                start_mask = 0 if chunk_idx == 0 else half_overlap_mask
                end_mask = None if chunk_idx + 1 == n_chunks else -half_overlap_mask

                if pred.ndim == 4:
                    preds['overlap'].append(pred[:, start_mask:end_mask])
                else:
                    preds['overlap'].append(pred[:, None])

            if 'average' in inference_types or 'max' in inference_types or 'gaussian' in inference_types:
                # list of movie frames ids in the chunk:
                chunk_frames = chunks[chunk_idx].tolist()

                for idx, frame_idx in enumerate(chunk_frames):
                    # idx is the index of the frame in the chunk
                    # frame_idx is the index of the frame in the movie
                    output_frames[frame_idx].append(pred[:, idx])

            chunk_idx += 1

    if 'overlap' in inference_types:
        # concatenated predictions for a single video:
        preds['overlap'] = torch.cat(preds['overlap'], dim=1)

    if 'average' in inference_types or 'max' in inference_types or 'gaussian' in inference_types:
        # combine predictions from all chunks for each frame

        if 'average' in inference_types:
            # average predictions from all chunks
            preds['average'] = [torch.stack(p).mean(dim=0)
                                for p in output_frames.values()]

        if 'max' in inference_types:
            # for each pixel, keep 4-classes prediction that contains the
            # highest probability

            preds['max'] = []
            for p in output_frames.values():
                p = torch.stack(p)

                # get max probability for each pixel
                max_ids = p.max(dim=1)[0].max(dim=0)[1]
                # view as list of pixels and expand to frame size
                max_ids = max_ids.view(-1)[None, None,
                                           :].expand(p.size(0), p.size(1), -1)

                # view p as list of pixels and gather predictions for each
                # pixel according to chunk with highest probability
                preds['max'].append(p.view(p.size(0), p.size(1), -1).gather(
                                    dim=0, index=max_ids).view(*p.size())[0])

        if 'gaussian' in inference_types:
            # combine predictions using a gaussian function
            preds['gaussian'] = [(torch.stack(p) * gaussian(len(p)).view(-1, 1, 1, 1)).sum(dim=0)
                                 for p in output_frames.values()]

    for i, p in preds.items():
        if i != 'overlap':
            p = torch.stack(p)
            p = p.swapaxes(0, 1)
            preds[i] = p

    if len(inference_types) == 1:
        preds = preds[inference_types[0]]

    return preds


# function to run a test sample (i.e., a test dataset) in the UNet
@torch.no_grad()
def get_preds(
    network, test_dataset, compute_loss, device,
    criterion=None, detect_nan=False, test_dataloader=None, batch_size=None,
    inference_types=None
):
    '''
    Given a trained model, a test sample (i.e., a test dataset) and a device,
    run the sample in the model and return the predictions.
    If compute_loss is True, also return the loss. In this case, a criterion
    must be provided.
    If detect_nan is True, detect nan in the input and annotation tensors.
    If test_dataloader is None, create a dataloader with the given batch_size
    (which must be provided in this case).
    If inference_types is None, use the inference type defined in the test
    dataset. Otherwise, use the inference types provided in the list.

    Returns:
        xs:         original movie (movie duration x 64 x 512)
        ys:         original annotations (movie duration x 64 x 512)
        preds:      predictions (4 x movie duration x 64 x 512) if 
                    if inference_types is None, otherwise returns a dictionary
                    with inference type as key and predictions as value
        loss:       loss value if compute_loss is True
    '''

    # check if function parameters are correct
    if compute_loss:
        assert criterion is not None, "provide criterion if computing loss"

    if test_dataloader is None:
        assert batch_size is not None, "provide batch_size if no dataloader is provided"

    if inference_types is None:
        assert test_dataset.inference in ['overlap', 'average', 'gaussian', 'max'], \
            f"inference type '{i}' not implemented yet"
    else:
        for i in inference_types:
            assert i in ['overlap', 'average', 'gaussian', 'max'], \
                f"inference type '{i}' not implemented yet"

    if inference_types is None:
        inference_types = [test_dataset.inference]

    # create a dataloader if not provided
    if test_dataloader is None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # run movie in network and perform inference
    # preds are probabilities between 0 and 1
    # if len(inference_types) == 1, preds is a tensor of shape
    # 4 x movie duration x 64 x 512
    # otherwise, preds is a dictionary with inference type as key and
    # predictions as value

    preds = do_inference(
        network=network,
        test_dataset=test_dataset,
        test_dataloader=test_dataloader,
        device=device,
        detect_nan=detect_nan,
        compute_loss=compute_loss,
        inference_types=inference_types
    )

    # get original movie xs and annotations ys
    xs = test_dataset.data[0]
    ys = test_dataset.annotations[0]

    # remove padded frames
    if test_dataset.pad != 0:
        start_pad = test_dataset.pad // 2
        end_pad = -(test_dataset.pad // 2 + test_dataset.pad % 2)

        xs = xs[start_pad:end_pad]

        if test_dataset.temporal_reduction:
            start_pad = start_pad // test_dataset.num_channels
            end_pad = end_pad // test_dataset.num_channels

        if ys.ndim == 3:
            ys = ys[start_pad:end_pad]
            if len(inference_types) == 1:
                preds = preds[:, start_pad:end_pad]
            else:
                preds = {i: p[:, start_pad:end_pad] for i, p in preds.items()}

    # If original sample was shorter than current movie duration, remove
    # additional padded frames
    if test_dataset.movie_duration < xs.shape[0]:
        pad = xs.shape[0] - test_dataset.movie_duration
        start_pad = pad // 2
        end_pad = -(pad // 2 + pad % 2)

        xs = xs[start_pad:end_pad]

        if test_dataset.temporal_reduction:
            start_pad = start_pad // test_dataset.num_channels
            end_pad = end_pad // test_dataset.num_channels

        ys = ys[start_pad:end_pad]
        if len(inference_types) == 1:
            preds = preds[:, start_pad:end_pad]
        else:
            preds = {i: p[:, start_pad:end_pad] for i, p in preds.items()}

    if compute_loss:  # Compute loss
        # ys is torch.ByteTensor
        # ys = ys[None]  # .to(device, non_blocking=True)

        # if ys.ndim == 4:
        if ys.ndim == 3:
            if len(inference_types) == 1:
                preds_loss = preds[:, test_dataset.ignore_frames:-
                                   test_dataset.ignore_frames]
            else:
                raise NotImplementedError
                # still need to adapt code to compute loss for list of inference
                # types, however usually loss shoulb be computed only during
                # training, and therefore inference_types should be None

            ys_loss = ys[test_dataset.ignore_frames:-
                         test_dataset.ignore_frames]
        else:
            # not implemented yet
            raise NotImplementedError
            # output = pred
            # target = y

        if type(criterion).__name__ == "mySoftDiceLoss":
            # set regions in pred where label is ignored to 0
            preds_loss = preds_loss * (ys_loss != 4)
            ys_loss = ys_loss * (ys_loss != 4)
        else:
            ys_loss = ys_loss.long()[None, :]
            preds_loss = preds_loss[None, :]

        loss = criterion(preds_loss, ys_loss).item()

        return xs.numpy(), ys.numpy(), preds.numpy(), loss

    else:
        if len(inference_types) == 1:
            preds = preds.numpy()
        else:
            preds = {i: p.numpy() for i, p in preds.items()}

        return xs.numpy(), ys.numpy(), preds


def run_samples_in_model(network, device, datasets):
    """
    Process al movies in the UNet and get all predictions and movies as numpy
    arrays.
    """
    network.eval()

    if hasattr(datasets[0], "video_name"):
        xs_all_videos = {}
        ys_all_videos = {}
        preds_all_videos = {}
    else:
        xs_all_videos = []
        ys_all_videos = []
        preds_all_videos = []

    for test_dataset in datasets:

        # run sample in UNet
        xs, ys, preds = get_preds(
            network=network,
            test_dataset=test_dataset,
            compute_loss=False,
            device=device,
        )

        # if dataset has video_name attribute, store results as dictionaries
        if hasattr(test_dataset, "video_name"):
            xs_all_videos[test_dataset.video_name] = xs
            ys_all_videos[test_dataset.video_name] = ys
            preds_all_videos[test_dataset.video_name] = preds
        else:
            xs_all_videos.append(xs)
            ys_all_videos.append(ys)
            preds_all_videos.append(preds)

    return xs_all_videos, ys_all_videos, preds_all_videos


################################ test function #################################


def test_function(
    network,
    device,
    criterion,
    ignore_frames,
    testing_datasets,
    training_name,
    output_dir,
    batch_size,
    training_mode=True,
    debug=False
):
    r"""
    Validate UNet during training.
    Output segmentation is computed using argmax values and Otsu threshold (to
    remove artifacts & avoid using thresholds).
    Removing small events prior metrics computation as well.

    network:            the model being trained
    device:             current device
    criterion:          loss function to be computed on the validation set
    testing_datasets:   list of SparkDataset instances
    ignore_frames:      frames ignored by the loss function
    training_name:      training name used to save predictions on disk
    output_dir:         directory where the predicted movies are saved
    training_mode:      bool, if True, separate events using a simpler algorithm
    """

    network.eval()

    # initialize dicts that will contain the results
    tot_preds = {'sparks': 0, 'puffs': 0, 'waves': 0}
    tp_preds = {'sparks': 0, 'puffs': 0, 'waves': 0}
    ignored_preds = {'sparks': 0, 'puffs': 0, 'waves': 0}
    unlabeled_preds = {'sparks': 0, 'puffs': 0, 'waves': 0}
    tot_ys = {'sparks': 0, 'puffs': 0, 'waves': 0}
    tp_ys = {'sparks': 0, 'puffs': 0, 'waves': 0}
    undetected_ys = {'sparks': 0, 'puffs': 0, 'waves': 0}

    # initialize class attributes that are shared by all datasets

    ca_release_events = ["sparks", "puffs", "waves"]

    # sparks_type = testing_datasets[0].sparks_type
    # temporal_reduction = testing_datasets[0].temporal_reduction
    # if temporal_reduction:
    #     num_channels = testing_datasets[0].num_channels

    # spark instances detection parameters
    min_dist_xy = testing_datasets[0].min_dist_xy
    min_dist_t = testing_datasets[0].min_dist_t
    radius = math.ceil(min_dist_xy / 2)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    disk = x**2 + y**2 <= radius**2
    conn_mask = np.stack([disk] * (min_dist_t), axis=0)

    # TODO: use better parameters !!!
    pixel_size = 0.2
    spark_min_width = 3
    spark_min_t = 3
    puff_min_t = 5
    wave_min_width = round(15 / pixel_size)

    sigma = 3

    # connectivity for event instances detection
    connectivity = 26

    # maximal gap between two predicted puffs or waves that belong together
    max_gap = 2  # i.e., 2 empty frames

    # parameters for correspondence computation
    # threshold for considering annotated and pred ROIs a match
    iomin_t = 0.5

    # compute loss on all samples
    sum_loss = 0.0

    # concatenate annotations and preds to compute segmentation-based metrics
    ys_concat = []
    preds_concat = []

    # define dicts to count number of annotated and pred events per class
    # n_ys_per_class = {'sparks': 0, 'puffs': 0, 'waves': 0}
    # n_preds_per_class = {'sparks': 0, 'puffs': 0, 'waves': 0}

    # define dict to count number of false negatives per class
    # n_fn_per_class = {'sparks': 0, 'puffs': 0, 'waves': 0}

    for test_dataset in testing_datasets:

        # if debug:
        #     # count number of annotated and pred event in each video
        #     n_ys_temp = {'sparks': 0, 'puffs': 0, 'waves': 0}
        #     n_preds_temp = {'sparks': 0, 'puffs': 0, 'waves': 0}

        ########################## run sample in UNet ##########################

        start = time.time()

        # get video name
        video_name = test_dataset.video_name

        logger.debug(f"Testing function: running sample {video_name} in UNet")

        # run sample in UNet
        xs, ys, preds, loss = get_preds(
            network=network,
            test_dataset=test_dataset,
            compute_loss=True,
            device=device,
            criterion=criterion,
            batch_size=batch_size,
            detect_nan=False,
        )
        # preds is a list [background, sparks, waves, puffs]

        # sum up losses of each sample
        sum_loss += loss

        # compute exp of predictions
        preds = np.exp(preds)

        # save preds as videos
        write_videos_on_disk(
            xs=xs,
            ys=ys,
            preds=preds,
            training_name=training_name,
            video_name=video_name,
            path=output_dir,
        )

        logger.debug(
            f"Time to run sample {video_name} in UNet: {time.time() - start:.2f} s"
        )

        ####################### re-organise annotations ########################

        start = time.time()
        logger.debug("Testing function: re-organising annotations")

        # ys_instances is a dict with classified event instances, for each class
        ys_instances = get_event_instances_class(
            event_instances=test_dataset.events, class_labels=ys, shift_ids=True
        )

        # remove ignored events entry from ys_instances
        ys_instances.pop("ignore", None)

        # get annotations as a dictionary
        # ys_classes = {
        #     "sparks": np.where(ys == 1, 1, 0),
        #     "puffs": np.where(ys == 3, 1, 0),
        #     "waves": np.where(ys == 2, 1, 0),
        # }

        # get pixels labelled with 4
        # TODO: togliere ignored frames ??????????????????????????????
        ignore_mask = np.where(ys == 4, 1, 0)

        # get number of annotated events per class
        # for ca_event in ca_release_events:
        #     n_ys_per_class[ca_event] += (
        #         len(np.unique(ys_instances[ca_event]))-1)

        # if debug:
        #    n_ys_temp[ca_event] += (len(np.unique(ys_instances[ca_event]))-1)

        logger.debug(
            f"Time to re-organise annotations: {time.time() - start:.2f} s"
        )

        ######################### get processed output #########################

        logger.debug(
            "Testing function: getting processed output (segmentation and instances)")

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
            training_mode=training_mode,
            debug=debug
        )

        # get number of predicted events per class
        # for ca_event in ca_release_events:
        #     # n_preds_per_class[ca_event] += (
        #     #    len(np.unique(preds_instances[ca_event]))-1)

        #     if debug:
        #         n_preds_temp[ca_event] += (
        #             len(np.unique(preds_instances[ca_event]))-1)

        # logger.debug(
        #     f"Number of predicted events in video {video_name}:\n{n_preds_temp}")
        # logger.debug(
        #     f"Number of annotated events in video {video_name}:\n{n_ys_temp}")

        ##################### stack ys and segmented preds #####################

        # stack annotations and remove marginal frames
        ys_concat.append(empty_marginal_frames(
            video=ys, n_frames=ignore_frames))

        # stack preds and remove marginal frames
        temp_preds = np.zeros_like(ys)
        for event_type in ca_release_events:
            class_id = class_to_nb(event_type)
            temp_preds += class_id * preds_segmentation[event_type]
        preds_concat.append(
            empty_marginal_frames(video=temp_preds, n_frames=ignore_frames)
        )

        logger.debug(
            f"Time to process predictions: {time.time() - start:.2f} s")

        ############### compute pairwise scores (based on IoMin) ###############

        start = time.time()

        if debug:
            n_ys_events = max(
                [np.max(ys_instances[event_type])
                 for event_type in ca_release_events]
            )

            n_preds_events = max(
                [np.max(preds_instances[event_type])
                 for event_type in ca_release_events]
            )
            logger.debug(
                f"Testing function: computing pairwise scores between {n_ys_events} annotated events and {n_preds_events} predicted events")

        iomin_scores = get_score_matrix(
            ys_instances=ys_instances,
            preds_instances=preds_instances,
            ignore_mask=None,
            score="iomin",
        )

        logger.debug(
            f"Time to compute pairwise scores: {time.time() - start:.2f} s")

        ####################### get matches summary #######################

        start = time.time()

        logger.debug("Testing function: getting matches summary")

        matched_ys_ids, matched_preds_ids = get_matches_summary(
            ys_instances=ys_instances,
            preds_instances=preds_instances,
            scores=iomin_scores,
            t=iomin_t,
            ignore_mask=ignore_mask,
        )

        # count number of categorized events that are necessary for the metrics
        for ca_event in ca_release_events:
            tot_preds[ca_event] += len(matched_preds_ids[ca_event]['tot'])
            tp_preds[ca_event] += len(matched_preds_ids[ca_event]['tp'])
            ignored_preds[ca_event] += len(
                matched_preds_ids[ca_event]['ignored'])
            unlabeled_preds[ca_event] += len(
                matched_preds_ids[ca_event]['unlabeled'])

            tot_ys[ca_event] += len(matched_ys_ids[ca_event]['tot'])
            tp_ys[ca_event] += len(matched_ys_ids[ca_event]['tp'])
            undetected_ys[ca_event] += len(matched_ys_ids[ca_event]
                                           ['undetected'])

        # confusion matrix cols and rows indices: {background, sparks, puffs, waves}
        # confusion_matrix[video_name] = confusion_matrix_res[0]

        # dict indexed by predicted event indices, s.t. each entry is the
        # list of annotated event ids that match the predicted event
        # matched_events[video_name] = confusion_matrix_res[1]

        # count number of predicted events that are ignored per class
        # ignored_preds[video_name] = confusion_matrix_res[2]

        # get false negative (i.e., labelled but not detected in the correct class)
        # fn_events = confusion_matrix_res[3]
        # for event_type in ca_release_events:
        #    n_fn_per_class[event_type] += len(fn_events[event_type])

        # get undetected events(i.e., labelled and not detected in any class)
        # undetected_events = confusion_matrix_res[4]

        # logger.debug(
        #    f"Confusion matrix for video {video_name}:\n{confusion_matrix[video_name]}"
        # )

        logger.debug(
            f"Time to get matches summary: {time.time() - start:.2f} s")

    ############################## reduce metrics ##############################

    start = time.time()

    logger.debug("Testing function: reducing metrics")

    metrics = {}

    # Compute average validation loss
    metrics["validation_loss"] = sum_loss / len(testing_datasets)
    # logger.info(f"\tvalidation loss: {loss:.4g}")

    ##################### compute instances-based metrics ######################

    """
    Metrics that can be computed (event instances):
    - Confusion matrix
    - Precision & recall (TODO)
    - F-score (e.g. beta = 0.5,1,2) (TODO)
    (- Matthews correlation coefficient (MCC))??? (TODO)
    """

    # get confusion matrix of all summed events
    # metrics["events_confusion_matrix"] = sum(confusion_matrix.values())

    # get other metrics (precision, recall, % correctly classified, % detected)
    metrics_all = get_metrics_from_summary(tot_preds=tot_preds,
                                           tp_preds=tp_preds,
                                           ignored_preds=ignored_preds,
                                           unlabeled_preds=unlabeled_preds,
                                           tot_ys=tot_ys,
                                           tp_ys=tp_ys,
                                           undetected_ys=undetected_ys)

    metrics.update(metrics_all)

    #################### compute segmentation-based metrics ####################

    """
    Metrics that can be computed (raw sparks, puffs, waves):
    - Jaccard index (IoU)
    - Dice score (TODO)
    - Precision & recall (TODO)
    - F-score (e.g. beta = 0.5,1,2) (TODO)
    - Accuracy (biased since background is predominant) (TODO)
    - Matthews correlation coefficient (MCC) (TODO)
    - Confusion matrix
    """

    # concatenate annotations and preds
    ys_concat = np.concatenate(ys_concat, axis=0)
    preds_concat = np.concatenate(preds_concat, axis=0)

    # concatenate ignore masks
    ignore_concat = ys_concat == 4

    for event_type in ca_release_events:
        class_id = class_to_nb(event_type)
        class_preds = preds_concat == class_id
        class_ys = ys_concat == class_id

        metrics["segmentation/" + event_type + "_IoU"] = compute_iou(
            ys_roi=class_ys, preds_roi=class_preds, ignore_mask=ignore_concat
        )

    # get average IoU across all classes
    metrics["segmentation/average_IoU"] = np.mean(
        [metrics["segmentation/" + event_type + "_IoU"]
         for event_type in ca_release_events]
    )

    # compute confusion matrix
    metrics["segmentation_confusion_matrix"] = sk_confusion_matrix(
        y_true=ys_concat.flatten(), y_pred=preds_concat.flatten(), labels=[0, 1, 2, 3]
    )

    logger.debug(f"Time to reduce metrics: {time.time() - start:.2f} s")

    return metrics
