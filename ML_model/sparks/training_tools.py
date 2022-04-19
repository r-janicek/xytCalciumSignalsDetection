'''
Functions that are used during the training of the neural network
'''

import os
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import imageio
import pandas as pd
from scipy import ndimage as ndi

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

#torch.__file__
import unet
#from focal_losses import focal_loss

from metrics_tools import *


import wandb



__all__ = ["training_step",
           "test_function",
           "validate_2d_unet",
           "mycycle",
           "sampler",
           "store_results_video",
           "store_results",
           #"get_preds_coords",
           #"get_labels_coords"
           ]


# Make one step of the training (update parameters and compute loss)
def training_step(sampler, network, optimizer, device, criterion,
                  dataset_loader, ignore_frames, wandb_log):
    #start = time.time()

    network.train()

    x, y = sampler(dataset_loader)
    x = x.to(device)
    y = y.to(device)

    y_pred = network(x[:, None])

    #Compute loss
    loss = criterion(y_pred[:,:,ignore_frames:-ignore_frames],
                     y[:,ignore_frames:-ignore_frames].long())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if wandb_log:
        wandb.log({"U-Net training loss": loss.item()})

    #end = time.time()
    #print(f"Runtime for 1 training step: {end-start}")

    return {"loss": loss.item()}



def training_step_new(network, optimizer, device, criterion,
                     dataset_loader, ignore_frames, summary_writer,
                     wandb_log=True):
    #start = time.time()

    network.train()
    running_loss = 0.0

    for x,y in dataset_loader:
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)
        y_pred = network(x[:, None])

        #Compute loss

        batch_loss = criterion(y_pred[:,:,ignore_frames:-ignore_frames],
                               y[:,ignore_frames:-ignore_frames].long())

        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item()
        summary_writer.add_scalar("training/batch_loss",
                                  batch_loss.item(), global_step=None)

        if wandb_log:
            wandb.log({"U-Net batch training loss": batch_loss.item()})

    running_loss /= len(dataset_loader)
    if wandb_log:
        wandb.log({"U-Net epoch training loss": running_loss})

    #end = time.time()
    #print(f"Runtime for 1 training step: {end-start}")
    #exit()

    return {"loss": running_loss}



# Compute some metrics on the predictions of the network
def test_function(network, device, criterion, testing_datasets, logger,
                  summary_writer, thresholds, idx_fixed_threshold,
                  ignore_frames, wandb_log):
    # Requires a list of testing dataset as input
    # (every test video has its own dataset)
    # TODO: implementation not finished (using test_function_fixed_t temporarily)
    network.eval()

    duration = testing_datasets[0].duration
    step = testing_datasets[0].step
    half_overlap = (duration-step)//2 # to re-build videos from chunks

    # (duration-step) has to be even
    assert (duration-step)%2 == 0, "(duration-step) is not even"


    loss = 0.0
    metrics = [] # store metrics for each video

    for test_dataset in testing_datasets:
        xs = []
        ys = []
        preds = []

        with torch.no_grad():
            if (len(test_dataset)>1):
                x,y = test_dataset[0]
                xs.append(x[:-half_overlap])
                ys.append(y[:-half_overlap])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                pred = network(x[None, None])

                loss += criterion(pred[:,:,:-half_overlap],
                                 y[:,:-half_overlap].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,:-half_overlap])

                for i in range(1,len(test_dataset)-1):
                    x,y = test_dataset[i]
                    xs.append(x[half_overlap:-half_overlap])
                    ys.append(y[half_overlap:-half_overlap])

                    x = torch.Tensor(x).to(device)
                    y = torch.Tensor(y[None]).to(device)
                    pred = network(x[None, None])

                    loss += criterion(pred[:,:,half_overlap:-half_overlap],
                                     y[:,half_overlap:-half_overlap].long())

                    pred = pred[0].cpu().numpy()
                    preds.append(pred[:,half_overlap:-half_overlap])

                x,y = test_dataset[-1]
                xs.append(x[half_overlap:])
                ys.append(y[half_overlap:])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                pred = network(x[None, None])

                loss += criterion(pred[:,:,half_overlap:],
                                 y[:,half_overlap:].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,half_overlap:])
            else:
                x,y = test_dataset[0]
                xs.append(x)
                ys.append(y)

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                pred = network(x[None, None])

                loss += criterion(pred, y.long())

                pred = pred[0].cpu().numpy()
                preds.append(pred)

        # concatenated frames and predictions for a single video:
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=1)

        if test_dataset.pad != 0:
            xs = xs[:-test_dataset.pad]
            ys = ys[:-test_dataset.pad]
            preds = preds[:,:-test_dataset.pad]

        # predictions have logarithmic values

        # save preds as videos
        #write_videos_on_disk(xs,ys,preds,test_dataset.video_name)


        # compute predicted sparks and correspondences
        sparks = np.exp(preds[1])
        #sparks = sparks[ignore_frames:-ignore_frames]
        #sparks = np.pad(sparks,((ignore_frames,),(0,),(0,)), mode='constant')
        sparks = empty_marginal_frames(sparks, ignore_frames)

        #min_radius = 3 # minimal "radius" of a valid event

        #coords_preds = process_spark_prediction(sparks,
        #                                        t_detection=(threshold),
        #                                        min_radius=min_radius)

        #sparks_mask_true = ys[ignore_frames:-ignore_frames]
        #sparks_mask_true = np.pad(sparks_mask_true,((ignore_frames,),(0,),(0,)),
        #                          mode='constant')
        sparks_mask_true = empty_marginal_frames(ys, ignore_frames)
        sparks_mask_true = np.where(sparks_mask_true==1, 1.0, 0.0)


        #coords_true = nonmaxima_suppression(sparks_mask_true)

        #metrics.append(Metrics(*correspondences_precision_recall(coords_true,
        #                                    coords_preds, match_distance=6)))


        metrics.append(compute_prec_rec(sparks_mask_true, sparks, thresholds))


    # Compute average validation loss
    loss = loss.item()
    loss /= len(testing_datasets)

    # Compute metrics comparing ys and y_preds
    _, precs, recs, a_u_c = reduce_metrics_thresholds(metrics)

    prec = precs[idx_fixed_threshold]
    rec = recs[idx_fixed_threshold]
    logger.info("\tPrecision: {:.4g}".format(prec))
    logger.info("\tRecall: {:.4g}".format(rec))
    #logger.info("\tArea under the curve: {:.4g}".format(a_u_c))
    logger.info("\tValidation loss: {:.4g}".format(loss))

    '''
    # TODO: not working properly
    # save precision recall plot (on disk and TB)
    figure = plt.figure()
    plt.plot(recs, precs, marker = '.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title("Precision-recall plot")

    #print("RECALLS", recs)
    #print("PRECISIONS", precs)
    #print("ADDING FIGURE TO TENSORBOARD")
    summary_writer.add_figure("testing/sparks/prec_rec_plot", figure)
    figure.savefig("prec_rec_plot.png")
    '''

    results = {"sparks/precision": prec,
               "sparks/recall": rec,
               #"sparks/area_under_curve": a_u_c,
               "validation_loss": loss}

    if wandb_log:
        wandb.log(results)

    return results

def test_function_fixed_t(network, device, criterion, testing_datasets, logger,
                  summary_writer, threshold,
                  ignore_frames, wandb_log):
    # Requires a list of testing dataset as input
    # (every test video has its own dataset)
    # Compute precision and recall only for a fixed threshold
    # TODO: fix 'test_function' s.t. it works properly w.r.t. prec recall plot

    network.eval()

    duration = testing_datasets[0].duration
    step = testing_datasets[0].step
    half_overlap = (duration-step)//2 # to re-build videos from chunks

    # (duration-step) has to be even
    assert (duration-step)%2 == 0, "(duration-step) is not even"


    loss = 0.0
    metrics = [] # store metrics for each video

    for test_dataset in testing_datasets:
        xs = []
        ys = []
        preds = []

        with torch.no_grad():
            if (len(test_dataset)>1):
                x,y = test_dataset[0]
                xs.append(x[:-half_overlap])
                ys.append(y[:-half_overlap])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                pred = network(x[None, None])

                loss += criterion(pred[:,:,:-half_overlap],
                                 y[:,:-half_overlap].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,:-half_overlap])

                for i in range(1,len(test_dataset)-1):
                    x,y = test_dataset[i]
                    xs.append(x[half_overlap:-half_overlap])
                    ys.append(y[half_overlap:-half_overlap])

                    x = torch.Tensor(x).to(device)
                    y = torch.Tensor(y[None]).to(device)
                    pred = network(x[None, None])

                    loss += criterion(pred[:,:,half_overlap:-half_overlap],
                                     y[:,half_overlap:-half_overlap].long())

                    pred = pred[0].cpu().numpy()
                    preds.append(pred[:,half_overlap:-half_overlap])

                x,y = test_dataset[-1]
                xs.append(x[half_overlap:])
                ys.append(y[half_overlap:])

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                pred = network(x[None, None])

                loss += criterion(pred[:,:,half_overlap:],
                                 y[:,half_overlap:].long())

                pred = pred[0].cpu().numpy()
                preds.append(pred[:,half_overlap:])
            else:
                x,y = test_dataset[0]
                xs.append(x)
                ys.append(y)

                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y[None]).to(device)
                pred = network(x[None, None])

                loss += criterion(pred, y.long())

                pred = pred[0].cpu().numpy()
                preds.append(pred)

        # concatenated frames and predictions for a single video:
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        preds = np.concatenate(preds, axis=1)

        if test_dataset.pad != 0:
            xs = xs[:-test_dataset.pad]
            ys = ys[:-test_dataset.pad]
            preds = preds[:,:-test_dataset.pad]

        # predictions have logarithmic values

        # save preds as videos
        #write_videos_on_disk(xs,ys,preds,test_dataset.video_name)


        # compute predicted sparks and correspondences
        sparks = np.exp(preds[1])
        #sparks = sparks[ignore_frames:-ignore_frames]
        #sparks = np.pad(sparks,((ignore_frames,),(0,),(0,)), mode='constant')
        sparks = empty_marginal_frames(sparks, ignore_frames)

        #min_radius = 3 # minimal "radius" of a valid event

        #coords_preds = process_spark_prediction(sparks,
        #                                        t_detection=(threshold),
        #                                        min_radius=min_radius)

        #sparks_mask_true = ys[ignore_frames:-ignore_frames]
        #sparks_mask_true = np.pad(sparks_mask_true,((ignore_frames,),(0,),(0,)),
        #                          mode='constant')
        sparks_mask_true = empty_marginal_frames(ys, ignore_frames)
        sparks_mask_true = np.where(sparks_mask_true==1, 1.0, 0.0)


        #coords_true = nonmaxima_suppression(sparks_mask_true)

        #metrics.append(Metrics(*correspondences_precision_recall(coords_true,
        #                                    coords_preds, match_distance=6)))


        #metrics.append(compute_prec_rec(sparks_mask_true, sparks, thresholds))
        metrics.append(compute_prec_rec(sparks_mask_true, sparks, [threshold]))


    # Compute average validation loss
    loss = loss.item()
    loss /= len(testing_datasets)

    # Compute metrics comparing ys and y_preds
    _, precs, recs, a_u_c = reduce_metrics_thresholds(metrics)

    #prec = precs[fixed_threshold_idx]
    #rec = recs[fixed_threshold_idx]
    prec = precs[0]
    rec = recs[0]
    logger.info("\tPrecision: {:.4g}".format(prec))
    logger.info("\tRecall: {:.4g}".format(rec))
    #logger.info("\tArea under the curve: {:.4g}".format(a_u_c))
    logger.info("\tValidation loss: {:.4g}".format(loss))

    '''
    # TODO: not working properly
    # save precision recall plot (on disk and TB)
    figure = plt.figure()
    plt.plot(recs, precs, marker = '.')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title("Precision-recall plot")

    #print("RECALLS", recs)
    #print("PRECISIONS", precs)
    #print("ADDING FIGURE TO TENSORBOARD")
    summary_writer.add_figure("testing/sparks/prec_rec_plot", figure)
    figure.savefig("prec_rec_plot.png")
    '''

    results = {"sparks/precision": prec,
               "sparks/recall": rec,
               #"sparks/area_under_curve": a_u_c,
               "validation_loss": loss}

    if wandb_log:
        wandb.log(results)

    return results

''' versione con ignore_frames e vecchia versione per prec + rec
def test_function(network, device, class_weights, testing_datasets, logger,
                  wandb, threshold=0.99, ignore_ind=2, ignore_frames = 4):
    # Requires a list of testing dataset as input
    # (every test video has its own dataset)
    network.eval()

    duration = testing_datasets[0].duration
    step = testing_datasets[0].step
    half_overlap = (duration-step)//2
    # (duration-step) has to be even

    assert(ignore_frames == step)

    loss = 0
    metrics = [] # store metrics for each video

    for test_dataset in testing_datasets:
        xs = []
        ys = []
        preds = []
        sparks = []
        puffs = []
        waves = []

        with torch.no_grad():
        #    if (len(test_dataset)>1):
            # pad "ignore_frames" at the beginning and process in the unet
            x,y = test_dataset[0]
            x = x[:duration-ignore_frames]
            x = np.pad(x, ((ignore_frames,0),(0,0),(0,0)), mode='constant')
            x = torch.Tensor(x).to(device)
            pred = network(x[None, None])[0].cpu().numpy()
            preds.append(pred[:,0:-half_overlap])
            sparks.append(pred[1,0:-half_overlap])
            puffs.append(pred[2,0:-half_overlap])
            waves.append(pred[3,0:-half_overlap])

            # first true chunk from the sample
            x,y = test_dataset[0]
            xs.append(x[0:-half_overlap])
            ys.append(y[0:-half_overlap])

            x = torch.Tensor(x).to(device)
            pred = network(x[None, None])[0].cpu().numpy()
            preds.append(pred[:,half_overlap:-half_overlap])
            sparks.append(pred[1,half_overlap:-half_overlap])
            puffs.append(pred[2,half_overlap:-half_overlap])
            waves.append(pred[3,half_overlap:-half_overlap])

            for i in range(1,len(test_dataset)-1):
                x,y = test_dataset[i]
                xs.append(x[half_overlap:-half_overlap])
                ys.append(y[half_overlap:-half_overlap])

                x = torch.Tensor(x).to(device)
                pred = network(x[None, None])[0].cpu().numpy()
                preds.append(pred[:,half_overlap:-half_overlap])
                sparks.append(pred[1,half_overlap:-half_overlap])
                puffs.append(pred[2,half_overlap:-half_overlap])
                waves.append(pred[3,half_overlap:-half_overlap])

            x,y = test_dataset[-1]

            #if (test_dataset.pad == 0):
            xs.append(x[half_overlap:])
            ys.append(y[half_overlap:])

            x = torch.Tensor(x).to(device)
            pred = network(x[None, None])[0].cpu().numpy()
            preds.append(pred[:,half_overlap:-half_overlap])
            sparks.append(pred[1,half_overlap:-half_overlap])
            puffs.append(pred[2,half_overlap:-half_overlap])
            waves.append(pred[3,half_overlap:-half_overlap])

            # pad "ignore_frames" at the end and process in the unet
            x = (x.cpu().numpy())[step:]
            x = np.pad(x, ((0,ignore_frames),(0,0),(0,0)), mode='constant')
            x = torch.Tensor(x).to(device)
            pred = network(x[None, None])[0].cpu().numpy()
            preds.append(pred[:,half_overlap:])
            sparks.append(pred[1,half_overlap:])
            puffs.append(pred[2,half_overlap:])
            waves.append(pred[3,half_overlap:])


        # concatenated frames and predictions for a single video:
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        sparks = np.concatenate(sparks, axis=0)
        waves = np.concatenate(waves, axis=0)
        puffs = np.concatenate(puffs, axis=0)

        sparks = sparks[ignore_frames:-ignore_frames-test_dataset.pad]
        waves = waves[ignore_frames:-ignore_frames-test_dataset.pad]
        puffs = puffs[ignore_frames:-ignore_frames-test_dataset.pad]

        # compute predicted events and correspondences
        coords_preds = nonmaxima_suppression(sparks,
                                             threshold=np.log(threshold))
        coords_true = test_dataset.csv_data[:, [0, 2, 1]]

        metrics.append(Metrics(*correspondences_precision_recall(coords_true,
                                                                 coords_preds,
                                                                 6)))

        # compute validation loss
        ys_loss = torch.Tensor(ys[None,:])

        preds = np.concatenate(preds, axis=1)
        preds = preds[None,:,ignore_frames:-ignore_frames-test_dataset.pad]
        preds = torch.Tensor(preds)


        loss += nn.functional.nll_loss(preds,
                                ys_loss.long(),
                                weight=class_weights,
                                ignore_index=ignore_ind).item()

        # save preds as videos
        imageio.volwrite(os.path.join("predictions",
                                      test_dataset.video_name + "_xs.tif"),
                                      xs)#np.uint8(255*xs))
        imageio.volwrite(os.path.join("predictions",
                                      test_dataset.video_name + "_ys.tif"),
                                      ys)#np.uint8(127*ys))
        imageio.volwrite(os.path.join("predictions",
                                      test_dataset.video_name + "_sparks.tif"),
                                      np.exp(sparks))#np.uint8(255*np.exp(y_preds)))
        imageio.volwrite(os.path.join("predictions",
                                      test_dataset.video_name + "_waves.tif"),
                                      np.exp(waves))#np.uint8(255*np.exp(y_preds)))
        imageio.volwrite(os.path.join("predictions",
                                      test_dataset.video_name + "_puffs.tif"),
                                      np.exp(puffs))#np.uint8(255*np.exp(y_preds)))

    # Compute average validation loss
    loss /= len(testing_datasets)

    # Compute metrics comparing ys and y_preds
    results = reduce_metrics(metrics)
    prec = results[0]
    rec = results[1]
    logger.info("\tPrecision: {:.4g}".format(prec))
    logger.info("\tRecall: {:.4g}".format(rec))
    logger.info("\tValidation loss: {:.4g}".format(loss))

    results = {"precision": prec,
               "recall": rec,
               "validation_loss": loss}

    wandb.log(results)

    return results
'''

# Iterator (?) over the dataset
def mycycle(dataset_loader):
    while True:
        for x in dataset_loader:
            yield x

#_cycle = mycycle(dataset_loader)

def sampler(dataset_loader):
    return next(mycycle(dataset_loader))#(_cycle)


''' OLD FUNCTIONS
############### results processing methods ###############


def store_results_video(network, device, test_dataset):
    # store a single video together with labels and predictions as a list
    # returns: video_name, [video, labels, predictions]
    half_overlap = int((test_dataset.duration-test_dataset.step)/2)
    nb_chunks = len(test_dataset)

    xs = []
    ys = []
    y_preds = []

    with torch.no_grad():
        if nb_chunks > 1:
            x,y = test_dataset[0]
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            y_pred = network(x[None, None])[0,1]

            xs.append(x[0:-half_overlap].cpu().numpy())
            ys.append(y[0:-half_overlap].cpu().numpy())
            y_preds.append(y_pred[0:-half_overlap].cpu().numpy())

            for i in range(1, nb_chunks-1):
                x,y = test_dataset[i]
                x = torch.Tensor(x).to(device)
                y = torch.Tensor(y).to(device)
                y_pred = network(x[None, None])[0,1]

                xs.append(x[half_overlap:-half_overlap].cpu().numpy())
                ys.append(y[half_overlap:-half_overlap].cpu().numpy())
                y_preds.append(y_pred[half_overlap:-half_overlap].cpu().numpy())

            x,y = test_dataset[-1]
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            y_pred = network(x[None, None])[0,1]

            if (test_dataset.pad == 0):
                xs.append(x[half_overlap:].cpu().numpy())
                ys.append(y[half_overlap:].cpu().numpy())
                y_preds.append(y_pred[half_overlap:].cpu().numpy())
            else: # remove padded frames if necessary
                xs.append(x[half_overlap:-test_dataset.pad].cpu().numpy())
                ys.append(y[half_overlap:-test_dataset.pad].cpu().numpy())
                y_preds.append(y_pred[half_overlap:-test_dataset.pad].cpu().numpy())

        else: # only one element(=chunk) in dataset
            x,y = test_dataset[0]
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            y_pred = network(x[None,None])[0,1]

            xs.append(x.cpu().numpy())
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

        # concatenate chunks of video, labels and predictions
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)

    return test_dataset.video_name, [xs, ys, y_preds]

def store_results(network, device, test_datasets):
    results = []
    for test_dataset in test_datasets:
        results.append(store_results_video(network, device, test_dataset))

    return results




def get_preds_coords(y_preds, threshold):
    # threshold between 0 and 1
    return nonmaxima_suppression(y_preds, threshold=np.log(threshold))

def get_labels_coords(ys, ignore_ind = 2):
    ys[ys==ignore_ind] = 0
    ys = ndi.binary_erosion(ys)
    return np.argwhere(ys==1)
'''
