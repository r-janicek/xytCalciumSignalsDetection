
from collections import namedtuple

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize, spatial
from skimage import morphology
from sklearn.metrics import roc_auc_score
from bisect import bisect_left

import imageio
import os


__all__ = ["Metrics",
           "nonmaxima_suppression",
           "process_spark_prediction",
           "inverse_argwhere",
           "correspondences_precision_recall",
           #"new_correspondences_precision_recall",
           "reduce_metrics",
           "empty_marginal_frames",
           "write_videos_on_disk",
           "compute_prec_rec",
           "reduce_metrics_thresholds",
           "take_closest"
           ]


################################ Generic utils #################################

def empty_marginal_frames(video, n_frames):
    # Set first and last n_frames of a video to zero
    video = video[n_frames:-n_frames]
    video = np.pad(video,((n_frames,),(0,),(0,)), mode='constant')
    return video


def write_videos_on_disk(xs, ys, preds, video_name):
    # Write all videos on disk
    # xs : input video used by network
    # ys: segmentation video used in loss function
    # preds : all u-net preds [bg preds, sparks preds, puffs preds, waves preds]

    imageio.volwrite(os.path.join("predictions", video_name + "_xs.tif"),
                                  xs)
    imageio.volwrite(os.path.join("predictions", video_name + "_ys.tif"),
                                  np.uint8(ys))
    imageio.volwrite(os.path.join("predictions", video_name + "_sparks.tif"),
                                  np.exp(preds[1]))
    imageio.volwrite(os.path.join("predictions", video_name + "_waves.tif"),
                                  np.exp(preds[2]))
    imageio.volwrite(os.path.join("predictions", video_name + "_puffs.tif"),
                                  np.exp(preds[3]))


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before


################################ Sparks metrics ################################

'''
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
'''

Metrics = namedtuple('Metrics', ['precision', 'recall', 'tp', 'tp_fp', 'tp_fn'])


def in_bounds(points, shape):

    return np.logical_and.reduce([(coords_i >= 0) & (coords_i < shape_i)
                                for coords_i, shape_i in zip(points.T, shape)])


def nonmaxima_suppression(img, return_mask=False, neighborhood_radius=5, threshold=0.5):

    smooth_img = ndi.gaussian_filter(img, 2) # 2 instead of 1
    dilated = ndi.grey_dilation(smooth_img, (neighborhood_radius,) * img.ndim)
    argmaxima = np.logical_and(smooth_img == dilated, img > threshold)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def process_spark_prediction(pred, t_detection = 0.9, neighborhood_radius = 5, min_radius = 4, return_mask = False, return_clean_pred = False):
    # remove small objects
    min_size = (2 * min_radius) ** pred.ndim

    pred_boolean = pred > t_detection
    small_objs_removed = morphology.remove_small_objects(pred_boolean, min_size=min_size)
    big_pred = np.where(small_objs_removed, pred, 0)

    if return_clean_pred:
        return big_pred

    gaussian = ndi.gaussian_filter(big_pred, 2)
    dilated = ndi.grey_dilation(gaussian, (neighborhood_radius,) * pred.ndim)

    # detect events (nonmaxima suppression)
    argmaxima = np.logical_and(gaussian == dilated, big_pred > t_detection)
    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def inverse_argwhere(coords, shape, dtype):
    """
    Creates an array with given shape and dtype such that

    np.argwhere(inverse_argwhere(coords, shape, dtype)) == coords

    up to a rounding of `coords`.
    """

    res = np.zeros(shape, dtype=dtype)
    intcoords = np.int_(np.round(coords))
    intcoords = intcoords[in_bounds(intcoords, shape)]
    res[intcoords[:, 0], intcoords[:, 1], intcoords[:, 2]] = 1
    return res


def correspondences_precision_recall(coords_real, coords_pred, match_distance):
    """
    Compute best matches given two sets of coordinates, one from the
    ground-truth and another one from the network predictions. A match is
    considered correct if the Euclidean distance between coordinates is smaller
    than `match_distance`. With the computed matches, it estimates the precision
    and recall of the prediction.
    """

    w = spatial.distance_matrix(coords_real, coords_pred)
    w[w > match_distance] = 9999999 # NEW
    row_ind, col_ind = optimize.linear_sum_assignment(w)

    tp = np.count_nonzero(w[row_ind, col_ind] <= match_distance)
    tp_fp = len(coords_pred)
    tp_fn = len(coords_real)

    if tp_fp > 0:
        precision = tp / tp_fp
    else:
        precision = 1.0

    if tp_fn > 0:
        recall = tp / tp_fn
    else:
        recall = 1.0

    return precision, recall, tp, tp_fp, tp_fn

# new function for correspondences: closest point to annotation instead of minimal weight matching

'''def new_correspondences_precision_recall(coords_real, coords_pred, match_distance):
    tp_fn = len(coords_real)
    tp_fp = len(coords_pred)

    if (tp_fn == 0 or tp_fp == 0):
        tp = 0
    else:
        w = spatial.distance_matrix(coords_real, coords_pred)
        #print("distance matrix shape ", np.shape(w))
        closest_annotation = np.argmin(w, axis=1)
        tp = np.count_nonzero(w[range(w.shape[0]),closest_annotation] <= match_distance)

        # TODO:
        # se due annotations vengono assegnate alla stessa
        # prediction, prendere la piu vicina

    if tp_fn == 0: # no annotations
        recall = 1
    else:
        recall = tp / tp_fn

    if tp_fp == 0: # no predictions
        precision = 1
    else:
        precision = tp / tp_fp

    return precision, recall, tp, tp_fp, tp_fn'''


def reduce_metrics(results):

    tp = sum(i.tp for i in results)
    tp_fp = sum(i.tp_fp for i in results)
    tp_fn = sum(i.tp_fn for i in results)

    if tp_fp > 0:
        precision = tp / tp_fp
    else:
        precision = 1.0
    recall = tp / tp_fn

    return Metrics(precision, recall, tp, tp_fp, tp_fn)


def compute_prec_rec(annotations, preds, thresholds):
    # annotations: video of sparks segmentation w/ values in {0,1}
    # preds: video of sparks preds w/ values in [0,1]
    # thresholds : list of thresholds applied to the preds over which events are kept
    # returns a list of Metrics tuples corresponding to thresholds and AUC

    min_radius = 3 # minimal "radius" of a valid event
    match_distance = 6

    metrics = [] # list of 'Metrics' tuples: precision, recall, tp, tp_fp, tp_fn
    #prec = []
    #rec = []

    coords_true = nonmaxima_suppression(annotations)

    # compute prec and rec for every threshold
    for t in thresholds:
        coords_preds = process_spark_prediction(preds,
                                                t_detection=t,
                                                min_radius=min_radius)

        prec_rec = Metrics(*correspondences_precision_recall(coords_true,
                                                             coords_preds,
                                                             match_distance))

        metrics.append(prec_rec)
        #print("threshold", t)
        #prec.append(prec_rec.precision)
        #rec.append(prec_rec.recall)


    # compute AUC for this sample
    #area_under_curve = auc(rec, prec)

    return metrics#, area_under_curve

def reduce_metrics_thresholds(results):
    # apply metrics reduction to results corresponding to different thresholds
    # results is a list that for every video contains a list of 'Metrics'
    # thresholds is the list of used thresholds
    # returns a list of reduced 'Metrics' instances for every threshold

    results = list(map(list, zip(*results))) # "transpose" list

    reduced_metrics = []
    prec = []
    rec = []

    for idx_t, res in enumerate(results):
        # res is a list of 'Metrics' of all videos wrt a threshold
        reduced_res = reduce_metrics(res)

        reduced_metrics.append(reduced_res)
        prec.append(reduced_res.precision)
        rec.append(reduced_res.recall)

    # compute area under the curve for reduced metrics
    #print("REC",rec)
    #print("PREC",prec)
    #area_under_curve = roc_auc_score(rec, prec)
    #print("AREA UNDER CURVE", area_under_curve)

    return reduced_metrics, prec, rec, None



############################ Puffs and waves metrics ###########################

'''
Utils for computing metrics related to puffs and waves, e.g.:
- Jaccard index
- exclusion region for Jaccard index
'''
