"""
24.10.2022

Script with function for metrics on UNet output computation.

REMARKS
24.10.2022: functions that aren't currently used are commented out and put at
the end of the script (such as ..., )
16.06.2023: removed some unused functions (both in code and at the end of the script)

"""
from bisect import bisect_left
from collections import namedtuple
import logging

import numpy as np
from scipy import optimize, spatial, sparse

logger = logging.getLogger(__name__)

################################ Global params #################################

# physiological params to get sparks locations
# these have to be coherent in the whole project

PIXEL_SIZE = 0.2  # 1 pixel = 0.2 um x 0.2 um
global MIN_DIST_XY
MIN_DIST_XY = round(1.8 / PIXEL_SIZE)  # min distance in space between sparks
TIME_FRAME = 6.8  # 1 frame = 6.8 ms
global MIN_DIST_T
MIN_DIST_T = round(20 / TIME_FRAME)  # min distance in time between sparks

################################ Generic utils #################################


def diff(l1, l2):
    # l1 and l2 are lists
    # return l1 - l2
    return list(map(list, (set(map(tuple, l1))).difference(set(map(tuple, l2)))))


############################### Generic metrics ################################


# from UNet outputs and labels, get dict of metrics for a single video
def get_metrics_from_summary(tot_preds,
                             tp_preds,
                             ignored_preds,
                             unlabeled_preds,
                             tot_ys,
                             tp_ys,
                             undetected_ys):
    r"""
    Compute instance-based metrics from matched events summary.

    Instance-based metrics are:
    # - confusion matrix
    - precision per class
    - recall per class
    - % correctly classified events per class
    - % detected events per class

    Parameters
    ----------
    tot_preds :         dict of total number of predicted events per class
    tp_preds :          dict of true positive predicted events per class
    ignored_preds :     dict of ignored predicted events per class
    unlabeled_preds :  dict of unlabeled predicted events per class
    tot_ys :            dict of total number of annotated events per class
    tp_ys :             dict of true positive annotated events per class
    undetected_ys :     dict of undetected annotated events per class
    """
    ca_release_events = ["sparks", "puffs", "waves"]

    # compute metrics
    metrics = {}

    for event_type in ca_release_events:
        # precision
        denom = tot_preds[event_type] - ignored_preds[event_type]
        if denom > 0:
            metrics[event_type + "/precision"] = (tp_preds[event_type] / denom)
        else:
            metrics[event_type + "/precision"] = 0

        # recall
        if tot_ys[event_type] > 0:
            metrics[event_type + "/recall"] = (tp_ys[event_type] /
                                               tot_ys[event_type])
        else:
            metrics[event_type + "/recall"] = 0

        # % correctly classified events
        denom = tot_preds[event_type] - \
            ignored_preds[event_type] - unlabeled_preds[event_type]
        if denom > 0:
            metrics[event_type +
                    "/correctly_classified"] = (tp_preds[event_type] / denom)
        else:
            metrics[event_type + "/correctly_classified"] = 0

        # % detected events
        if tot_ys[event_type] > 0:
            metrics[event_type +
                    "/detected"] = (1 - (undetected_ys[event_type] / tot_ys[event_type]))
        else:
            metrics[event_type + "/detected"] = 0

    # compute average over classes for each metric
    for m in ["precision", "recall", "correctly_classified", "detected"]:
        metrics["average/" + m] = np.mean(
            [metrics[event_type + "/" + m]
                for event_type in ca_release_events]
        )

    return metrics


def compute_iou(ys_roi, preds_roi, ignore_mask=None, debug=False):
    """
    Compute IoU for given single annotated and predicted events.
    ys_roi :            annotated event ROI
    preds_roi :         predicted event ROI
    ignore_mask :       mask that is ignored by loss function during training
    debug:              if true, print when both ys and preds are empty
    """
    # define mask where pixels aren't ignored by loss function
    if ignore_mask is not None:
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    intersection = np.logical_and(ys_roi, preds_roi_real)
    union = np.logical_or(ys_roi, preds_roi_real)
    if np.count_nonzero(union):
        iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    else:
        iou = 1.0
        if debug:
            logger.warning("Warning: both annotations and preds are empty")
    return iou


def compute_inter_min(ys_roi, preds_roi, ignore_mask=None):
    """
    Compute intersection over minimum area for given single annotated and predicted events.
    ys_roi :            annotated event ROI
    preds_roi :         predicted event ROI
    ignore_mask :       mask that is ignored by loss function during training
    """
    # define mask where pixels aren't ignored by loss function
    if ignore_mask is not None:
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    intersection = np.logical_and(ys_roi, preds_roi_real)
    ys_area = np.count_nonzero(ys_roi)
    preds_area = np.count_nonzero(preds_roi_real)

    if preds_area > 0:
        iomin = np.count_nonzero(intersection) / min(preds_area, ys_area)
    else:
        iomin = 0
    return iomin


def compute_iomin_one_hot(y_vector, preds_array):
    """
    Compute intersection over minimum score for given single annotated event
    and all predicted events.
    y_vector :          flattened, one-hot encoded annotated event csr_array
                        (shape = 1 x movie shape)
    preds_array :       flattened, one-hot encoded predicted events csr_array
                        (shape = #preds x movie shape)
                        !!! intersect predicted events with negation of ignore
                        maks before passing them here !!!

    Returns: list of iomin scores for each predicted event
    """

    # check that y_vector is not empty
    assert y_vector.count_nonzero != 0, "y_vector is empty"

    # compute intersection of csr_array y_vector with each row of csr_array preds_array
    intersection = y_vector.multiply(preds_array)

    if intersection.count_nonzero() == 0:
        return np.zeros(preds_array.shape[0])

    else:
        # compute non-zero elements for each row (=predicted events) of intersection
        intersection_area = intersection.getnnz(axis=1).astype(float)

        # get denominator for iomin using non-zero elements of y_vector and preds_array
        denominator = np.minimum(preds_array.getnnz(axis=1),
                                 y_vector.getnnz()).astype(float)

        # compute iomin by dividing intersection by denominator
        # if denominator is 0, set iomin to 0
        scores = np.divide(
            intersection_area,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float16),
            where=(denominator > 0),
        )

        return scores


################################ Sparks metrics ################################
"""
Utils for computing metrics related to sparks, e.g.
- compute correspondences between annotations and preds
- compute precision and recall
"""

Metrics = namedtuple(
    "Metrics", ["precision", "recall", "f1_score", "tp", "tp_fp", "tp_fn"]
)


def correspondences_precision_recall(
    coords_real,
    coords_pred,
    match_distance_t=MIN_DIST_T,
    match_distance_xy=MIN_DIST_XY,
    return_pairs_coords=False,
    return_nb_results=False,
):
    """
    Compute best matches given two sets of coordinates, one from the
    ground-truth and another one from the network predictions. A match is
    considered correct if the Euclidean distance between coordinates is smaller
    than `match_distance`. With the computed matches, it estimates the precision
    and recall of the prediction.

    If return_pairs_coords == True, return paired sparks coordinated
    If return_nb_results == True, return only tp, tp_fp, tp_fn as a dict
    """
    # convert coords to arrays
    coords_real = np.asarray(coords_real, dtype=float)
    coords_pred = np.asarray(coords_pred, dtype=float)

    # divide temporal coords by match_distance_t and spatial coords by
    # match_distance_xy
    if coords_real.size > 0:
        coords_real[:, 0] /= match_distance_t
        coords_real[:, 1] /= match_distance_xy
        coords_real[:, 2] /= match_distance_xy

    if coords_pred.size > 0:
        coords_pred[:, 0] /= match_distance_t
        coords_pred[:, 1] /= match_distance_xy
        coords_pred[:, 2] /= match_distance_xy  # check if integer!!!!!!!!!!!

    if coords_real.size and coords_pred.size > 0:
        w = spatial.distance_matrix(coords_real, coords_pred)
        w[w > 1] = 9999999  # NEW
        row_ind, col_ind = optimize.linear_sum_assignment(w)

    if return_pairs_coords:
        if coords_real.size > 0:
            # multiply coords by match distances
            coords_real[:, 0] *= match_distance_t
            coords_real[:, 1] *= match_distance_xy
            coords_real[:, 2] *= match_distance_xy

        if coords_pred.size > 0:
            coords_pred[:, 0] *= match_distance_t
            coords_pred[:, 1] *= match_distance_xy
            coords_pred[:, 2] *= match_distance_xy

        if coords_real.size and coords_pred.size > 0:
            # true positive pairs:
            paired_real = [
                coords_real[i].tolist()
                for i, j in zip(row_ind, col_ind)
                if w[i, j] <= 1
            ]
            paired_pred = [
                coords_pred[j].tolist()
                for i, j in zip(row_ind, col_ind)
                if w[i, j] <= 1
            ]

            # false positive (predictions):
            false_positives = sorted(diff(coords_pred, paired_pred))

            # false negative (annotations):
            false_negatives = sorted(diff(coords_real, paired_real))

            if return_nb_results:
                tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
                tp_fp = len(coords_pred)
                tp_fn = len(coords_real)

                res = {"tp": tp, "tp_fp": tp_fp, "tp_fn": tp_fn}
                return res, paired_real, paired_pred, false_positives, false_negatives
            else:
                return paired_real, paired_pred, false_positives, false_negatives
        else:
            if return_nb_results:
                tp = 0
                tp_fp = len(coords_pred)
                tp_fn = len(coords_real)

                res = {"tp": tp, "tp_fp": tp_fp, "tp_fn": tp_fn}
                return res, [], [], coords_pred, coords_real
            else:
                return [], [], coords_pred, coords_real

    else:
        if (coords_real.size > 0) and (coords_pred.size > 0):
            tp = np.count_nonzero(w[row_ind, col_ind] <= 1)
        else:
            tp = 0

        tp_fp = len(coords_pred)
        tp_fn = len(coords_real)

        if return_nb_results:
            return {"tp": tp, "tp_fp": tp_fp, "tp_fn": tp_fn}

        if tp_fp > 0:
            precision = tp / tp_fp
        else:
            precision = 1.0

        if tp_fn > 0:
            recall = tp / tp_fn
        else:
            recall = 1.0

        f1_score = compute_f_score(precision, recall)

        return precision, recall, f1_score, tp, tp_fp, tp_fn


def compute_f_score(prec, rec, beta=1):
    if beta == 1:
        f_score = 2 * prec * rec / (prec + rec) if prec + rec != 0 else 0.0
    else:
        f_score = (
            (1 + beta * beta) * (prec + rec) / (beta * beta * prec + rec)
            if prec + rec != 0
            else 0.0
        )
    return f_score


########################### Instances-based metrics ############################


def _get_sparse_binary_encoded_mask(mask):
    """
    mask is an array with labelled event instances
    return a sparse matrix with one-hot encoding of the events mask
    (each row corresponds to a different event)
    """
    # flatten mask
    v = mask.flatten()

    # get one-hot encoding of events as sparse matrix
    rows = v
    cols = np.arange(v.size)
    data = np.ones(v.size, dtype=bool)
    sparse_v = sparse.csr_array(
        (data, (rows, cols)), shape=(v.max() + 1, v.size), dtype=bool
    )

    # remove background "event"
    sparse_v = sparse_v[1:]
    return sparse_v


def get_score_matrix(ys_instances, preds_instances, ignore_mask=None, score="iomin"):
    r"""
    Compute pair-wise scores between annotated event instances (ys_segmentation)
    and predicted event instances (preds_instances).

    ys_instances, preds_instances:  dicts indexed by ca release event types
                                    each entry is an int array
    ignore_mask:                    binary mask, ROIs ignored during training
    score:                          scoring method (IoMin or IoU)

    return an array of shape n_ys_events x n_preds_events
    """

    assert score in ["iomin", "iou"], "score type must be 'iomin' or 'iou'."

    # compute matrices with all separated events summed
    ys_all_events = sum(ys_instances.values())
    preds_all_events = sum(preds_instances.values())

    if score == "iomin":
        # intersect predicted events with negation of ignore mask
        if ignore_mask is not None:
            preds_all_events = np.logical_and(
                preds_all_events, np.logical_not(ignore_mask)
            )

        # convert to one-hot encoding and transpose matrices
        ys_all_events = _get_sparse_binary_encoded_mask(ys_all_events)
        preds_all_events = _get_sparse_binary_encoded_mask(preds_all_events)

        # compute pairwise scores
        scores = []
        for y_vector in ys_all_events:
            y_scores = compute_iomin_one_hot(
                y_vector=y_vector, preds_array=preds_all_events
            )
            scores.append(y_scores)

        scores = np.array(scores)

    # TODO (if necessary)
    # elif score == "iou":
    #     # get number of annotated and predicted events
    #     n_ys_events = ys_all_events.max()
    #     n_preds_events = preds_all_events.max()

    #     # create empty score matrix for IoU or IoMin
    #     scores = np.zeros((n_ys_events, n_preds_events))

    #     # intersect predicted events with negation of ignore mask
    #     if ignore_mask is not None:
    #         preds_all_events = np.logical_and(
    #             preds_all_events, np.logical_not(ignore_mask)
    #         )

    #     # compute pairwise scores
    #     for pred_id in range(1, n_preds_events + 1):
    #         preds_roi = preds_all_events == pred_id
    #         # check that the predicted ROI is not empty
    #         # assert preds_roi.any(), f"the predicted ROI n.{pred_id} is empty!"

    #         # check if predicted ROI intersect at least one annotated event
    #         if np.count_nonzero(np.logical_and(ys_all_events, preds_roi)) != 0:

    #             for ys_id in range(1, n_ys_events + 1):
    #                 ys_roi = ys_all_events == ys_id
    #                 # check that the annotated ROI is not empty
    #                 # assert ys_roi.any(), f"the annotated ROI n.{ys_roi} is empty!"

    #                 # check if predicted and annotated ROIs intersect
    #                 if np.count_nonzero(np.logical_and(ys_roi, preds_roi)) != 0:

    #                     # compute scores
    #                     if score == "iomin":
    #                         scores[ys_id - 1, pred_id - 1] = compute_inter_min(
    #                             ys_roi=ys_roi, preds_roi=preds_roi
    #                         )
    #                         # logger.debug(f"score computation between y_id={ys_id} and pred_id={pred_id} took {time.time() - start:.2f}s")
    #                     elif score == "iou":
    #                         scores[ys_id - 1, pred_id - 1] = compute_iou(
    #                             ys_roi=ys_roi, preds_roi=preds_roi
    #                         )

    # assertions take ~45s to be computed...

    return scores


def get_matches_summary(ys_instances, preds_instances, scores, t, ignore_mask):
    r"""
    Analyze matched of each predicted event with annotated events and assert
    whether the event is correct, mispredicted, ignored, or unlabeled. Keep
    also track on undetected labelled events.

    ys_instances, preds_instances:  dicts indexed with ca release event types
                                    each entry is a int array
    scores:                         array of shape n_ys_events x n_preds_events
    t:                              min threshold to consider two events a match
    ignore_mask:                    binary mask, ROIs ignored during training
    """

    # define calcium release event types
    ca_release_events = ["sparks", "puffs", "waves"]

    # initialize lists that summarize the results
    matched_ys_ids = {ca_class: {} for ca_class in ca_release_events}
    matched_preds_ids = {ca_class: {} for ca_class in ca_release_events}

    for ca_class in ca_release_events:
        # dicts need to be initialized here because used in the next loop

        # get set of IDs of annotated events
        matched_ys_ids[ca_class]['tot'] = set(
            np.unique(ys_instances[ca_class]))
        matched_ys_ids[ca_class]['tot'].remove(0)

        # get name of other classes
        other_classes = ca_release_events[:]
        other_classes.remove(ca_class)

        for other_class in other_classes:
            # init mispredicted events
            matched_preds_ids[ca_class][other_class] = set()

            # keep track of mispredicted events in annotations as well
            matched_ys_ids[ca_class][other_class] = set()

        # init undetected annotated events
        matched_ys_ids[ca_class]['undetected'] = matched_ys_ids[ca_class]['tot'].copy()

    for ca_class in ca_release_events:
        # get set of IDs of predicted events
        matched_preds_ids[ca_class]['tot'] = set(
            np.unique(preds_instances[ca_class]))
        matched_preds_ids[ca_class]['tot'].remove(0)

        # init sets of correctly matched annotations and predictions
        matched_preds_ids[ca_class]['tp'] = set()
        matched_ys_ids[ca_class]['tp'] = set()

        # init ignored predicted events
        matched_preds_ids[ca_class]['ignored'] = set()

        # init predicted events not matched with any label
        matched_preds_ids[ca_class]['unlabeled'] = set()

        ### go through predicted events and match them with annotated events ###
        for pred_id in matched_preds_ids[ca_class]['tot']:
            # get set of y_ids that are matched with pred_id (score > t):
            matched_events = set(np.where(scores[:, pred_id - 1] >= t)[0] + 1)

            # if matched_events is empty, chech if pred_id is ignored
            if not matched_events:
                pred_roi = preds_instances[ca_class] == pred_id
                pred_roi_size = np.count_nonzero(pred_roi)

                ignored_roi = np.logical_and(pred_roi, ignore_mask)
                ignored_roi_size = np.count_nonzero(ignored_roi)

                overlap = ignored_roi_size / pred_roi_size

                if overlap >= t:
                    # mark detected event as ignored
                    matched_preds_ids[ca_class]['ignored'].add(pred_id)
                else:
                    # detected event does not match any labelled event
                    matched_preds_ids[ca_class]['unlabeled'].add(pred_id)

            # otherwise, pred_id matches with at least one labelled event
            else:
                for other_class in ca_release_events:
                    # check if pred_id matched with an event of the other class
                    matched_other_class = matched_events & matched_ys_ids[other_class]['tot']

                    # remove matched events from undetected events
                    matched_ys_ids[other_class]['undetected'] -= matched_other_class

                    if matched_other_class:
                        if other_class == ca_class:
                            # pred_id is a correct prediction
                            matched_preds_ids[ca_class]['tp'].add(pred_id)
                            matched_ys_ids[ca_class]['tp'] |= matched_other_class
                        else:
                            # pred_id is misclassified
                            matched_preds_ids[ca_class][other_class].add(
                                pred_id)
                            matched_ys_ids[other_class][ca_class] |= matched_other_class

    return matched_ys_ids, matched_preds_ids
