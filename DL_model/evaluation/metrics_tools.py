"""
Script with function for metrics on UNet output computation.

REMARKS
24.10.2022: functions that aren't currently used are commented out and put at
the end of the script (such as ..., )
16.06.2023: removed some unused functions (both in code and at the end of the script)

Author: Prisca Dotti
Last modified: 28.09.2023
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from config import config

logger = logging.getLogger(__name__)

__all__ = [
    "list_difference",
    "get_metrics_from_summary",
    "compute_iou",
    "compute_inter_min",
    "compute_iomin_one_hot",
    "get_score_matrix",
    "get_matches_summary",
    # "correspondences_precision_recall",
    "compute_f_score",
    "compute_mcc",
    # "get_precision_recall",
    # "get_precision_recall_f1",
    # "get_precision_recall_f1_from_coords",
    "get_df_summary_events",
    "get_df_metrics",
]

################################ Generic utils #################################


def list_difference(l1: List, l2: List) -> List:
    """
    Compute the difference between two lists, l1 - l2.

    Args:
        l1 (list): The first list.
        l2 (list): The second list.

    Returns:
        list: The elements that are in l1 but not in l2.
    """
    return [item for item in l1 if item not in l2]


############################### Generic metrics ################################


def get_metrics_from_summary(
    tot_preds: dict,
    tp_preds: dict,
    ignored_preds: dict,
    unlabeled_preds: dict,
    tot_ys: dict,
    tp_ys: dict,
    undetected_ys: dict,
) -> dict:
    """
    Compute instance-based metrics from matched events summary.

    Instance-based metrics are:
    - precision per class
    - recall per class
    - % correctly classified events per class
    - % detected events per class

    Parameters:
        tot_preds (dict): Total number of predicted events per class.
        tp_preds (dict): True positive predicted events per class.
        ignored_preds (dict): Ignored predicted events per class.
        unlabeled_preds (dict): Unlabeled predicted events per class.
        tot_ys (dict): Total number of annotated events per class.
        tp_ys (dict): True positive annotated events per class.
        undetected_ys (dict): Undetected annotated events per class.

    Returns:
        dict: Dictionary of computed metrics.
    """
    metrics = {}

    for event_type in config.classes_dict.keys():
        if event_type not in config.event_types:
            continue

        denom_preds = tot_preds[event_type] - ignored_preds[event_type]
        denom_ys = tot_ys[event_type]

        precision = tp_preds[event_type] / denom_preds if denom_preds > 0 else 0
        recall = tp_ys[event_type] / denom_ys if denom_ys > 0 else 0
        correctly_classified = (
            tp_preds[event_type] / (denom_preds - unlabeled_preds[event_type])
            if denom_preds > 0
            else 0
        )
        detected = 1 - (undetected_ys[event_type] / denom_ys) if denom_ys > 0 else 0

        metrics[event_type + "/precision"] = precision
        metrics[event_type + "/recall"] = recall
        metrics[event_type + "/correctly_classified"] = correctly_classified
        metrics[event_type + "/detected"] = detected

    # Also compute metrics with respect to all events
    denom_preds = sum(tot_preds.values()) - sum(ignored_preds.values())
    denom_ys = sum(tot_ys.values())

    precision = sum(tp_preds.values()) / denom_preds if denom_preds > 0 else 0
    recall = sum(tp_ys.values()) / denom_ys if denom_ys > 0 else 0
    correctly_classified = (
        sum(tp_preds.values()) / (denom_preds - sum(unlabeled_preds.values()))
        if denom_preds > 0
        else 0
    )
    detected = 1 - (sum(undetected_ys.values()) / denom_ys) if denom_ys > 0 else 0

    metrics["total/precision"] = precision
    metrics["total/recall"] = recall
    metrics["total/correctly_classified"] = correctly_classified
    metrics["total/detected"] = detected

    # Compute average over classes for each metric
    for m in ["precision", "recall", "correctly_classified", "detected"]:
        metrics["average/" + m] = np.mean(
            [metrics[event_type + "/" + m] for event_type in config.event_types]
        )

    return metrics


def compute_iou(
    ys_roi: np.ndarray,
    preds_roi: np.ndarray,
    ignore_mask: np.ndarray = np.array([]),
    debug: bool = False,
) -> float:
    """
    Compute Intersection over Union (IoU) for given single annotated and predicted
    events.

    Args:
        ys_roi (numpy.ndarray): Annotated event ROI (binary mask).
        preds_roi (numpy.ndarray): Predicted event ROI (binary mask).
        ignore_mask (numpy.ndarray, optional): Mask that is ignored by the loss
            function during training.
        debug (bool, optional): If True, print a warning when both ys and preds are
            empty.

    Returns:
        float: The computed IoU value.
    """
    # Define a mask where pixels aren't ignored by the loss function
    if ignore_mask.any():
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    # Calculate the intersection and union of the masks
    intersection = np.logical_and(ys_roi, preds_roi_real)
    union = np.logical_or(ys_roi, preds_roi_real)

    # Compute IoU
    if np.count_nonzero(union):
        iou = np.count_nonzero(intersection) / np.count_nonzero(union)
    else:
        iou = 1.0
        if debug:
            logger.warning("Warning: both annotations and preds are empty")

    return iou


def compute_inter_min(
    ys_roi: np.ndarray, preds_roi: np.ndarray, ignore_mask: np.ndarray = np.array([])
) -> float:
    """
    Compute Intersection over Minimum Area for given single annotated and predicted
    events.

    Args:
        ys_roi (numpy.ndarray): Annotated event ROI (binary mask).
        preds_roi (numpy.ndarray): Predicted event ROI (binary mask).
        ignore_mask (numpy.ndarray, optional): Mask that is ignored by the loss
            function during training.

    Returns:
        float: The computed Intersection over Minimum Area (IoMin) value.
    """
    # Define a mask where pixels aren't ignored by the loss function
    if ignore_mask.any():
        compute_mask = np.logical_not(ignore_mask)
        preds_roi_real = np.logical_and(preds_roi, compute_mask)
    else:
        preds_roi_real = preds_roi

    # Calculate the intersection and areas of the masks
    intersection = np.logical_and(ys_roi, preds_roi_real)
    ys_area = np.count_nonzero(ys_roi)
    preds_area = np.count_nonzero(preds_roi_real)

    # Compute IoMin
    if preds_area > 0:
        iomin = np.count_nonzero(intersection) / min(preds_area, ys_area)
    else:
        iomin = 0

    return iomin


def compute_iomin_one_hot(
    y_vector: sparse.csr_matrix, preds_array: sparse.csr_matrix
) -> np.ndarray:
    """
    Compute Intersection over Minimum Score for a given single annotated event and
    all predicted events.

    Args:
        y_vector (csr_matrix): Flattened, one-hot encoded annotated event CSR matrix.
            (shape = 1 x movie shape)
        preds_array (csr_matrix): Flattened, one-hot encoded predicted events CSR
            matrix. Ensure that predicted events are intersected with the negation of
            the ignore mask before calling this function.
            (shape = #preds x movie shape)

    Returns:
        numpy.ndarray: List of IoMin scores for each predicted event.
    """
    # Check that y_vector is not empty
    assert y_vector.count_nonzero() != 0, "y_vector is empty"

    # Compute the intersection of CSR matrix y_vector with each row of CSR matrix
    # preds_array
    intersection = y_vector.multiply(preds_array)

    if intersection.count_nonzero() == 0:
        return np.zeros(preds_array.shape[0])

    else:
        # Compute non-zero elements for each row (predicted event) of intersection
        intersection_area = intersection.getnnz(axis=1).astype(float)

        # Get the denominator for IoMin using non-zero elements of preds_array and
        # y_vector
        denominator = np.minimum(preds_array.getnnz(axis=1), y_vector.getnnz()).astype(
            float
        )

        # Compute IoMin by dividing intersection_area by denominator
        # If denominator is 0, set IoMin to 0
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
# - compute correspondences between annotations and preds
- compute precision and recall
"""


def compute_f_score(precision: float, recall: float, beta: float = 1) -> float:
    """
    Compute the F-beta score given precision and recall.

    Args:
    - precision (float): Precision value.
    - recall (float): Recall value.
    - beta (float, optional): Weight parameter for balancing precision and
        recall.

    Returns:
    - float: F-beta score.
    """
    if precision + recall == 0:
        return 0.0

    if beta == 1:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = (
            (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        )

    return f_score


def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    if (tp != 0) and (tn == 0) and (fp == 0) and (fn == 0):
        mcc = 1
    elif (tn != 0) and (tp == 0) and (fp == 0) and (fn == 0):
        mcc = 1
    elif (fp != 0) and (tp == 0) and (tn == 0) and (fn == 0):
        mcc = -1
    elif (fn != 0) and (tp == 0) and (tn == 0) and (fp == 0):
        mcc = -1
    elif (tp == 0) and (fn == 0) and (fp != 0) and (tn != 0):
        mcc = 0
    elif (fp == 0) and (tn == 0) and (tp != 0) and (fn != 0):
        mcc = 0
    elif (tp == 0) and (fp == 0) and (fn != 0) and (tn != 0):
        mcc = 0
    elif (fn == 0) and (tn == 0) and (tp != 0) and (fp != 0):
        mcc = 0
    else:
        mcc = (tp * tn - fp * fn) / np.sqrt(
            float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )

    return mcc


########################### Instances-based metrics ############################


def _get_sparse_binary_encoded_mask(mask: np.ndarray) -> sparse.csr_matrix:
    """
    Create a sparse binary encoded mask from an array with labeled event instances.

    Args:
        mask (numpy.ndarray): Array with labeled event instances.

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix with one-hot encoding of the events
        mask (each row corresponds to a different event).
    """
    # Flatten mask
    v = mask.flatten()

    # Get one-hot encoding of events as a sparse matrix
    rows = v
    cols = np.arange(v.size)
    data = np.ones(v.size, dtype=bool)
    sparse_v = sparse.csr_array(
        (data, (rows, cols)), shape=(v.max() + 1, v.size), dtype=bool
    )

    # Remove background "event"
    sparse_v = sparse_v[1:]
    return sparse_v


def get_score_matrix(
    ys_instances: Dict[str, np.ndarray],
    preds_instances: Dict[str, np.ndarray],
    ignore_mask: np.ndarray = np.array([]),
) -> np.ndarray:
    """
    Compute pairwise IoMin scores between annotated event instances and predicted
    event instances.

    Args:
        ys_instances (dict): Dictionary of annotated event instances, indexed by
            event types (each entry is an int array).
        preds_instances (dict): Dictionary of predicted event instances, indexed by
            event types (each entry is an int array).
        ignore_mask (numpy.ndarray, optional): Binary mask indicating ROIs ignored
            during training.

    Returns:
        numpy.ndarray: Array of shape (n_ys_events, n_preds_events) containing
        pairwise scores.
    """
    # Compute matrices with all separated events summed
    ys_all_events = np.sum(list(ys_instances.values()), axis=0)
    preds_all_events = np.sum(list(preds_instances.values()), axis=0)

    # Intersect predicted events with the negation of the ignore mas
    if ignore_mask.any():
        preds_all_events = np.logical_and(preds_all_events, np.logical_not(ignore_mask))

    # Convert to one-hot encoding and transpose matrices
    ys_all_events = _get_sparse_binary_encoded_mask(ys_all_events)
    preds_all_events = _get_sparse_binary_encoded_mask(preds_all_events)

    # Compute pairwise scores
    scores = []
    for y_vector in ys_all_events:
        y_scores = compute_iomin_one_hot(
            y_vector=y_vector, preds_array=preds_all_events
        )
        scores.append(y_scores)

    scores = np.array(scores)

    return scores


def get_matches_summary(
    ys_instances: Dict[str, np.ndarray],
    preds_instances: Dict[str, np.ndarray],
    scores: np.ndarray,
    ignore_mask: np.ndarray,
) -> Tuple[Dict[str, Dict[str, set]], Dict[str, Dict[str, set]]]:
    """
    Analyze matched predicted events with annotated events and categorize them.

    Args:
        ys_instances (dict): Dictionary of annotated event instances.
        preds_instances (dict): Dictionary of predicted event instances.
        scores (numpy.ndarray): Array of pairwise scores.
        ignore_mask (numpy.ndarray): Binary mask, ROIs ignored during training.

    Returns:
        Tuple: A tuple containing matched_ys_ids (annotated events summary) and
        matched_preds_ids (predicted events summary).
    """
    # Get list of class names
    ca_classes = list(preds_instances.keys())  # sparks, puffs, waves

    # Initialize dicts that summarize the results
    matched_ys_ids = {ca_class: {} for ca_class in ca_classes}
    matched_preds_ids = {ca_class: {} for ca_class in ca_classes}

    for ca_class in ca_classes:
        # Get set of IDs of all annotated events
        ys_ids = set(np.unique(ys_instances[ca_class])) - {0}
        matched_ys_ids[ca_class]["tot"] = ys_ids.copy()

        # Get set of IDs of all predicted events
        preds_ids = set(np.unique(preds_instances[ca_class])) - {0}
        matched_preds_ids[ca_class]["tot"] = preds_ids.copy()

        for other_class in ca_classes:
            if other_class == ca_class:
                continue

            # Initialize mispredicted events (in annotations and predictions)
            matched_ys_ids[ca_class][other_class] = set()
            matched_preds_ids[ca_class][other_class] = set()

        # Initialize undetected annotated events
        matched_ys_ids[ca_class]["undetected"] = ys_ids.copy()

    for ca_class in ca_classes:
        # Initialize sets of correctly matched annotations and predictions
        matched_preds_ids[ca_class][ca_class] = set()
        matched_ys_ids[ca_class][ca_class] = set()

        # Initialize ignored predicted events
        matched_preds_ids[ca_class]["ignored"] = set()

        # Initialize predicted events not matched with any label
        matched_preds_ids[ca_class]["unlabeled"] = set()

        ### Go through predicted events and match them with annotated events ###
        for pred_id in matched_preds_ids[ca_class]["tot"]:
            # Get set of y_ids that are matched with pred_id (score > t):
            matched_events = set(
                np.where(scores[:, pred_id - 1] >= config.iomin_t)[0] + 1
            )

            # If matched_events is empty, chech if pred_id is ignored
            if not matched_events:
                pred_roi = preds_instances[ca_class] == pred_id
                ignored_roi = np.logical_and(pred_roi, ignore_mask)
                overlap = np.count_nonzero(ignored_roi) / np.count_nonzero(pred_roi)

                if overlap >= config.iomin_t:
                    # Mark detected event as ignored
                    matched_preds_ids[ca_class]["ignored"].add(pred_id)
                else:
                    # Detected event does not match any labelled event
                    matched_preds_ids[ca_class]["unlabeled"].add(pred_id)

            # Otherwise, pred_id matches with at least one labelled event
            else:
                for other_class in ca_classes:
                    # Check if pred_id matched with an event of the other class
                    matched_other_class = (
                        matched_events & matched_ys_ids[other_class]["tot"]
                    )

                    # Remove matched events from undetected events
                    matched_ys_ids[other_class]["undetected"] -= matched_other_class

                    if matched_other_class:
                        if other_class == ca_class:
                            # pred_id is a correct prediction
                            matched_preds_ids[ca_class][ca_class].add(pred_id)
                            matched_ys_ids[ca_class][ca_class] |= matched_other_class
                        else:
                            # pred_id is misclassified
                            matched_preds_ids[ca_class][other_class].add(pred_id)
                            matched_ys_ids[other_class][ca_class] |= matched_other_class

    return matched_ys_ids, matched_preds_ids


############################# DataFrame functions ##############################

import pandas as pd


def get_df_summary_events(
    inference_type: str,
    matched_ids: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    matched_percent: Dict[str, Dict[str, Dict[str, float]]],
    is_detected: bool = True,
) -> pd.DataFrame:
    """
    Create a summary DataFrame of detected or labeled events for a given
    inference type, where columns are event types and index are metrics (tp,
    ignored, etc.).

    Args:
        inference_type (str): The type of inference (e.g., "overlap" or
            "average").
        matched_ids (dict): Dictionary containing matched events.
        matched_percent (dict): Dictionary containing matched event percentages.
        is_detected (bool): If True, create a summary for detected events;
            otherwise, for labeled events.

    Returns:
        pd.DataFrame: Summary DataFrame of detected or labeled events.
    """
    # Define the event type label
    event_type_label = "Detected" if is_detected else "Labeled"

    # Create DataFrames from matched_ids and matched_percent
    df_ids = pd.DataFrame(matched_ids[inference_type]["sum"])
    df_percent = pd.DataFrame(matched_percent[inference_type])

    # Rename columns with '%' for percentages
    df_percent = df_percent.rename(
        columns={event_type: f"% {event_type}" for event_type in config.event_types}
    )

    # Combine DataFrames
    df = pd.concat([df_ids, df_percent], axis=1)

    # Define a dictionary for renaming index labels
    index_labels = {
        "tot": f"Total {event_type_label}",
        **{
            event_type: f"Matched with {event_type_label} {event_type.capitalize()}"
            for event_type in config.event_types
        },
    }

    # Rename the index labels
    df = df.rename(index=index_labels)

    # Define data types for the DataFrame
    convert_dict = {event_type: int for event_type in config.event_types}

    return df.astype(convert_dict)


def get_df_metrics(
    inference_type: str, metrics_all: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Create a DataFrame of metrics for the specified inference type.

    Args:
        inference_type (str): The type of inference (e.g., "overlap" or
            "average").
        metrics_all (dict): Dictionary containing metrics data for all inference
            types.

    Returns:
        pd.DataFrame: DataFrame of metrics for the specified inference type.
    """
    # Define the event types to consider
    event_types = [
        *[
            event_type
            for event_type in config.classes_dict.keys()
            if event_type not in ["background", "ignore"]
        ],
        "average",
    ]

    # Initialize a dictionary to store the metrics data
    df_data = {event_type: {} for event_type in event_types}

    # Iterate through metrics for the specified inference type
    for type_metric, val in metrics_all[inference_type].items():
        for event_type in event_types:
            if type_metric.startswith(event_type):
                # Remove the event type prefix from the metric name
                metric_name = type_metric[len(event_type) + 1 :]
                df_data[event_type][metric_name] = val

    # Create a DataFrame where index is event types and columns are metrics
    df = pd.DataFrame(df_data).T
    return df
