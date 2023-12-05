"""
Script with functions for any type of data processing in the project
(annotations, predictions, sample movies, ...)

REMARKS
24.10.2022: functions that aren't currently used are commented out and put at
the end of the script (such as compute_filtered_butter, ...)

Author: Prisca Dotti
Last modified: 30.10.2023
"""

import logging
import time
from typing import Dict, List, Tuple, Union

import cc3d
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy import signal, spatial
from scipy.ndimage import (
    binary_fill_holes,
    center_of_mass,
    distance_transform_edt,
    find_objects,
    label,
)
from scipy.stats import ttest_rel
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk
from skimage.segmentation import watershed

from DL_model.config import config

logger = logging.getLogger(__name__)

__all__ = [
    "keep_percentile",
    "reduce_sparks_size",
    "annotate_undefined_around_peaks",
    "annotate_sparks_with_peaks",
    "apply_ignore_regions_to_events",
    "trim_and_pad_video",
    "remove_padding",
    "exclude_marginal_sparks_coords",
    "get_convex_hull",
    "get_smallest_event",
    "get_argmax_segmentation",
    "get_otsu_argmax_segmentation",
    "get_separated_events",
    "count_instances_per_class",
    "renumber_labelled_mask",
    "masks_to_instances_dict",
    "merge_labels",
    "analyse_spark_roi",
    "remove_small_events",
    "process_raw_predictions",
    "preds_dict_to_mask",
    "detect_single_roi_peak",
    "detect_spark_peaks",
    "simple_nonmaxima_suppression",
    "one_sided_non_inferiority_ttest",
    "compute_filtered_butter",
    "get_event_parameters",
    "get_event_parameters_simple",
    "moving_average",
    "count_classes_in_roi",
    "get_cell_mask",
    "compute_snr",
]


########################## Annotations preprocessing ###########################


def keep_percentile(
    movie: np.ndarray, roi_mask: np.ndarray, percentile: int = 75, min_roi_size: int = 2
) -> np.ndarray:
    """
    For a given event ROI, keep only points above a certain percentile.

    Args:
        movie (numpy.ndarray): Input movie, possibly smoothed.
        roi_mask (numpy.ndarray): ROI corresponding to one event.
        percentile (int): Percentile value (e.g., 75).
        min_roi_size (int): Minimum size of the ROI in pixels.

    Returns:
        numpy.ndarray: New ROI corresponding to the event.
    """
    # Extract movie ROI for the event
    movie_event = np.where(roi_mask, movie, 0)

    # Compute the specified percentile
    movie_event_values = movie_event[movie_event > 0]
    threshold_value = np.percentile(movie_event_values, percentile)

    # Threshold the ROI to keep values above the percentile
    new_roi_mask = movie_event >= threshold_value

    # Check if any of the new ROI dims is smaller than min_roi_size
    roi_sizes = np.max(np.where(new_roi_mask), axis=1) - np.min(
        np.where(new_roi_mask), axis=1
    )

    # If any dimension is too small, dilate the ROI
    if roi_sizes.min() < min_roi_size:
        new_roi_mask = ndi.binary_dilation(new_roi_mask, iterations=1)

    # Get the convex hull of the ROI for a more regular shape
    new_roi_mask = get_convex_hull(new_roi_mask)

    return new_roi_mask


def reduce_sparks_size(
    movie: np.ndarray,
    class_mask: np.ndarray,
    event_mask: np.ndarray,
    sigma: float = 2.0,
    k: int = 75,
) -> np.ndarray:
    """
    Reduce spark dimensions in the class annotation mask.

    Args:
        movie (ndarray): Input movie array.
        class_mask (ndarray): Mask with classified events.
        event_mask (ndarray): Mask with identified events.
        sigma (float): Sigma parameter of the Gaussian filter.
        k (int): Value of percentile.

    Returns:
        ndarray: Class mask where removed parts are labeled as undefined (4).
    """
    # Normalize input movie between 0 and 1
    if movie.max() > 1:
        movie = (movie - movie.min()) / (movie.max() - movie.min())

    # Get sparks event mask
    spark_mask = np.where(class_mask == 1, event_mask, 0)

    # Get list of sparks IDs
    event_list = [label for label in np.unique(spark_mask) if label != 0]

    # Smooth movie if sigma is greater than one
    if sigma > 0:
        movie = ndi.gaussian_filter(movie, sigma=sigma)

    # Create a new class mask using the percentile method
    new_class_mask = np.copy(class_mask)

    # Set spark ROIs to ignore index
    new_class_mask[new_class_mask == 1] = config.ignore_index

    # Reduce the size of each ROI
    for event_id in event_list:
        event_roi = spark_mask == event_id

        # Reduce spark size dimension with respect to percentile
        new_roi_mask = keep_percentile(movie=movie, roi_mask=event_roi, percentile=k)

        # Set new smaller spark peaks to 1
        new_peak = np.logical_and(event_roi, new_roi_mask)
        new_class_mask = np.where(new_peak, 1, new_class_mask)

    assert np.all(
        class_mask.astype(bool) == new_class_mask.astype(bool)
    ), "New class mask events don't match old class mask events."

    return new_class_mask


def annotate_undefined_around_peaks(
    peak_mask: np.ndarray, inner_radius: float = 2.5, outer_radius: float = 3.5
) -> np.ndarray:
    """
    Annotate regions around spark peaks as undefined (class 4).

    Parameters:
    - mask (numpy.ndarray): Binary mask containing spark peaks.
    - inner_radius (float): Inner radius for spark annotation.
    - outer_radius (float): Outer radius for undefined annotation.

    Returns:
    - numpy.ndarray: Annotated mask with spark and undefined regions.

    """
    # Convert the input mask to boolean (True for peaks, False for background)
    peak_mask = peak_mask.astype(bool)

    dist_transform = np.array(ndi.distance_transform_edt(~peak_mask))

    annotated_mask = np.zeros(peak_mask.shape, dtype=np.int64)
    annotated_mask[dist_transform < outer_radius] = config.ignore_index
    annotated_mask[dist_transform < inner_radius] = config.classes_dict["sparks"]

    return annotated_mask


def annotate_sparks_with_peaks(
    video: np.ndarray,
    labels_mask: np.ndarray,
    peak_radius: int = 3,
    ignore_radius: int = 2,
    ignore_frames: int = 0,
) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    Annotate sparks in a binary mask using peak detection.

    Parameters:
    - video (numpy.ndarray): Input video data.
    - labels_mask (numpy.ndarray): Mask containing segmented labeled events.
    - peak_radius (int): Radius for spark peak region.
    - ignore_radius (int): Radius for spark ignore region.
    - ignore_frames (int): Number of frames to ignore from the start and end of the
        video.

    Returns (tuple):
    - List[Tuple[int, int, int]]: List of spark peak locations.
    - numpy.ndarray: Annotated mask with spark regions.
    """
    sparks_label = config.classes_dict["sparks"]

    # Detect spark peak locations
    if sparks_label in labels_mask:
        sparks_mask = np.where(labels_mask == 1, sparks_label, 0)

        sparks_loc, sparks_peak_mask = simple_nonmaxima_suppression(
            input_array=video,
            mask=sparks_mask,
            min_distance=config.conn_mask,
            threshold=0.0,
            sigma=2.0,
        )

        logger.debug(f"Number of sparks detected: {len(sparks_loc)}")

        # Ignore sparks ROIs detected in ignored frames
        if ignore_frames > 0:
            # remove sparks from maxima mask
            sparks_peak_mask = trim_and_pad_video(sparks_peak_mask, ignore_frames)

        # Add spark ignore regions
        sparks_peak_mask = annotate_undefined_around_peaks(
            peak_mask=sparks_peak_mask,
            inner_radius=peak_radius,
            outer_radius=peak_radius + ignore_radius,
        )

        # Remove sparks from the original mask
        no_sparks_mask = np.where(labels_mask == 1, 0, labels_mask)

        # Create a new mask with spark annotations
        new_mask = np.where(sparks_peak_mask != 0, sparks_peak_mask, no_sparks_mask)

        # Return spark locations and annotated mask
        if ignore_frames > 0:
            # Remove sparks detected in ignored frames from locations
            mask_duration = labels_mask.shape[0]
            sparks_loc = exclude_marginal_sparks_coords(
                spark_coords=sparks_loc,
                n_exclude_frames=ignore_frames,
                video_duration=mask_duration,
            )
        return sparks_loc, new_mask

    else:
        # Return empty lists and the original mask if no sparks found
        return [], labels_mask


def apply_ignore_regions_to_events(
    mask: np.ndarray, ignore_radii: list = [1, 5, 3], apply_erosion: list = [0, 1, 1]
) -> np.ndarray:
    """
    Apply ignore regions around events in a segmentation mask.

    Args:
    - mask (numpy.ndarray): The input segmentation mask.
    - ignore_radii (list): List of radii for ignore regions for each class.
    - apply_erosion (list): List of binary flags for applying erosion for each
        class.

    Returns:
    - numpy.ndarray: The mask with ignore regions applied.
    """
    # Ensure that ignore_radii and apply_erosion have values for each class
    assert len(ignore_radii) == config.num_classes - 1, (
        f"list_radius_ignore must contain {config.num_classes - 1} values, "
        f"but contains {len(ignore_radii)}."
    )
    assert len(apply_erosion) == config.num_classes - 1, (
        f"list_erosion must contain {config.num_classes - 1} values, "
        f"but contains {len(apply_erosion)}."
    )

    for class_id in config.classes_dict.values():
        if class_id in mask:
            class_mask = np.where(mask == class_id, 1, 0)
            dilated_mask = ndi.binary_dilation(
                class_mask, iterations=ignore_radii[class_id - 1]
            )
            if apply_erosion[class_id - 1] == 1:
                eroded_mask = ndi.binary_erosion(
                    class_mask, iterations=ignore_radii[class_id - 1]
                )
                ignore_mask = np.logical_xor(dilated_mask, eroded_mask)
            else:
                ignore_mask = np.logical_xor(dilated_mask, class_mask)

            mask = np.where(ignore_mask, config.ignore_index, mask)

    return mask


########################### General masks processing ###########################


def trim_and_pad_video(
    video: np.ndarray, n_margin_frames: int, pad_value: int = 0
) -> np.ndarray:
    """
    Trim and pad a video by removing a specified number of frames from the
    beginning and end and padding with the given padding value.

    Args:
    - video (numpy.ndarray): The input video.
    - margin_frames (int): The number of frames to remove from each end.
    - pad_value (int): Padding value for the marginal frames.

    Returns:
    - numpy.ndarray: The trimmed and padded video.
    """
    if n_margin_frames > 0:
        trimmed_video = video[n_margin_frames:-n_margin_frames]
        trimmed_video = np.pad(
            trimmed_video,
            ((n_margin_frames,), (0,), (0,)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        trimmed_video = video

    assert np.shape(video) == np.shape(trimmed_video)

    return trimmed_video


def remove_padding(preds: torch.Tensor, original_duration: int) -> torch.Tensor:
    """
    Remove padding from the predictions to match the original duration.

    Args:
        preds (torch.Tensor): Predictions with padding.
        original_duration (int): Original duration to crop to.

    Returns:
        torch.Tensor: Cropped predictions without padding.
    """
    pad = preds.size(-3) - original_duration
    if pad > 0:
        start_pad = pad // 2
        end_pad = -(pad // 2 + pad % 2)
        preds = preds[..., start_pad:end_pad, :, :]

    return preds


def exclude_marginal_sparks_coords(
    spark_coords: List[Tuple[int, int, int]], n_exclude_frames: int, video_duration: int
) -> List[Tuple[int, int, int]]:
    """
    Exclude spark coordinates located in the first and last 'n_exclude_frames'
    frames of a video with 'video_duration'.

    Args:
    - spark_coords (list of tuples): List of spark coordinates, each represented
        as a tuple (frame, y, x).
    - n_exclude_frames (int): Number of frames to exclude from the beginning and
        end of the video.
    - video_duration (int): Total duration of the video.

    Returns:
    - List of filtered spark coordinates.
    """
    if n_exclude_frames > 0:
        if spark_coords:
            start_frame = n_exclude_frames
            end_frame = video_duration - n_exclude_frames

            new_coords = [
                loc
                for loc in spark_coords
                if loc[0] >= start_frame and loc[0] < end_frame
            ]
            return new_coords

    return spark_coords


def get_convex_hull(image: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull of a binary Numpy array.

    Args:
    - image (numpy.ndarray): Binary input array.

    Returns:
    - numpy.ndarray: Binary array representing the convex hull.
    """
    points = np.transpose(np.where(image))
    hull = spatial.ConvexHull(points)
    delaunay = spatial.Delaunay(points[hull.vertices])
    indices = np.stack(np.indices(image.shape), axis=-1)
    non_zero_indices = np.nonzero(delaunay.find_simplex(indices) + 1)
    conv_hull_image = np.zeros(image.shape)
    conv_hull_image[non_zero_indices] = 1
    return conv_hull_image


def get_smallest_event(
    events_mask: np.ndarray,
    get_shortest_duration: bool = False,
    get_smallest_width: bool = False,
) -> Dict[str, int]:
    """
    Get the dimensions of the 'smallest' event within the given class mask.

    Parameters:
    - events_mask (numpy.ndarray): Mask containing instances of events in a
        class (along the second and third axes).
    - get_shortest_duration (bool): If True, get the shortest event duration
        (along the first axis).
    - get_smallest_width (bool): If True, get the smallest event width.

    Returns:
    - dict: A dictionary containing the requested event dimensions.

    """
    event_min_dims = {}

    if events_mask.any():
        event_slices = ndi.measurements.find_objects(events_mask)

        if get_shortest_duration:
            events_durations = [
                event_slice[0].stop - event_slice[0].start
                for event_slice in event_slices
                if event_slice is not None
            ]
            event_min_dims["shortest_duration"] = np.min(events_durations)

        if get_smallest_width:
            events_widths = [
                min(
                    event_slice[1].stop - event_slice[1].start,
                    event_slice[2].stop - event_slice[2].start,
                )
                for event_slice in event_slices
            ]
            event_min_dims["smallest_width"] = np.min(events_widths)

    return event_min_dims


######################### UNet predictions processing ##########################


def get_argmax_segmentation(
    class_predictions_raw: np.ndarray, return_classes: bool = True
) -> Union[np.ndarray, Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Get class-wise segmentation predictions from raw UNet outputs.

    Args:
    - class_predictions_raw (numpy.ndarray): Raw UNet outputs for each class.
                             Shape: (num_classes x duration x height x width).
    - return_classes (bool): If True, return class-wise predictions.
                             If False, return the argmax class predictions.

    Returns:
    - numpy.ndarray or dict: Segmentation predictions.
        If return_classes is True, returns a dictionary of class-wise predictions.
        If return_classes is False, returns the argmax class predictions.
    """
    argmax_classes = np.argmax(class_predictions_raw, axis=0)

    if not return_classes:
        return argmax_classes

    class_predictions_dict = {
        event_type: (argmax_classes == event_label)
        for event_type, event_label in config.classes_dict.items()
    }

    return class_predictions_dict, argmax_classes


def get_otsu_argmax_segmentation(
    preds: Dict[str, np.ndarray], return_classes: bool = True, debug: bool = False
) -> Union[np.ndarray, Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """
    Compute segmentation predictions using Otsu thresholding. Compute Otsu
    threshold with respect to the sum of positive predictions and remove
    predictions below that threshold, then get argmax predictions on thresholded
    UNet output.

    Args:
    - preds (dict): Raw (exponential) UNet outputs for each class.
                    Dictionary with event types as keys and numpy arrays as values.
    - return_classes (bool): If True, return class-wise predictions.
                             If False, return the argmax class predictions.
    - debug (bool): If True, print debugging information.

    Returns:
    - numpy.ndarray or tuple: Segmentation predictions.
        If return_classes is True, returns a tuple (argmax_preds, class_preds).
        If return_classes is False, returns the argmax class predictions.
    """
    # Compute threshold on summed predicted events
    sum_preds = np.sum(
        [
            preds[event_type]
            for event_type in preds.keys()
            if event_type != "background"
        ],
        axis=0,
    )

    t_otsu = threshold_otsu(sum_preds)
    if debug:
        logger.debug(f"Events detection threshold: {t_otsu:.3f}")

    # Get binary mask of valid predictions
    binary_sum_preds = sum_preds > t_otsu

    # Create new empty mask of shape (num_classes x duration x height x width)
    masked_class_preds = np.zeros((config.num_classes, *binary_sum_preds.shape))

    # Mask out removed events from UNet preds for each class
    # This is necessary because the classes need to be in the right order
    for event_type, event_label in config.classes_dict.items():
        if event_type == "ignore":
            continue
        masked_class_preds[event_label] = binary_sum_preds * preds[event_type]

    # Get argmax of classes
    return get_argmax_segmentation(
        class_predictions_raw=masked_class_preds, return_classes=return_classes
    )


def get_separated_events(
    argmax_preds: Dict[str, np.ndarray],
    movie: np.ndarray,
    debug: bool = False,
    training_mode: bool = False,
    watershed_classes: List[str] = ["sparks"],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Tuple[int, int, int]]],]:
    """
    Separate each class into event instances using connected components and
    watershed separation algorithm for the classes listed in "watershed_classes".

    Args:
    - argmax_preds (dict): Segmented UNet output with class-wise predictions.
    - movie (numpy.ndarray): Input movie.
    - debug (bool): If True, print intermediate results.
    - training_mode (bool): If True, separate events using a simpler algorithm.
    - watershed_classes (list of str): list of event types for which Watershed
        algorithm is used in addition to connected components.

    Returns (tuple):
    - dict: Separated events with keys config.classes_dict.keys() where each
    entry is an array with labelled events (from 1 to n_events).
    - dict: Dictionary with list of peaks locations (keys: watersheds_classes).
    """
    separated_events = {}
    coords_events = {}

    for event_type in config.event_types:
        # Separate connected components in puff (3) and wave classes (2)
        if event_type not in watershed_classes:
            separated_events[event_type] = cc3d.connected_components(
                argmax_preds[event_type],
                connectivity=config.connectivity,
                return_N=False,
            )

        # Compute event peaks locations of classes in watershed_classes list
        else:
            coords_events[event_type], mask_loc = simple_nonmaxima_suppression(
                input_array=movie,
                mask=argmax_preds[event_type],
                min_distance=config.conn_mask,
                threshold=0.0,
                sigma=config.sparks_sigma,
            )

            if debug:
                logger.debug(
                    f"Number of {event_type} detected by nonmaxima suppression:"
                    f" {len(coords_events[event_type])}"
                )

            # Compute smooth version of input video
            smooth_xs = ndi.gaussian_filter(movie, sigma=config.sparks_sigma)
            smooth_xs = smooth_xs.astype(float)

            # Compute watershed separation
            markers = label(mask_loc)[0]

            split_event_mask = watershed(
                image=-smooth_xs,
                markers=markers,
                mask=argmax_preds[event_type],
                connectivity=3,
                compactness=1,
            )

            if not training_mode:
                # Labelling sparks with peaks in all connected components only
                # if not training otherwise, it is not very important

                # Check if all connected components have been labelled
                all_ccs_labelled = np.all(
                    split_event_mask.astype(bool)
                    == argmax_preds[event_type].astype(bool)
                )

                if not all_ccs_labelled:
                    if debug:
                        logger.debug(
                            "Not all sparks were labelled, computing missing events..."
                        )
                        logger.debug(
                            f"Number of sparks before correction: {np.max(split_event_mask)}"
                        )

                    # Get number of labelled events
                    n_split_events = np.max(split_event_mask)

                    # If not all CCs have been labelled, obtain unlabelled CCs
                    # and split them
                    missing_peaks = np.logical_xor(
                        split_event_mask.astype(bool),
                        argmax_preds[event_type].astype(bool),
                    )

                    # Separate unlabelled CCs and label them
                    labelled_missing_peaks = cc3d.connected_components(
                        missing_peaks, connectivity=config.connectivity, return_N=False
                    )

                    # Increase labels by number of sparks already present
                    labelled_missing_peaks = np.where(
                        labelled_missing_peaks,
                        labelled_missing_peaks + n_split_events,
                        0,
                    )

                    # Merge sparks with peaks and sparks without them
                    split_event_mask += labelled_missing_peaks

                    # Get peak location of missing sparks and add it to peaks lists
                    missing_peaks_ids = list(np.unique(labelled_missing_peaks))
                    missing_peaks_ids.remove(0)
                    for peak_id in missing_peaks_ids:
                        peak_roi_xs = np.where(
                            labelled_missing_peaks == peak_id, smooth_xs, 0
                        )

                        peak_loc = np.unravel_index(
                            peak_roi_xs.argmax(), peak_roi_xs.shape
                        )

                        coords_events[event_type].append(list(peak_loc))

                    # Assert that now all CCs have been labelled
                    all_ccs_labelled = np.all(
                        split_event_mask.astype(bool)
                        == argmax_preds[event_type].astype(bool)
                    )

                    if debug:
                        logger.debug(
                            f"Number of sparks after correction: {np.max(split_event_mask)}"
                        )

                assert all_ccs_labelled, "Some sparks CCs haven't been labelled!"

            separated_events[event_type] = split_event_mask

        if debug:
            # Check that event IDs are ordered and consecutive
            assert len(np.unique(separated_events[event_type])) - 1 == np.max(
                separated_events[event_type]
            ), f"{event_type} IDs are not consecutive: {np.unique(np.unique(separated_events[event_type]))}"

    return separated_events, coords_events


###################### Event instances' masks processing #######################


def count_instances_per_class(instances_dict: Dict[str, List[int]]) -> Dict[str, int]:
    """
    Given a dictionary of class-wise event instances, count the number of
    instances for each class.

    Args:
    - instances_dict (dict): Dictionary of class-wise event instances.

    Returns:
    - dict: Dictionary of event instances counts for each class.
    """

    instances_counts = {}

    for event_type, event_instances in instances_dict.items():
        instances_counts[event_type] = len(np.unique(event_instances)) - 1

    return instances_counts


def renumber_labelled_mask(labelled_mask: np.ndarray, shift_id: int = 0) -> np.ndarray:
    """
    Renumber labelled events in a mask to consecutive integers.

    Args:
    - labelled_mask (numpy.ndarray): Mask with labelled events (positive integers).
    - shift_id (int): Shift each label by this integer.

    Returns:
    - numpy.ndarray: Renumber labelled events in a mask to consecutive integers
    (shifted by shift_id if specified).

    """

    if labelled_mask.max() > 0:
        unique_labels = np.unique(labelled_mask)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background label
        new_mask = np.zeros_like(labelled_mask)

        for new_id, old_id in enumerate(unique_labels):
            new_mask[labelled_mask == old_id] = new_id + shift_id + 1

        # Check that the events have been renumbered correctly
        new_labels = np.unique(new_mask)
        new_labels = new_labels[new_labels != 0]
        expected_labels = np.arange(shift_id + 1, shift_id + len(unique_labels) + 1)
        assert np.array_equal(
            new_labels, expected_labels
        ), f"New labels are incorrect: {new_labels}."
    else:
        new_mask = labelled_mask

    return new_mask


def masks_to_instances_dict(
    instances_mask: np.ndarray, labels_mask: np.ndarray, shift_ids: bool = False
) -> Dict[str, np.ndarray]:
    """
    Given two integer masks, one with event labels and one with event instances,
    get a dictionary indexed by event labels with values containing the mask of
    event instances.

    Args:
    - labels_mask (numpy.ndarray): Mask with event labels (in
        config.clases_dict.keys()).
    - instances_mask (numpy.ndarray): Mask with event instances (with values in
        0,1,...,n_events).
    - shift_ids (bool): If True, events in different classes have different IDs,
        otherwise, they are numbered starting from 1.

    Returns:
    - dict: Dictionary with event types as keys and event instances masks as
        values.
    """
    instances_dict = {}
    shift_id = 0

    for event_type, event_label in config.classes_dict.items():
        if event_type not in config.event_types:
            continue
        instances_dict[event_type] = np.where(
            labels_mask == event_label, instances_mask, 0
        )

        # Renumber event instances
        instances_dict[event_type] = renumber_labelled_mask(
            labelled_mask=instances_dict[event_type], shift_id=shift_id
        )

        if shift_ids:
            shift_id = max(shift_id, np.max(instances_dict[event_type]))

    return instances_dict


def merge_labels(labelled_mask: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Merge labels in the input mask if their distance in time is smaller than
    the max gap.

    Args:
    - labelled_mask (numpy.ndarray): Mask with labelled events.
    - max_gap (int): Maximum time gap for merging labels.

    Returns:
    - numpy.ndarray: Mask with merged labels.
    """
    if not labelled_mask.any():
        return labelled_mask  # No labels to merge if the mask is empty

    # Dilate annotations along time
    struct = np.zeros((max_gap + 1, 3, 3))
    struct[:, 1, 1] = 1
    dilated_labels_t = ndi.binary_dilation(labelled_mask, structure=struct)

    # Count if the number of connected components has decreased
    conn = 26
    merged_labels, n_merged_labels = cc3d.connected_components(
        dilated_labels_t, connectivity=conn, return_N=True
    )

    # If some labels have merged, re-label the mask accordingly
    if n_merged_labels < len(np.unique(labelled_mask)) - 1:
        merged_labelled_mask = np.where(labelled_mask, merged_labels, 0)
        return merged_labelled_mask
    else:
        return labelled_mask


# Currently not used, but maybe it could be useful
def analyse_spark_roi(spark_mask: np.ndarray) -> Tuple[int, float, int]:
    """
    Analyze a spark event ROI mask.

    Args:
        spark_mask (numpy.ndarray): ROI denoting a spark event in the movie.

    Returns:
        tuple: A tuple containing the spark's duration, maximum radius, and
        number of pixels in the ROI.
    """
    # Calculate the coordinates of foreground pixels
    locations = np.where(spark_mask)
    n_pixels = len(locations[0])

    # Calculate the duration by counting unique frames
    frames = np.unique(locations[0])
    duration = len(frames)

    # Calculate the maximum radius
    max_radius = 0
    for frame in frames:
        # Calculate the Euclidean distance transform for the spark mask
        dt = distance_transform_edt(spark_mask[frame])
        temp_radius = np.amax(dt)
        max_radius = max(max_radius, temp_radius)

    return duration, max_radius, n_pixels


def remove_small_events(
    instances_dict: Dict[str, np.ndarray], new_id: int = 0
) -> Dict[str, np.ndarray]:
    """
    Remove small predicted events and merge events belonging together.

    Args:
    - instances_dict (dict): A dictionary containing event type as keys and
        numpy arrays with labelled events (positive integers) as values.
    - new_id (int): Value used to replace removed events (default: 0).

    Returns:
    - dict: A dictionary with removed or relabelled events for each event type.
    """
    clean_instances_dict = {}
    for event_type in instances_dict.keys():
        clean_instances_mask = np.copy(instances_dict[event_type])

        if event_type not in config.event_types:
            clean_instances_dict[event_type] = clean_instances_mask
            continue

        # Merge event labels of event separated by a small gap
        max_gap = config.max_gap[event_type]
        if max_gap:
            clean_instances_mask = merge_labels(
                labelled_mask=clean_instances_mask, max_gap=max_gap
            )

        # Remove small events
        events_ids = list(np.unique(clean_instances_mask))
        events_ids.remove(0)

        for event_id in events_ids:
            # Get ROI of single event
            event_roi = clean_instances_mask == event_id
            # Get ranges of current event (duration, height, width)
            slices = find_objects(event_roi)[0]

            for min_size, slice in zip(config.min_size[event_type], slices):
                # If a minimal size is specified for the current axis in the
                # config.py file, check that the event is large enough
                if min_size:
                    event_size = slice.stop - slice.start
                    if event_size < min_size:
                        # If the event is smaller than the minimal size in the
                        # specified axis, replace its label with new_id
                        clean_instances_mask[event_roi] = new_id

                        break
        clean_instances_dict[event_type] = clean_instances_mask
    return clean_instances_dict


def process_raw_predictions(
    raw_preds_dict: Dict[str, np.ndarray],
    input_movie: np.ndarray,
    training_mode: bool = False,
    debug: bool = False,
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[Tuple[int, int, int]]]
]:
    """
    Process raw UNet outputs to compute segmented predictions and event instances.

    Args:
    - raw_preds_dict (dict): Raw UNet outputs for each class.
    - input_movie (numpy.ndarray): Original input movie.
    - training_mode (bool): If True, separate events using a simpler algorithm.
    - debug (bool): If True, print debugging information.

    Returns:
    - Tuple: A tuple containing processed event instances, segmented predictions,
      and event coordinates.
    """
    raw_preds_list = list(raw_preds_dict.values())
    raw_preds_dict["background"] = 1 - np.sum(raw_preds_list, axis=0)

    assert (
        raw_preds_dict["background"].shape == input_movie.shape
    ), "U-Net outputs and input movie must have the same shape."

    # Get argmax segmentation using Otsu threshold on summed predictions
    preds_classes_dict, _ = get_otsu_argmax_segmentation(
        preds=raw_preds_dict, return_classes=True, debug=debug
    )

    # Separate events in predictions
    preds_instances_dict, coords_events = get_separated_events(
        argmax_preds=preds_classes_dict,
        movie=input_movie,
        debug=debug,
        training_mode=training_mode,
    )

    start = time.time()
    # Remove small events and merge events that belong together
    preds_instances_dict = remove_small_events(
        instances_dict=preds_instances_dict, new_id=0
    )

    for event_type in preds_instances_dict.keys():
        # Update segmented predicted masks accordingly
        preds_classes_dict[event_type] = preds_instances_dict[event_type].astype(bool)

        # Remove spark peak locations of sparks that have been removed
        if event_type in coords_events.keys():
            corrected_loc = []
            for t, y, x in coords_events[event_type]:
                if preds_instances_dict[event_type][t, y, x] != 0:
                    corrected_loc.append([t, y, x])
            coords_events[event_type] = corrected_loc

    # Renumber event instances so that each event has a unique ID
    shift_id = 0
    for event_type in preds_instances_dict.keys():
        preds_instances_dict[event_type] = renumber_labelled_mask(
            labelled_mask=preds_instances_dict[event_type], shift_id=shift_id
        )
        shift_id = max(shift_id, np.max(preds_instances_dict[event_type]))

    if debug:
        logger.debug(f"Time for removing small events: {time.time() - start:.2f} s")

    return preds_instances_dict, preds_classes_dict, coords_events


def preds_dict_to_mask(preds_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert a dict of binary masks representing a class of calcium release events
    to a single mask with values in their corresponding labels.

    Args:
    - preds_dict (dict): Dict with binary masks for each class.

    Returns:
    - np.ndarray: A mask with values representing different event classes.
    """
    labels_mask = np.zeros_like(list(preds_dict.values())[0], dtype=int)
    for event_type, event_label in config.classes_dict.items():
        if event_type not in config.event_types:
            continue
        labels_mask = np.where(preds_dict[event_type], event_label, labels_mask)
    return labels_mask


##################### Sparks processing in dataset labels ######################


# functions based on Miguel's idea (max * std for each pixel)
def detect_single_roi_peak(
    movie: np.ndarray, roi_mask: np.ndarray, max_filter_size: int = 10
) -> Tuple[int, int, int]:
    """
    Given a movie and a ROI, extract peak coordinates of the movie inside the ROI.

    Args:
    - movie (numpy.ndarray): Input movie (could be smoothed).
    - roi_mask (numpy.ndarray): Binary mask with the same shape as movie
        (containing one connected component).
    - max_filter_size (int): Size for the maximum filter.

    Returns:
    - tuple: Coordinates (t, y, x) of the movie's maximum inside the ROI.
    """
    roi_movie = np.where(roi_mask, movie, 0.0)

    # Compute max along t
    t_max = roi_movie.max(axis=0)
    # Compute standard deviation along t
    t_std = np.std(roi_movie, axis=0)
    # Multiply max and std
    prod = t_max * t_std

    # Maximum filter
    dilated = ndi.maximum_filter(prod, size=max_filter_size)
    # Find locations (y, x) of peaks
    argmaxima = np.logical_and(prod == dilated, prod != 0)
    argwhere = np.argwhere(argmaxima)

    if len(argwhere) != 1:
        raise ValueError(f"Found more than one peak in ROI: {len(argwhere)} peaks")

    # Find slice t corresponding to max location
    y, x = argwhere[0]
    t = np.argmax(roi_movie[:, y, x])

    return int(t), int(y), int(x)


# function based on max in ROI + gaussian smoothing
def detect_spark_peaks(
    movie: np.ndarray,
    instances_mask: np.ndarray,
    sigma: int = 2,
    max_filter_size: int = 10,
    return_mask: bool = False,
) -> Union[List[Tuple[int, int, int]], Tuple[List[Tuple[int, int, int]], np.ndarray]]:
    """
    Extract local maxima from input array (t,x,y) such that each ROI contains
    one peak (used for dataset processing, not for U-Net predictions processing).

    Args:
    - movie (numpy.ndarray): Input array.
    - instances_mask (numpy.ndarray): Mask with event ROIs.
    - sigma (int): Sigma parameter of the Gaussian filter.
    - max_filter_size (int): Dimension of the maximum filter size.
    - return_mask (bool): If True, return both masks with maxima and locations, if
      False, only returns locations.

    Returns:
    - list or tuple: List of peak coordinates (t, y, x) or a tuple containing
      the list of peak coordinates and the peaks mask if return_mask is True.
    """
    # Get a list of unique spark IDs
    event_list = list(np.unique(instances_mask))
    event_list.remove(0)

    if sigma > 0:
        # Smooth movie only on (y,x)
        smooth_movie = np.array(ndi.gaussian_filter(movie, sigma=(0, sigma, sigma)))
    else:
        smooth_movie = movie

    peaks_coords = []

    # Find one peak in each ROI
    for id_roi in event_list:
        t, y, x = detect_single_roi_peak(
            movie=smooth_movie,
            roi_mask=(instances_mask == id_roi),
            max_filter_size=max_filter_size,
        )

        peaks_coords.append((t, y, x))

    if return_mask:
        peaks_mask = np.zeros_like(instances_mask)
        peaks_mask[np.array(peaks_coords)] = 1
        return peaks_coords, peaks_mask

    return peaks_coords


def simple_nonmaxima_suppression(
    input_array: np.ndarray,
    mask: np.ndarray = np.array([]),
    min_distance: Union[int, np.ndarray] = 0,
    threshold: float = 0.5,
    sigma: float = 2.0,
) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    Extract local maxima from an input image.

    Args:
    - input_array (numpy.ndarray): Input array.
    - mask (numpy.ndarray, optional): Mask for limiting the search to specific
        regions.
    - min_distance (int, optional): Minimum distance between two peaks.
        locations.
    - threshold (float, optional): Minimum value of maximum points.
    - sigma (float, optional): Sigma parameter of the Gaussian filter.

    Returns:
    - tuple containing the peak coordinates and the maxima mask.
    """
    input_array = input_array.astype(np.float64)

    # Handle min_distance as a connectivity mask
    if np.isscalar(min_distance):
        if min_distance == 0:
            min_distance = 1
        else:
            c_min_dist = ndi.generate_binary_structure(input_array.ndim, min_distance)
    else:
        c_min_dist = np.array(min_distance, bool)
        if c_min_dist.ndim != input_array.ndim:
            raise ValueError("Connectivity nb of dimensions must be same as input")

    if sigma > 0:
        if mask.size > 0:
            # Keep only the region inside the mask
            masked_img = np.where(mask, input_array, 0.0)

            # Smooth the masked input image
            smooth_img = ndi.gaussian_filter(masked_img, sigma=sigma)
        else:
            smooth_img = ndi.gaussian_filter(input_array, sigma=sigma)
    else:
        smooth_img = input_array

    if mask.size > 0:
        # Hypothesis: maxima belong to the maxima mask
        smooth_img = np.where(mask, smooth_img, 0.0)

    # Search for local maxima
    dilated = ndi.maximum_filter(smooth_img, footprint=min_distance)
    smooth_img = smooth_img.astype(np.float64)
    is_local_maxima = np.logical_and(smooth_img == dilated, smooth_img > threshold)

    # Get list containing the coordinated of the peaks in is_local_maxima
    peak_coordinates = list(np.argwhere(is_local_maxima))

    return peak_coordinates, is_local_maxima


############################ Statistical functions #############################

# https://mverbakel.github.io/2021-02-24/non-inferiority-test
# https://docs.scipy.org/doc/scipy/reference/stats.html


def one_sided_non_inferiority_ttest(
    sample1: np.ndarray, sample2: np.ndarray, increase_is_desirable: bool = True
) -> Tuple[float, float]:
    """
    Perform a one-sided non-inferiority t-test for two independent samples.

    Args:
        sample1 (array-like): First sample data.
        sample2 (array-like): Second sample data.
        increase_is_desirable (bool): True if an increase in sample2 is considered desirable,
                                      False if a decrease is desirable.

    Returns:
        tuple: A tuple containing the t-statistic and one-sided p-value.
    """
    t_statistic, two_sided_p_value = ttest_rel(a=sample1, b=sample2)

    if increase_is_desirable:
        one_sided_p_value = two_sided_p_value / 2.0
    else:
        one_sided_p_value = 1 - two_sided_p_value / 2.0

    return t_statistic, one_sided_p_value


######################### Signal-to-Noise Ratio (SNR) ##########################


def get_cell_mask(x: np.ndarray) -> np.ndarray:
    """
    Given the original recording and the mask with segmented events,
    return a mask with the cell body.

    Args:
        x (numpy.ndarray): Original recording.

    Returns:
        numpy.ndarray: Mask with the cell body.
    """

    # Compute mean frame
    if x.ndim >= 3:
        x_mean = np.mean(np.array(x), axis=0)
    else:
        x_mean = np.array(x)

    # Normalize x_mean between 0 and 1
    x_mean = (x_mean - np.min(x_mean)) / (np.max(x_mean) - np.min(x_mean))

    percentile_high = np.percentile(x_mean, 99)
    percentile_low = np.percentile(x_mean, 1)
    percentile_mask = (x_mean < percentile_high) & (x_mean > percentile_low)

    # Apply otsu thresholding to percentile_mask -> Canny edges
    otsu_threshold = threshold_otsu(x_mean[percentile_mask])
    otsu_mask = x_mean > otsu_threshold
    otsu_mask = binary_fill_holes(otsu_mask).astype(bool)

    # Sobel edge detection
    sobel_mask = x_mean > np.percentile(x_mean, 50)
    sobel_mask = binary_fill_holes(sobel_mask).astype(bool)

    # Combine masks
    threshold_mask = np.logical_or(otsu_mask, sobel_mask)

    # keep only largest connected component
    threshold_mask = label(threshold_mask)
    props = regionprops(threshold_mask)
    areas = [prop.area for prop in props]
    largest_component = np.argmax(areas) + 1
    threshold_mask = threshold_mask == largest_component

    # Compute mask's binary closing
    threshold_mask = binary_closing(threshold_mask, disk(2))
    threshold_mask = binary_fill_holes(threshold_mask)

    return threshold_mask


def compute_snr(
    x: np.ndarray,
    y: np.ndarray,
    event_roi: np.ndarray = np.array([]),
    percentile: float = 99,
) -> float:
    """
    Compute the signal-to-noise ratio given the original recording and the mask
    with segmented events.
    Args:
        x (numpy.ndarray): Original recording.
        y (numpy.ndarray): Mask with segmented events.
        event_roi (numpy.ndarray): Mask with the region of interest (ROI) where
            events are expected.
        percentile (float): Percentile used to estimate the noise standard
            deviation.

    Returns:
        float: Signal-to-noise ratio.
    """
    # Convert the image series to double
    x = x.astype(float)

    # Create a cell mask from the mask y that includes both the cell body and events
    cell_mask = get_cell_mask(x)
    cell_mask = cell_mask | np.any(y, axis=0)

    # Create a repeated cell mask for each frame
    if cell_mask.ndim >= 3:
        cell_mask = np.repeat(cell_mask[np.newaxis, :, :], x.shape[0], axis=0)
    else:
        cell_mask = cell_mask

    # Create background mask
    background_mask = cell_mask & (y == 0)

    # Get event labels
    event_labels = np.unique(y[y != config.ignore_index]).tolist()
    event_labels.remove(0)

    # Calculate the 99th percentile of x within the events mask
    if event_roi.size == 0:
        percentile_mask = np.isin(y, event_labels)
    else:
        percentile_mask = event_roi

    avg_events = np.percentile(x[percentile_mask], percentile)
    # print(f"\t99th percentile of events: {avg_events}")

    # Calculate the average baseline from the cell area without events
    avg_baseline = np.mean(x[background_mask])
    # print(f"\tAverage baseline: {avg_baseline}")

    # Estimate noise standard deviation from the cell area without events
    sd_noise = np.std(x[background_mask])
    # print(f"\tStandard deviation of noise: {sd_noise}")

    # Calculate SNR
    snr = (avg_events - avg_baseline) / sd_noise

    return snr


####################### Other functions (from notebooks) #######################


def get_event_parameters_simple(
    event_mask: np.ndarray,
) -> Tuple[int, int, float, float]:
    """
    Get event parameters (start and end frames, and center of mass) from an
    event mask.

    Args:
        event_mask (numpy.ndarray): The ROI denoting an event in the movie.

    Returns:
        tuple or list: Event parameters.
    """
    # Find non-zero pixels in the event mask
    nonzero_pixels = np.transpose(np.nonzero(event_mask))

    # Extract frame numbers
    frames = np.unique(nonzero_pixels[:, 0])

    # Get start and end frames
    start_frame, end_frame = frames[0], frames[-1]

    # Calculate center of mass in the first frame
    y_center, x_center = center_of_mass(event_mask[start_frame])

    return (
        start_frame,
        end_frame,
        round(float(x_center), 2),
        round(float(y_center), 2),
    )


def get_event_parameters(
    event_mask: np.ndarray,
) -> List[List[List[int]]]:
    """
    Get event parameters (list of coordinates and frame numbers) from an
    event mask.

    Args:
        event_mask (numpy.ndarray): The ROI denoting an event in the movie.

    Returns:
        tuple or list: Event parameters.
    """
    # Find non-zero pixels in the event mask
    nonzero_pixels = np.transpose(np.nonzero(event_mask))

    # Extract frame numbers
    frames = np.unique(nonzero_pixels[:, 0])

    # Create a list of lists for each frame
    coord_list = []

    for frame in frames:
        frame_mask = event_mask[frame]
        y_array, x_array = np.where(frame_mask)
        coord_list.append([x_array.tolist(), y_array.tolist(), frame])

    return coord_list


def moving_average(movie: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute moving average of a movie.

    Args:
        movie (numpy.ndarray): Input movie.
        k (int): Window size (must be an odd number).

    Returns:
        numpy.ndarray: Rolling average of the movie along the first axis (frames).
    """
    assert k % 2 == 1, "Window size (k) must be an odd number."

    diff = k // 2
    avg_movie = np.zeros_like(movie, dtype=float)

    for frame_idx in range(len(movie)):
        start_idx = max(0, frame_idx - diff)
        end_idx = min(len(movie), frame_idx + diff + 1)
        window = movie[start_idx:end_idx]
        avg_movie[frame_idx] = np.mean(window, axis=0)

    return avg_movie


def count_classes_in_roi(
    events_mask: np.ndarray, classes_mask: np.ndarray, event_id: int
) -> int:
    """
    Count the number of unique classes within an event in the given masks.

    Args:
        events_mask (numpy.ndarray): Mask denoting events.
        classes_mask (numpy.ndarray): Mask denoting classes.
        event_id (int): ID of the event to count classes for.

    Returns:
        int: Number of unique classes within the specified event.
    """
    event_class_mask = np.where(events_mask == event_id, classes_mask, 0)

    unique_classes = np.unique(event_class_mask)
    unique_classes = unique_classes[unique_classes != 0]

    n_classes = len(unique_classes)

    if n_classes > 1:
        class_list = ", ".join(map(str, unique_classes))
        print(f"Event {event_id} contains {n_classes} classes: {class_list}")

    return n_classes


############################### Unused functions ###############################


def compute_filtered_butter(
    movie_array: np.ndarray,
    min_prominence: int = 2,
    band_stop_width: int = 2,
    min_freq: int = 7,
    filter_order: int = 4,
    Fs: int = 150,
    debug: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Apply Butterworth filter to input movie.

    movie_array: input movie to be filtered
    min_prominence: minimal prominence of filtered peaks in frequency domain
    band_stop_width: width of the filtered band for each peak
    min_freq: minimal frequence that can be filtered (???)
    filter_order: order of Butterworth filter
    Fs: sampling frequency [Hz]

    output: filtered version of input movie
    """

    # sampling period [s]
    T = 1 / Fs
    # signal's length [s]
    L = movie_array.shape[0]
    # time vector
    t = np.arange(L) / Fs

    # movie's signal average along time (time profile of image series)
    movie_average = np.mean(movie_array, axis=(1, 2))

    # get noise frequencies
    # compute Fourier transform
    fft = np.fft.fft(movie_average)
    # compute two-sided spectrum
    P2 = np.abs(fft / L)
    # compute single-sided spectrum
    P1 = P2[: (L // 2)]
    P1[1:-1] = 2 * P1[1:-1]

    freqs = np.fft.fftfreq(L, d=1 / Fs)
    f = freqs[: L // 2]

    # detrend single-sided spectrum
    # P1_decomposed = seasonal_decompose(P1, model='additive', period=1) # don't know period
    # P1_detrend = signal.detrend(P1) # WRONG??

    # set spectrum corresponding to frequencies lower than min freq to zero
    P1_cut = np.copy(P1)
    P1_cut[:min_freq] = 0

    # find peaks in spectrum
    peaks = signal.find_peaks(P1)[0]  # coords in P1 of peaks
    # peaks = signal.find_peaks(P1_detrend)[0] # need first to detrend data properly
    peaks_cut = peaks[peaks >= min_freq]  # coords in P1_cut of peaks

    # compute peaks prominence
    prominences = signal.peak_prominences(P1_cut, peaks_cut)[0]

    # keep only peaks with prominence large enough
    prominent_peaks = peaks_cut[prominences > min_prominence]

    # regions to filter
    bands_low = prominent_peaks - band_stop_width
    bands_high = prominent_peaks + band_stop_width
    bands_indices = np.transpose([bands_low, bands_high])

    bands_freq = f[bands_indices]

    # make sure that nothing is outside interval (0,max(f))
    if bands_freq.size > 0:
        bands_freq[:, 0][bands_freq[:, 0] < 0] = 0
        bands_freq[:, 1][bands_freq[:, 1] > max(f)] = (
            max(f) - np.mean(np.diff(f)) / 1000
        )

    # create butterworth filter
    filter_type = "bandstop"
    filtered = np.copy(movie_array)

    for i, band in enumerate(bands_freq):
        Wn = band / max(f)

        sos = signal.butter(N=filter_order, Wn=Wn, btype=filter_type, output="sos")

        filtered = signal.sosfiltfilt(sos, filtered, axis=0)

    if debug:
        # filtered movie's signal average along time (time profile of image series)
        filtered_movie_average = np.mean(filtered, axis=(1, 2))

        # get frequencies of filtered movie
        # compute Fourier transform
        filtered_fft = np.fft.fft(filtered_movie_average)
        # compute two-sided spectrum
        filtered_P2 = np.abs(filtered_fft / L)
        # compute single-sided spectrum
        filtered_P1 = filtered_P2[: (L // 2)]
        filtered_P1[1:-1] = 2 * filtered_P1[1:-1]

        # detrend single-sided spectrum
        # filtered_P1_detrend = signal.detrend(filtered_P1) # WRONG??

        return filtered, movie_average, filtered_movie_average, Fs, f, P1, filtered_P1

    return filtered
