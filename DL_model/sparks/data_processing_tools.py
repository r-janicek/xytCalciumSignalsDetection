"""
20.10.2022

Script with functions for any type of data processing in the project
(annotations, predictions, sample movies, ...)

REMARKS
24.10.2022: functions that aren't currently used are commented out and put at
the end of the script (such as compute_filtered_butter, )
"""

import logging
import math
import os
import time

import cc3d
import imageio
import numpy as np
from scipy import fftpack
from scipy import ndimage as ndi
from scipy import signal, spatial
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage import morphology
from skimage.draw import ellipsoid
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

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


########################## Annotations preprocessing ###########################


def keep_percentile(movie, roi_mask, k=75):
    """
    For a given event ROI, keep only points above k-th percentile.

    movie:                 input movie (possibly smoothed version)
    roi_mask:              ROI corrisponding to one event
    k:                     value of percentile
    keep_border:           if True, sum values of mask before dilation +
                           erosion to final mask
    dilation_erosion_iter: number of dilation and erosion iterations

    return:                new ROI corresponding to event
    """

    # extract movie ROI to event
    movie_event = np.where(roi_mask, movie, 0)

    # compute percentile
    movie_event_values = movie_event[np.nonzero(movie_event)]
    percentile = np.percentile(movie_event_values, k)

    new_roi_mask = movie_event >= percentile

    # if any of the new ROI dims is shorter than 3 pixels, first dilate it
    roi_sizes = np.max(np.where(new_roi_mask), axis=1) - np.min(
        np.where(new_roi_mask), axis=1
    )
    if roi_sizes.min() < 2:
        new_roi_mask = ndi.binary_dilation(new_roi_mask, iterations=1)
    new_roi_mask = get_convex_hull(new_roi_mask)[0]
    return new_roi_mask


def reduce_sparks_size(movie, class_mask, event_mask, sigma=2, k=75):
    """
    Reduce sparks dimension in class annotation mask.
    movie :                input array
    event_mask :           mask with identified events
    class_mask:            mask with classified events
    sigma :                sigma parameter of gaussian filter
    k:                     value of percentile

    return: class mask where removed part is labelled as undefined (4)
    """
    # normalise input movie between 0 and 1
    if movie.max() > 1:
        movie = (movie - movie.min()) / (movie.max() - movie.min())

    # get sparks event mask
    spark_mask = np.where(class_mask == 1, event_mask, 0)

    # get list of sparks IDs
    event_list = list(np.unique(spark_mask))
    event_list.remove(0)

    # smooth movie
    if sigma > 0:
        smooth_movie = ndi.gaussian_filter(movie, sigma=sigma)

    # new events mask using percentile method
    new_class_mask = np.copy(class_mask)
    # set spark ROIs to 4
    new_class_mask[new_class_mask == 1] = 4

    # reduce size of each ROI
    for id_roi in event_list:
        roi_mask = spark_mask == id_roi

        # reduce sparks size dimension wrt to percentile
        new_roi_mask = keep_percentile(movie=movie, roi_mask=roi_mask, k=k)
        # set new smaller spark peak to 1
        new_peak = np.logical_and(roi_mask, new_roi_mask)
        new_class_mask[new_peak] = 1

    assert np.all(class_mask.astype(bool) == new_class_mask.astype(bool))

    return new_class_mask


def final_mask(mask, radius1=2.5, radius2=3.5, ignore_ind=2):  # SLOW
    """
    add annotation region around spark peaks
    """
    dt = ndi.distance_transform_edt(1 - mask)
    new_mask = np.zeros(mask.shape, dtype=np.int64)
    new_mask[dt < radius2] = ignore_ind
    new_mask[dt < radius1] = 1

    return new_mask


def get_new_mask(
    video,
    mask,
    min_dist_xy=MIN_DIST_XY,
    min_dist_t=MIN_DIST_T,
    radius_event=3,
    radius_ignore=2,
    ignore_index=4,
    sigma=2,
    return_loc=False,
    return_loc_and_mask=False,
    ignore_frames=0,
):
    """
    from raw segmentation masks get masks where sparks are annotated by peaks
    """

    # get spark centres
    if 1 in mask:
        sparks_maxima_mask = np.where(mask == 1, 1, 0)

        # compute sparks connectivity mask
        connectivity_mask = sparks_connectivity_mask(min_dist_xy, min_dist_t)

        sparks_loc, sparks_mask = simple_nonmaxima_suppression(
            img=video,
            maxima_mask=sparks_maxima_mask,
            min_dist=connectivity_mask,
            return_mask=True,
            threshold=0,
            sigma=sigma,
        )

        logger.debug(f"Num of sparks: {len(sparks_loc)}")
        # print(sparks_loc)

        if return_loc:
            if ignore_frames > 0:
                # remove sparks from locations list
                mask_duration = mask.shape[0]
                sparks_loc = empty_marginal_frames_from_coords(
                    coords=sparks_loc, n_frames=ignore_frames, duration=mask_duration
                )
            return sparks_loc

        if ignore_frames > 0:
            # remove sparks from maxima mask
            sparks_mask = empty_marginal_frames(sparks_mask, ignore_frames)

        sparks_mask = final_mask(
            sparks_mask,
            radius1=radius_event,
            radius2=radius_event + radius_ignore,
            ignore_ind=ignore_index,
        )

        # remove sparks from old mask
        no_sparks_mask = np.where(mask == 1, 0, mask)

        # create new mask
        new_mask = np.where(sparks_mask != 0, sparks_mask, no_sparks_mask)

        if return_loc_and_mask:
            if ignore_frames > 0:
                # remove sparks from locations list
                mask_duration = mask.shape[0]
                sparks_loc = empty_marginal_frames_from_coords(
                    coords=sparks_loc, n_frames=ignore_frames, duration=mask_duration
                )
            return sparks_loc, new_mask

        else:
            return new_mask

    else:
        if return_loc:
            return []
        elif return_loc_and_mask:
            return [], mask
        else:
            return mask


def get_new_mask_raw_sparks(
    mask,
    radius_ignore_sparks=1,
    radius_ignore_puffs=3,
    radius_ignore_waves=5,
    ignore_index=4,
):
    """
    from raw segmentation masks get masks where each event has an ignore region
    around itself
    """

    ignore_mask_sparks = None
    if 1 in mask:
        sparks_mask = np.where(mask == 1, 1, 0)
        dilated_mask = ndi.binary_dilation(
            sparks_mask, iterations=radius_ignore_sparks)
        eroded_mask = ndi.binary_erosion(
            sparks_mask, iterations=radius_ignore_sparks)
        ignore_mask_sparks = np.logical_xor(dilated_mask, eroded_mask)
        # imageio.volwrite("TEST_IGNORE_MASK_SPARKS.tif", np.uint8(ignore_mask_sparks))

    ignore_mask_waves = None
    if 2 in mask:
        waves_mask = np.where(mask == 2, 1, 0)
        dilated_mask = ndi.binary_dilation(
            waves_mask, iterations=radius_ignore_waves)
        eroded_mask = ndi.binary_erosion(
            waves_mask, iterations=radius_ignore_waves)
        ignore_mask_waves = np.logical_xor(dilated_mask, eroded_mask)
        # imageio.volwrite("TEST_IGNORE_MASK_WAVES.tif", np.uint8(ignore_mask_waves))

    ignore_mask_puffs = None
    if 3 in mask:
        puffs_mask = np.where(mask == 3, 1, 0)
        dilated_mask = ndi.binary_dilation(
            puffs_mask, iterations=radius_ignore_puffs)
        eroded_mask = ndi.binary_erosion(
            puffs_mask, iterations=radius_ignore_puffs)
        ignore_mask_puffs = np.logical_xor(dilated_mask, eroded_mask)
        # imageio.volwrite("TEST_IGNORE_MASK_PUFFS.tif", np.uint8(ignore_mask_puffs))

    if ignore_mask_sparks is not None:
        mask = np.where(ignore_mask_sparks, ignore_index, mask)
    if ignore_mask_puffs is not None:
        mask = np.where(ignore_mask_puffs, ignore_index, mask)
    if ignore_mask_waves is not None:
        mask = np.where(ignore_mask_waves, ignore_index, mask)

    return mask

    # remove sparks from old mask
    no_sparks_mask = np.where(mask == 1, 0, mask)

    # create new mask
    new_mask = np.where(sparks_mask != 0, sparks_mask, no_sparks_mask)

    return new_mask


########################### General masks processing ###########################


def class_to_nb(event_type):
    r"""
    Given an ca release event type (sparks, puffs, waves or ignore)
    return corresponding int used in annotation masks.
    """

    assert event_type in [
        "sparks",
        "puffs",
        "waves",
        "ignore",
    ], "event_type must be in ['sparks', 'puffs', 'waves', 'ignore']."

    class_to_nb_dict = {"sparks": 1, "puffs": 3, "waves": 2, "ignore": 4}

    return class_to_nb_dict[event_type]


def sparks_connectivity_mask(min_dist_xy=MIN_DIST_XY, min_dist_t=MIN_DIST_T):
    """
    Compute the mask that defines the minimal distance between two spark peaks.
    """
    # compute sparks connectivity mask
    radius = math.ceil(min_dist_xy / 2)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    disk = x**2 + y**2 <= radius**2
    connectivity_mask = np.stack([disk] * (min_dist_t), axis=0)

    return connectivity_mask


def empty_marginal_frames(video, n_frames):
    """
    Set first and last n_frames of a video to zero.
    """
    if n_frames > 0:
        new_video = video[n_frames:-n_frames]
        new_video = np.pad(
            new_video, ((n_frames,), (0,), (0,)), mode="constant")
    else:
        new_video = video

    assert np.shape(video) == np.shape(new_video)

    return new_video


def empty_marginal_frames_from_coords(coords, n_frames, duration):
    """
    Remove sparks 'coords' located in first and last 'n_frames' of a video of
    duration 'duration'.
    """
    if n_frames > 0:
        if len(coords) > 0:
            n_frames_up = duration - n_frames

            if type(coords[0]) != list:
                [loc.tolist() for loc in coords]

            new_coords = [
                loc for loc in coords if loc[0] >= n_frames and loc[0] < n_frames_up
            ]
            return new_coords

    return coords


def get_convex_hull(image):
    """
    Compute convex hull of a Numpy array.
    """
    points = np.transpose(np.where(image))
    hull = spatial.ConvexHull(points)
    deln = spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


def get_smallest_event(events_mask, event_type):
    r"""
    Given the mask containing the instances of events in a given class,
    get the dimension of the 'smallest' event.

    If event_type == 'puffs': get shortest event duration
    If event_type == 'sparks': get smallest and shortest event values
    """
    assert (
        event_type == "sparks" or event_type == "puffs"
    ), f"Function is not defined for class {event_type}"

    shortest_event = 9999
    smallest_event = 9999

    if events_mask.any():
        event_slices = ndi.measurements.find_objects(events_mask)
        events_durations = [
            event_slice[0].stop - event_slice[0].start
            for event_slice in event_slices
            if event_slice is not None
        ]
        shortest_event = np.min(events_durations)

        if event_type == "sparks":
            events_widths = [
                min(
                    event_slice[1].stop - event_slice[1].start,
                    event_slice[2].stop - event_slice[2].start,
                )
                for event_slice in event_slices
            ]
            smallest_event = np.min(events_widths)

    if event_type == "sparks":
        return shortest_event, smallest_event
    elif event_type == "puffs":
        return shortest_event


######################### UNet predictions processing ##########################


def get_argmax_segmented_output(preds, get_classes=True):
    """
    preds are the (exponential) raw outputs of the unet for each class:
    [background, sparks, waves, puffs] (4 x duration x 64 x 512)
    """

    argmax_classes = np.argmax(preds, axis=0)

    if not get_classes:
        return argmax_classes

    preds = {}
    preds["sparks"] = np.where(argmax_classes == 1, 1, 0)
    preds["waves"] = np.where(argmax_classes == 2, 1, 0)
    preds["puffs"] = np.where(argmax_classes == 3, 1, 0)

    # imageio.volwrite("TEST_argmax.tif", np.uint8(argmax_classes))

    return preds, argmax_classes


def get_argmax_segmentation_otsu(preds, get_classes=True, debug=False):
    """
    preds are the (exponential) raw outputs of the unet for each class:
    dict with keys 'sparks', 'puffs', 'waves', 'background'

    compute otsu threshold with respect to the sum of positive predictions
    (i.e., sparks+puffs+waves) and remove preds below that threshold,
    then get argmax predictions on thresholded UNet output

    if get_classes==False, return an array with values in {0,1,2,3}
    if get_classes==True, return a pair (argmax_preds, classes_preds) where
    argmax_preds is a dict with keys 'sparks', 'waves' and 'puffs' and
    classes_preds is an array with values in {0,1,2,3}
    """

    # compute threshold on summed predicted events
    sum_preds = 1 - preds["background"]  # everything but the background
    # check if sum_preds contains nan
    # logger.info("Checking if UNet predictions contain nan values...")
    # if np.isnan(sum_preds).any():
    # logger.warning(
    # "UNet predictions contain nan values, replacing them with 0.")
    # sum_preds = np.nan_to_num(sum_preds)
    # logger.info("sum_preds shape: " + str(sum_preds.shape))
    # logger.info("sum_preds min: " + str(sum_preds.min()))
    # logger.info("sum_preds max: " + str(sum_preds.max()))
    # logger.info("sum_preds dtype: " + str(sum_preds.dtype))

    t_otsu = threshold_otsu(sum_preds)
    if debug:
        logger.debug(f"Events detection threshold: {t_otsu:.3f}")

    # get binary mask of valid predictions
    binary_sum_preds = sum_preds > t_otsu

    # mask out removed events from UNet preds
    masked_class_preds = binary_sum_preds * (
        [preds["background"], preds["sparks"], preds["waves"], preds["puffs"]]
    )

    # get argmax of classes
    return get_argmax_segmented_output(
        preds=masked_class_preds, get_classes=get_classes
    )


def get_separated_events(
    argmax_preds,
    movie,
    sigma,
    connectivity,
    connectivity_mask,
    return_sparks_loc=False,
    debug=False,
    training_mode=False
):
    """
    Given the segmented output, separate each class into event instances.

    Return a dict with keys 'sparks', 'puffs' and 'waves' where each entry is an
    array with labelled events (from 1 to n_events).

    Using watershed separation algorithm to separate spark events.

    movie:              input movie
    sigma:              sigma valued used for nonmaxima suppression and
                        watershed separation of sparks
    argmax_preds:       segmented UNet output (dict with keys 'sparks', 'puffs',
                        'waves')
    connectivity:       int, define how puffs and waves are separated
    connectivity_mask:  3d matrix, define how sparks are separated
    return_sparks_loc:  bool, if True, return spark peaks locations together with
                        separated events dict
    debug:              bool, if True, print intermediate results
    training_mode:      bool, if True, separate events using a simpler algorithm
    """

    # separate CCs in puff and wave classes
    ccs_class_preds = {
        class_name: cc3d.connected_components(
            class_argmax_preds, connectivity=connectivity, return_N=False
        )
        for class_name, class_argmax_preds in argmax_preds.items()
        if class_name != "sparks"
    }

    # compute spark peaks locations
    loc, mask_loc = simple_nonmaxima_suppression(
        img=movie,
        maxima_mask=argmax_preds["sparks"],
        min_dist=connectivity_mask,
        return_mask=True,
        threshold=0.0,
        sigma=sigma,
    )

    logger.debug(
        f"Number of sparks detected by nonmaxima suppression: {len(loc)}")

    # compute smooth version of input video
    smooth_xs = ndi.gaussian_filter(movie, sigma=sigma)

    # compute watershed separation
    markers, _ = ndi.label(mask_loc)

    split_event_mask = watershed(
        image=-smooth_xs,
        markers=markers,
        mask=argmax_preds["sparks"],
        connectivity=3,
        compactness=1,
    )

    if not training_mode:
        # labelling sparks with peaks in all connected components only if not training
        # otherwise, it is not very important

        # check if all connected components have been labelled
        all_ccs_labelled = np.all(
            split_event_mask.astype(
                bool) == argmax_preds["sparks"].astype(bool)
        )

        if not all_ccs_labelled:
            if debug:
                logger.debug(
                    "Not all sparks were labelled, computing missing events...")
                logger.debug(
                    f"Number of sparks before correction: {np.max(split_event_mask)}")

            # get number of labelled events
            n_split_events = np.max(split_event_mask)

            # if not all CCs have been labelled, obtain unlabelled CCs and split them
            missing_sparks = np.logical_xor(
                split_event_mask.astype(
                    bool), argmax_preds["sparks"].astype(bool)
            )

            # separate unlabelled CCs and label them
            labelled_missing_sparks = cc3d.connected_components(
                missing_sparks, connectivity=connectivity, return_N=False
            )

            # increase labels by number of sparks already present
            labelled_missing_sparks = np.where(
                labelled_missing_sparks, labelled_missing_sparks + n_split_events, 0
            )

            # merge sparks with peaks and sparks without them
            split_event_mask += labelled_missing_sparks

            # get peak location of missing sparks and add it to peaks lists
            missing_sparks_ids = list(np.unique(labelled_missing_sparks))
            missing_sparks_ids.remove(0)
            for spark_id in missing_sparks_ids:
                spark_roi_xs = np.where(
                    labelled_missing_sparks == spark_id, smooth_xs, 0)

                peak_loc = np.unravel_index(
                    spark_roi_xs.argmax(), spark_roi_xs.shape)

                loc.append(list(peak_loc))

            # assert that now all CCs have been labelled
            all_ccs_labelled = np.all(
                split_event_mask.astype(
                    bool) == argmax_preds["sparks"].astype(bool)
            )

            if debug:
                logger.debug(
                    f"Number of sparks after correction: {np.max(split_event_mask)}")

        assert all_ccs_labelled, "Some sparks CCs haven't been labelled!"

    if debug:
        # check that event IDs are ordered and consecutive
        assert len(np.unique(split_event_mask)) - 1 == np.max(
            split_event_mask
        ), f"spark IDs are not consecutive: {np.unique(split_event_mask)}"
        assert len(np.unique(ccs_class_preds["puffs"])) - 1 == np.max(
            ccs_class_preds["puffs"]
        ), f"puff IDs are not consecutive: {np.unique(ccs_class_preds['puffs'])}"
        assert len(np.unique(ccs_class_preds["waves"])) - 1 == np.max(
            ccs_class_preds["waves"]
        ), f"wave IDs are not consecutive: {np.unique(ccs_class_preds['waves'])}"

    separated_events = {
        "sparks": split_event_mask,
        "puffs": ccs_class_preds["puffs"],
        "waves": ccs_class_preds["waves"],
    }

    if return_sparks_loc:
        return separated_events, loc

    return separated_events


###################### Event instances' masks processing #######################

# Functions related to the processing of masks with labelled event instances


def renumber_labelled_mask(labelled_mask, shift_id=0):
    r"""
    labelled_mask: numpy array with labelled events (positive integers)
    shift_id: shift each label by this integer

    return mask where such events are numbered in a consecutive way
    """

    if labelled_mask.max() > 0:
        # renumber labelled events
        old_labels = list(np.unique(labelled_mask))
        old_labels.remove(0)
        new_mask = np.zeros_like(labelled_mask)

        for new_id, old_id in enumerate(old_labels):
            new_mask[labelled_mask == old_id] = new_id + shift_id + 1

        # check that the number of events hasn't changed
        new_labels = np.unique(new_mask)
        assert (
            len(old_labels) == np.max(new_labels) - shift_id
        ), f"New labels are wrong: {new_labels}"
    else:
        new_mask = labelled_mask

    return new_mask


def get_event_instances_class(
    event_instances, class_labels, shift_ids=False, to_tensor=False
):
    r"""
    From the array containing the unclassified event instances and the array
    containing the class of each event, get a dictionary (one entry for each
    class: 'sparks', 'puffs', 'waves', 'ignore') of classified event instances.

    event_instances is an array with int values
    class_labels is an array with values in {0,1,2,3,4}
    if shift_ids == True, events in different classes have different IDs
    if to_tensor == True, return dict of torch tensors, instead of numpy arrays
    """
    temp_events = {}
    shift_id = 0

    for event_type in ["sparks", "puffs", "waves", "ignore"]:
        # get binary annotated mask of current event type
        class_mask = class_labels == class_to_nb(event_type)

        # get separated events belonging to this class
        event_mask = np.where(class_mask, event_instances, 0)

        # renumber annotated events
        temp_events[event_type] = renumber_labelled_mask(
            labelled_mask=event_mask, shift_id=shift_id
        )

        if shift_ids:
            shift_id = max(shift_id, np.max(temp_events[event_type]))

    return temp_events


# this fct is especially used for predicted puff instances
def merge_labels(labelled_mask, max_gap):
    """
    merge labels in input mask, if their distance in time is
    smaller than the max gap
    """
    # only run code if labelled_mask is non-empty

    if labelled_mask.any():
        # dilate annotations along time
        struct = np.zeros((max_gap + 1, 3, 3))
        struct[:, 1, 1] = 1
        dilated_labels_t = ndi.binary_dilation(labelled_mask, structure=struct)

        # count if number of waves has decreased
        conn = 26
        merged_labels, n_merged_labels = cc3d.connected_components(
            dilated_labels_t, connectivity=conn, return_N=True
        )

        # if some labels have merged, re-label mask accordingly
        if n_merged_labels < len(np.unique(labelled_mask)) - 1:
            # logger.debug(f"Merging events since their gap distance is below {max_gap}")
            # logger.debug(f"Labels before merging: {np.unique(labelled_mask)}")

            merged_labelled_mask = np.where(labelled_mask, merged_labels, 0)
            # logger.debug(f"Labels after merging: {np.unique(merged_labelled_mask)}")
            return merged_labelled_mask
        else:
            return labelled_mask

    else:
        return labelled_mask


def remove_small_events(
    class_instances, class_type, min_t=None, min_width=None, max_gap=None
):
    r"""
    Remove small predicted events and merge events belonging together.
    """

    if class_type in ["puffs", "waves"]:
        # merge events that belong together
        assert max_gap is not None, "Provide 'max_gap' with puffs and waves."
        class_instances = merge_labels(
            labelled_mask=class_instances, max_gap=max_gap)

    # remove small events
    events_ids = list(np.unique(class_instances))
    events_ids.remove(0)

    for idx in events_ids:
        event_roi = class_instances == idx
        slices = ndi.measurements.find_objects(event_roi)[0]

        if class_type in ["sparks", "waves"]:
            assert min_width is not None, "Provide 'min_width' with sparks and waves."

            # if event is too small, remove it from predictions
            if class_type == "sparks":
                event_width = min(
                    slices[1].stop - slices[1].start, slices[2].stop -
                    slices[2].start
                )
            elif class_type == "waves":
                event_width = slices[2].stop - slices[2].start
            if event_width < min_width:
                # logger.debug(f"Removing event labelled with {idx} (too small)")
                class_instances = np.where(event_roi, 0, class_instances)

        if class_type in ["sparks", "puffs"]:
            assert min_t is not None, "Provide 'min_t' with sparks and puffs."

            # if event is too short, remove it from predictions
            event_t = slices[0].stop - slices[0].start
            if event_t < min_t:
                # logger.debug(f"Removing event labelled with {idx} (too short)")
                class_instances = np.where(event_roi, 0, class_instances)

    return class_instances


def get_processed_result(
    sparks,
    puffs,
    waves,
    xs,
    conn_mask,
    connectivity,
    max_gap,
    sigma,
    wave_min_width,
    puff_min_t,
    spark_min_t,
    spark_min_width,
    training_mode=False,
    debug=False
):
    """
    Starting from raw UNet outputs of a given video, compute
    - segmented predictions
    - instances of events in predictions
    removing artifacts (e.g., small objects) if not in training mode

    sparks, puffs, waves: raw UNet outputs (values between 0 and 1)
    xs: original movie
    conn_mask: mask defining when sparks are connected
    connectivity: integer defining how puffs and waves are connected
    max_gap: number of consecutive empty frames in an event (puffs and waves)
    sigma: sigma valued used for xs smoothing in spark peaks detection
    training_mode: bool, if True, separate events using a simpler algorithm
    """
    preds = {
        "sparks": sparks,
        "puffs": puffs,
        "waves": waves,
        "background": 1 - sparks - puffs - waves,
    }

    # Get argmax segmentation
    # Using Otsu threshold on summed predictions
    # preds_segmentation is a dict with binary mask for each class
    # _ has values in {0,1,2,3}
    preds_segmentation, _ = get_argmax_segmentation_otsu(
        preds=preds, get_classes=True, debug=debug
    )

    # Separate events in predictions
    preds_instances, sparks_loc = get_separated_events(
        argmax_preds=preds_segmentation,
        movie=xs,
        sigma=sigma,
        connectivity=connectivity,
        connectivity_mask=conn_mask,
        return_sparks_loc=True,
        debug=debug,
        training_mode=training_mode
    )

    start = time.time()
    # Remove small events and merge events that belong together
    # Waves
    preds_instances["waves"] = remove_small_events(
        class_instances=preds_instances["waves"],
        class_type="waves",
        min_width=wave_min_width,
        max_gap=max_gap,
    )

    # update segmented predicted mask accordingly
    preds_segmentation["waves"] = preds_instances["waves"].astype(bool)

    # Puffs
    preds_instances["puffs"] = remove_small_events(
        class_instances=preds_instances["puffs"],
        class_type="puffs",
        min_t=puff_min_t,
        max_gap=max_gap,
    )

    # update segmented predicted mask accordingly
    preds_segmentation["puffs"] = preds_instances["puffs"].astype(bool)

    # Sparks
    preds_instances["sparks"] = remove_small_events(
        class_instances=preds_instances["sparks"],
        class_type="sparks",
        min_t=spark_min_t,
        min_width=spark_min_width,
    )

    # update segmented predicted mask accordingly
    preds_segmentation["sparks"] = preds_instances["sparks"].astype(bool)

    # remove spark peak locations of sparks that have been removed
    corrected_loc = []
    for t, y, x in sparks_loc:
        if preds_instances["sparks"][t, y, x] != 0:
            corrected_loc.append([t, y, x])
    sparks_loc = corrected_loc

    # Renumber preds, so that each event has a unique ID
    shift_id = 0
    for event_type in ["sparks", "puffs", "waves"]:
        preds_instances[event_type] = renumber_labelled_mask(
            labelled_mask=preds_instances[event_type], shift_id=shift_id
        )
        shift_id = max(shift_id, np.max(preds_instances[event_type]))

    logger.debug(
        f"Time for removing small events: {time.time() - start:.2f} s")

    return preds_instances, preds_segmentation, sparks_loc


def preds_dict_to_mask(preds_dict):
    """
    Convert a dict of binary masks representing a class of calcium release events
    to a single mask with values in {0,1,2,3,4}.

    preds_dict: dict with binary masks for each class sparks, puffs, waves, ignore
    """
    preds_mask = np.zeros_like(list(preds_dict.values())[0], dtype=int)
    for event_type in preds_dict.keys():
        preds_mask = np.where(
            preds_dict[event_type], class_to_nb(event_type), preds_mask)
    return preds_mask


########################### Sparks' masks processing ###########################


def detect_single_roi_peak(movie, roi_mask, max_filter_size=10):
    """
    Given a movie and a ROI, extract peak coords of the movie inside the ROI.

    movie:      input movie (could be smoothed)
    roi_mask:   binary mask with same shape as movie (containing one CC)

    return:     coordinates t,y,x of movie's maximum inside ROI
    """
    roi_movie = np.where(roi_mask, movie, 0.0)

    # compute max along t, for slices in [start, end]
    t_max = roi_movie.max(axis=0)
    # compute std along t, for slices in [start, end]
    t_std = np.std(roi_movie, axis=0)
    # multiply max and std
    prod = t_max * t_std

    # maximum filter
    dilated = ndi.maximum_filter(prod, size=max_filter_size)
    # find locations (y,x) of peaks
    argmaxima = np.logical_and(prod == dilated, prod != 0)
    argwhere = np.argwhere(argmaxima)
    assert len(
        argwhere == 1), f"found more than one spark peak in ROI: {argwhere}"

    # find slice t corresponding to max location
    y, x = argwhere[0]
    t = np.argmax(roi_movie[:, y, x])

    return t, y, x


def detect_spark_peaks(
    movie,
    spark_mask,
    sigma=2,
    max_filter_size=10,
    return_mask=False,
    # annotations=False
):
    """
    Extract local maxima from input array (t,x,y) st each ROI contains one peak.
    movie :           input array
    spark_mask :      mask with spark ROIs
    sigma :           sigma parameter of gaussian filter
    max_filter_size:  dimension of maximum filter size
    return_mask :     if True return both masks with maxima and locations, if
                      False only returns locations
    annotations:      if true, apply specific processing for raw annotation masks
    """
    # get list of sparks IDs
    event_list = list(np.unique(spark_mask))
    event_list.remove(0)

    if sigma > 0:
        # smooth movie only on (y,x)
        smooth_movie = ndi.gaussian_filter(movie, sigma=(0, sigma, sigma))
    else:
        smooth_movie = movie

    peaks_coords = []
    # find one peak in each ROI
    for id_roi in event_list:
        t, y, x = detect_single_roi_peak(
            movie=smooth_movie,
            roi_mask=(spark_mask == id_roi),
            max_filter_size=max_filter_size,
        )

        peaks_coords.append([t, y, x])

    if return_mask:
        peaks_mask = np.zeros_like(spark_mask)
        peaks_mask[np.array(peaks_coords)] = 1
        return peaks_coords, peaks_mask

    return peaks_coords


def get_sparks_locations_from_mask(
    mask, min_dist_xy=MIN_DIST_XY, min_dist_t=MIN_DIST_T, ignore_frames=0
):
    """
    Get sparks coords from annotations mask with spark peaks.

    mask : annotations mask (values 0,1,2,3,4) where sparks are denoted by PEAKS
    ignore_frames: number of frames ignored by loss fct during training
    """

    sparks_mask = np.where(mask == 1, 1.0, 0.0)

    # compute sparks connectivity mask
    connectivity_mask = sparks_connectivity_mask(min_dist_xy, min_dist_t)

    # sparks_mask = empty_marginal_frames(sparks_mask, ignore_frames)
    coords = simple_nonmaxima_suppression(
        img=sparks_mask, min_dist=connectivity_mask, threshold=0, sigma=0.5
    )

    # remove first and last frames
    if ignore_frames > 0:
        mask_duration = mask.shape[0]
        coords = empty_marginal_frames_from_coords(
            coords=coords, n_frames=ignore_frames, duration=mask_duration
        )
    return coords


def process_spark_prediction(
    pred,
    movie=None,
    t_detection=0.9,
    min_dist_xy=MIN_DIST_XY,
    min_dist_t=MIN_DIST_T,
    min_radius=3,
    return_mask=False,
    return_clean_pred=False,
    ignore_frames=0,
    sigma=2,
):
    """
    Get sparks centres from preds: remove small events + nonmaxima suppression

    pred: network's sparks predictions
    movie: original sample movie
    t_detection: sparks detection threshold
    min_dist_xy : minimal spatial distance between two maxima
    min_dist_t : minimal temporal distance between two maxima
    min_radius: minimal 'radius' of a valid spark
    return_mask: if True return mask and locations of sparks
    return_clean_pred: if True only return preds without small events
    ignore_frames: set preds in region ignored by loss fct to 0
    sigma: sigma value used in gaussian smoothing in nonmaxima suppression
    """
    # get binary preds
    pred_boolean = pred > t_detection

    # remove small objects and get clean binary preds
    if min_radius > 0:
        min_size = (2 * min_radius) ** pred.ndim
        small_objs_removed = morphology.remove_small_objects(
            pred_boolean, min_size=min_size
        )
    else:
        small_objs_removed = pred_boolean

    # remove first and last object from sparks mask
    # small_objs_removed = empty_marginal_frames(small_objs_removed,
    #                                           ignore_frames)

    # imageio.volwrite("TEST_small_objs_removed.tif", np.uint8(small_objs_removed))
    # imageio.volwrite("TEST_clean_preds.tif", np.where(small_objs_removed, pred, 0))
    if return_clean_pred:
        # original movie without small objects:
        big_pred = np.where(small_objs_removed, pred, 0)
        return big_pred

    assert movie is not None, "Provide original movie to detect spark peaks"

    # detect events (nonmaxima suppression)
    conn_mask = sparks_connectivity_mask(min_dist_xy, min_dist_t)
    argwhere, argmaxima = simple_nonmaxima_suppression(
        img=movie,
        maxima_mask=small_objs_removed,
        min_dist=conn_mask,
        return_mask=True,
        threshold=0,
        sigma=sigma,
    )

    # remove first and last frames
    if ignore_frames > 0:
        mask_duration = pred.shape[0]
        argwhere = empty_marginal_frames_from_coords(
            coords=argwhere, n_frames=ignore_frames, duration=mask_duration
        )

    if not return_mask:
        return argwhere

    # set frames ignored by loss fct to 0
    argmaxima = empty_marginal_frames(argmaxima, ignore_frames)

    return argwhere, argmaxima


def simple_nonmaxima_suppression(
    img, maxima_mask=None, min_dist=None, return_mask=False,
    threshold=0.5, sigma=2
):
    """
    Extract local maxima from input array (t,x,y).
    img :           input array
    maxima_mask :   if not None, look for local maxima only inside the mask
    min_dist :      define minimal distance between two peaks
    return_mask :   if True return both masks with maxima and locations, if
                    False only returns locations
    threshold :     minimal value of maximum points
    sigma :         sigma parameter of gaussian filter
    """
    img = img.astype(np.float64)

    # handle min_dist connectivity mask
    if min_dist is None:
        min_dist = 1

    if np.isscalar(min_dist):
        c_min_dist = ndi.generate_binary_structure(img.ndim, min_dist)
    else:
        c_min_dist = np.array(min_dist, bool)
        if c_min_dist.ndim != img.ndim:
            raise ValueError("Connectivity dimension must be same as image")

    if sigma > 0:
        if maxima_mask is not None:
            # keep only region inside mask
            masked_img = np.where(maxima_mask, img, 0.0)

            # smooth masked input image
            smooth_img = ndi.gaussian_filter(masked_img, sigma=sigma)
        else:
            smooth_img = ndi.gaussian_filter(img, sigma=sigma)
    else:
        smooth_img = img

    if maxima_mask is not None:
        # hypothesis: maxima belong to maxima mask
        smooth_img = np.where(maxima_mask, smooth_img, 0.0)

    # search for local maxima
    dilated = ndi.maximum_filter(smooth_img, footprint=min_dist)
    argmaxima = np.logical_and(smooth_img == dilated, smooth_img > threshold)

    argwhere = np.argwhere(argmaxima)
    argwhere = np.argwhere(argmaxima).tolist()
    # argwhere = np.array(argwhere, dtype=float)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


############################### Unused functions ###############################


def compute_filtered_butter(movie_array,
                            min_prominence=2,
                            band_stop_width=2,
                            min_freq=7,
                            filter_order=4,
                            Fs=150,
                            debug=False):
    '''
    Apply Butterworth filter to input movie.

    movie_array: input movie to be filtered
    min_prominence: minimal prominence of filtered peaks in frequency domain
    band_stop_width: width of the filtered band for each peak
    min_freq: minimal frequence that can be filtered (???)
    filter_order: order of Butterworth filter
    Fs: sampling frequency [Hz]

    output: filtered version of input movie
    '''

    # sampling period [s]
    T = 1/Fs
    # signal's length [s]
    L = movie_array.shape[0]
    # time vector
    t = np.arange(L) / Fs

    # movie's signal average along time (time profile of image series)
    movie_average = np.mean(movie_array, axis=(1, 2))

    # get noise frequencies
    # compute Fourier transform
    fft = fftpack.fft(movie_average)
    # compute two-sided spectrum
    P2 = np.abs(fft/L)
    # compute single-sided spectrum
    P1 = P2[:(L//2)]
    P1[1:-1] = 2*P1[1:-1]

    freqs = fftpack.fftfreq(L) * Fs
    f = freqs[:L//2]

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
    bands_low = prominent_peaks-band_stop_width
    bands_high = prominent_peaks+band_stop_width
    bands_indices = np.transpose([bands_low, bands_high])

    bands_freq = f[bands_indices]

    # make sure that nothing is outside interval (0,max(f))
    if bands_freq.size > 0:
        bands_freq[:, 0][bands_freq[:, 0] < 0] = 0
        bands_freq[:, 1][bands_freq[:, 1] > max(f)] = max(
            f) - np.mean(np.diff(f))/1000

    # create butterworth filter
    filter_type = 'bandstop'
    filtered = np.copy(movie_array)

    for i, band in enumerate(bands_freq):
        Wn = band / max(f)

        sos = signal.butter(N=filter_order,
                            Wn=Wn,
                            btype=filter_type,
                            output='sos')

        filtered = signal.sosfiltfilt(sos, filtered, axis=0)

    if debug:
        # filtered movie's signal average along time (time profile of image series)
        filtered_movie_average = np.mean(filtered, axis=(1, 2))

        # get frequencies of filtered movie
        # compute Fourier transform
        filtered_fft = fftpack.fft(filtered_movie_average)
        # compute two-sided spectrum
        filtered_P2 = np.abs(filtered_fft/L)
        # compute single-sided spectrum
        filtered_P1 = filtered_P2[:(L//2)]
        filtered_P1[1:-1] = 2*filtered_P1[1:-1]

        # detrend single-sided spectrum
        # filtered_P1_detrend = signal.detrend(filtered_P1) # WRONG??

        return filtered, movie_average, filtered_movie_average, Fs, f, P1, filtered_P1

    return filtered


# OLD VERSION: keeping this only to remember used techniques
# def nonmaxima_suppression(img,maxima_mask=None,
#                          min_dist_xy=MIN_DIST_XY, min_dist_t=MIN_DIST_T,
#                          return_mask=False, threshold=0.5, sigma=2,
#                          annotations=False):
#    '''
#    Extract local maxima from input array (t,x,y).
#    img :           input array
#    maxima_mask :   if not None, look for local maxima only inside the mask
#    min_dist_xy :   minimal spatial distance between two maxima
#    min_dist_t :    minimal temporal distance between two maxima
#    return_mask :   if True return both masks with maxima and locations, if
#                    False only returns locations
#    threshold :     minimal value of maximum points
#    sigma :         sigma parameter of gaussian filter
#    annotations:    if true, apply specific processing for raw annotation masks
#    '''
#    img = img.astype(np.float64)
#
#    # compute shape for maximum filter -> min distance between peaks
#    #min_dist = ellipsoid(min_dist_t/2, min_dist_xy/2, min_dist_xy/2)
#    radius = math.ceil(min_dist_xy/2)
#    y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
#    disk = x**2+y**2 <= radius**2
#    min_dist = np.stack([disk]*(min_dist_t+1), axis=0)
#
#    if maxima_mask is not None:
#        # apply butterworth filter along t-axis
#        filtered_img = compute_filtered_butter(img) # apply butterworth filter
#
#        # apply dilation to maxima mask
#        #min_dist_eroded = ndi.binary_erosion(min_dist)
#        #maxima_mask_dilated = ndi.binary_dilation(maxima_mask, structure=min_dist_eroded)
#        #maxima_mask_dilated = ndi.binary_dilation(maxima_mask, iterations=round(sigma))
#        maxima_mask_dilated = maxima_mask
#
#        # mask out region from img with dilated mask
#        masked_img = np.where(maxima_mask_dilated, filtered_img, 0.)
#        #imageio.volwrite("TEST_masked_video.tif", masked_img)
#
#        # smooth masked input image
#        smooth_img = ndi.gaussian_filter(masked_img, sigma=sigma)
#        #imageio.volwrite("TEST_smooth_video.tif", smooth_img)
#
#    else:
#        smooth_img = ndi.gaussian_filter(img, sigma=sigma)
#
#    # search for local maxima
#
#    dilated = ndi.maximum_filter(smooth_img,
#                                 footprint=min_dist)
#    #imageio.volwrite("TEST_dilated.tif", dilated)
#
#    if maxima_mask is not None:
#        # hyp: maxima belong to maxima mask
#        masked_smooth_img = np.where(maxima_mask, smooth_img, 0.)
#        argmaxima = np.logical_and(smooth_img == dilated, masked_smooth_img > threshold)
#    else:
#        argmaxima = np.logical_and(smooth_img == dilated, smooth_img > threshold)
#
#
#    #imageio.volwrite("TEST_maxima.tif", np.uint8(argmaxima))
#
#    # save movie containing ALL local maxima
#    #dilated_all = ndi.maximum_filter(original_smoothed, footprint=min_dist)
#    #imageio.volwrite("TEST_all_maxima.tif", np.uint8(original_smoothed == dilated_all))
#    #imageio.volwrite("TEST_all_video_maxima.tif", np.uint8(np.logical_and(smooth_img == dilated, smooth_img > threshold)))
#
#    '''# multiply values of video inside maxima mask
#    #img = np.where(maxima_mask, img*1.5, img)
#    imageio.volwrite("TEST_DEBUG.tif", img)
#
#    smooth_img = ndi.gaussian_filter(img, sigma=sigma)
#    imageio.volwrite("TEST_smooth_video.tif", smooth_img)
#
#    if maxima_mask is not None:
#        # apply dilation to mask
#        #maxima_mask_dilated = ndi.binary_dilation(maxima_mask, iterations=round(sigma))
#        maxima_mask_dilated = maxima_mask
#        # set pixels outside maxima_mask to zero
#        masked_img = np.where(maxima_mask_dilated, smooth_img, 0.)
#        #masked_img = np.where(maxima_mask, smooth_img, 0.)
#        imageio.volwrite("TEST_masked_video.tif", masked_img)
#    else:
#        masked_img = smooth_img
#
#
#    # compute shape for maximum filter
#    #min_dist = ellipsoid(min_dist_t/2, min_dist_xy/2, min_dist_xy/2)
#    radius = round(min_dist_xy/2)
#    y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
#    disk = x**2+y**2 <= radius**2
#    min_dist = np.stack([disk]*min_dist_t, axis=0)
#
#    # detect local maxima
#    dilated = ndi.maximum_filter(smooth_img,
#                                 footprint=min_dist)
#    imageio.volwrite("TEST_dilated.tif", dilated)
#    argmaxima = np.logical_and(smooth_img == dilated, masked_img > threshold)
#    imageio.volwrite("TEST_maxima.tif", np.uint8(argmaxima))
#    #imageio.volwrite("TEST_all_video_maxima.tif", np.uint8(np.logical_and(smooth_img == dilated, smooth_img > threshold)))'''
#
#    argwhere = np.argwhere(argmaxima)
#
#    # DEBUG: compute minimal distance between pair of sparks
#
#    argwhere = np.array(argwhere, dtype=np.float)
#    '''if argwhere.size > 0:
#        argwhere[:,0] /= min_dist_t
#        argwhere[:,1] /= min_dist_xy
#        argwhere[:,2] /= min_dist_xy
#
#        w = spatial.distance_matrix(argwhere, argwhere)
#        w = np.tril(w)
#        w[w==0.0] = 9999999
#        min_w = np.min(w)
#        min_coords = np.argwhere(w==min_w)
#
#        argwhere[:,0] *= min_dist_t
#        argwhere[:,1] *= min_dist_xy
#        argwhere[:,2] *= min_dist_xy
#
#        close_coords = argwhere[min_coords][0]
#        print(f"Closest coordinates: \n{close_coords}")'''
#
#    if not return_mask:
#        return argwhere
#
#    return argwhere, argmaxima


# def separate_events(pred, t_detection=0.5, min_radius=4):
#    '''
#    Apply threshold to prediction and separate the events (1 event = 1 connected
#    component).
#    '''
#    # apply threshold to prediction
#    pred_boolean = pred >= t_detection
#
#    # clean events
#    min_size = (2 * min_radius) ** pred.ndim
#    pred_clean = morphology.remove_small_objects(pred_boolean,
#                                                 min_size=min_size)
#    #big_pred = np.where(small_objs_removed, pred, 0)
#
#    # separate events
#    connectivity = 26
#    labels, n_events = cc3d.connected_components(pred_clean,
#                                                 connectivity=connectivity,
#                                                 return_N=True)
#
#    return labels, n_events


# def process_puff_prediction(pred, t_detection = 0.5,
#                            min_radius = 4,
#                            ignore_frames = 0,
#                            convex_hull = False):
#    '''
#    Get binary clean predictions of puffs (remove small preds)
#
#    pred :          network's puffs predictions
#    min_radius :    minimal 'radius' of a valid puff
#    ignore_frames : set preds in region ignored by loss fct to 0
#    convex_hull :   if true remove holes inside puffs
#    '''
#    # get binary predictions
#    pred_boolean = pred > t_detection
#
#    if convex_hull:
#        # remove holes inside puffs (compute convex hull)
#        pred_boolean = binary_dilation(pred_boolean, iterations=5)
#        pred_boolean = binary_erosion(pred_boolean, iterations=5, border_value=1)
#
#    min_size = (2 * min_radius) ** pred.ndim
#    small_objs_removed = morphology.remove_small_objects(pred_boolean,
#                                                         min_size=min_size)
#
#    # set first and last frames to 0 according to ignore_frames
#    if ignore_frames != 0:
#        pred_puffs = empty_marginal_frames(small_objs_removed, ignore_frames)
#
#    return pred_puffs


# def process_wave_prediction(pred, t_detection = 0.5,
#                            min_radius = 4,
#                            ignore_frames = 0):
#
#    # for now: do the same as with puffs
#
#    return process_puff_prediction(pred, t_detection, min_radius, ignore_frames)

# def filter_nan_gaussian_david(arr, sigma):
#    # https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
#    """Allows intensity to leak into the nan area.
#    According to Davids answer:
#        https://stackoverflow.com/a/36307291/7128154
#    """
#    gauss = arr.copy()
#    gauss[np.isnan(gauss)] = 0
#    gauss = ndi.gaussian_filter(
#            gauss, sigma=sigma, mode='constant', cval=0)
#
#    norm = np.ones(shape=arr.shape)
#    norm[np.isnan(arr)] = 0
#    norm = ndi.gaussian_filter(
#            norm, sigma=sigma, mode='constant', cval=0)
#
#    # avoid RuntimeWarning: invalid value encountered in true_divide
#    norm = np.where(norm==0, 1, norm)
#    gauss = gauss/norm
#    gauss[np.isnan(arr)] = np.nan
#    return gauss
