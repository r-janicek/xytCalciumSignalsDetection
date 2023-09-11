"""
20.10.2022

Script with functions to either load data or save data to disc.
"""

import csv
import datetime
import glob
import logging
import os

import imageio
import numpy as np
from data_processing_tools import process_spark_prediction
from metrics_tools import correspondences_precision_recall
from PIL import Image
from visualization_tools import (
    add_colored_classes_to_video,
    add_colored_instances_to_video,
    add_colored_paired_sparks_to_video,
)

# REMARK
# Come salvare files .json
# with open(filename,"w") as f:
#    json.dump(dict_or_data,f)

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


################################ Loading utils #################################


def load_movies_ids(data_folder, ids, names_available=False, movie_names=None):
    """
    Load  movies corresponding to a given list of indices.

    data_folder:    folder where movies are saved, movies are saved as
                    "[0-9][0-9]*.tif"
    ids :           list of movies IDs (of the form "[0-9][0-9]")
    names_available: if True, can specify name of the movie file, such as
                    "XX_<movie_name>.tif"
    movie_names:     movie name, if available
    """
    xs_all_trainings = {}

    if names_available:
        xs_filenames = [
            os.path.join(data_folder, idx + "_" + movie_names + ".tif") for idx in ids
        ]
    else:
        xs_filenames = [
            os.path.join(data_folder, movie_name)
            for movie_name in os.listdir(data_folder)
            if movie_name.startswith(tuple(ids))
        ]

    for f in xs_filenames:
        video_id = os.path.split(f)[1][:2]
        xs_all_trainings[video_id] = np.asarray(imageio.volread(f))

    return xs_all_trainings


def load_annotations_ids(data_folder, ids, mask_names="video_mask"):
    """
    Same as load_annotations but must provide a list of ids of movies' masks to
    load.

    data_folder: folder where annotations are saved, annotations are saved as
                 "[0-9][0-9]_video_mask.tif"
    ids:         list of ids of movies to be considered
    mask_names:  name of the type of masks that will be loaded
    """
    ys_all_trainings = {}

    ys_filenames = [
        os.path.join(data_folder, idx + "_" + mask_names + ".tif") for idx in ids
    ]

    for f in ys_filenames:
        video_id = os.path.split(f)[1][:2]
        ys_all_trainings[video_id] = np.asarray(
            imageio.volread(f)).astype("int")

    return ys_all_trainings


def load_rgb_annotations_ids(data_folder, ids, mask_names="separated_events"):
    """
    Same as load_annotations_ids but load original rbg annotations with
    separated events.

    data_folder: folder where annotations are saved, annotations are saved as
                 "[0-9][0-9]_separated_events.tif"
    ids:         list of ids of movies to be considered
    mask_names:  name of the type of masks that will be loaded
    """

    ys_all_trainings = {}

    ys_filenames = [
        os.path.join(data_folder, idx + "_" + mask_names + ".tif") for idx in ids
    ]

    # integer representing white colour in rgb mask
    white_int = 255 * 255 * 255 + 255 * 255 + 255

    for f in ys_filenames:
        video_id = os.path.split(f)[1][:2]
        rgb_video = np.asarray(imageio.volread(f)).astype("int")

        mask_video = (
            255 * 255 * rgb_video[..., 0] + 255 *
            rgb_video[..., 1] + rgb_video[..., 2]
        )

        mask_video[mask_video == white_int] = 0

        ys_all_trainings[video_id] = mask_video

    return ys_all_trainings


def load_predictions_ids(training_name, epoch, metrics_folder, ids):
    """
    open and process annotations (where sparks have been processed), predicted
    sparks, puffs and waves for a given training name
    !!! the predictions movies have to be saved in metrics_folder for the given
        training name !!!

    training_name: saved training name
    epoch: training epoch whose predictions have to be loaded
    metrics_folder: folder where predictions and annotations are saved,
                    annotations are saved as "[0-9]*_ys.tif"
                    sparks are saved as "<base name>_[0-9][0-9]_sparks.tif"
                    puffs are saved as "<base name>_[0-9][0-9]_puffs.tif"
                    waves are saved as "<base name>_[0-9][0-9]_waves.tif"
    """

    # Import .tif files as numpy array
    base_name = os.path.join(
        metrics_folder, training_name + "_" + str(epoch) + "_")

    if "temporal_reduction" in training_name:
        # need to use annotations from another training
        # TODO: implement a solution ....
        logger.warning(
            """!!! method is using temporal reduction, processed annotations
                     have a different shape !!!"""
        )

    # get predictions and annotations filenames
    ys_filenames = sorted(
        [base_name + sample_id + "_ys.tif" for sample_id in ids])
    sparks_filenames = sorted(
        [base_name + sample_id + "_sparks.tif" for sample_id in ids]
    )
    puffs_filenames = sorted(
        [base_name + sample_id + "_puffs.tif" for sample_id in ids]
    )
    waves_filenames = sorted(
        [base_name + sample_id + "_waves.tif" for sample_id in ids]
    )

    # create dictionaires to store loaded data for each movie
    training_ys = {}
    training_sparks = {}
    training_puffs = {}
    training_waves = {}

    for y, s, p, w in zip(
        ys_filenames, sparks_filenames, puffs_filenames, waves_filenames
    ):

        # get movie name
        video_id = y.replace(base_name, "")[:2]

        ys_loaded = np.asarray(imageio.volread(y)).astype("int")
        training_ys[video_id] = ys_loaded

        if "temporal_reduction" in training_name:
            # repeat each frame 4 times
            logger.info(
                "training using temporal reduction, extending predictions...")
            s_preds = np.asarray(imageio.volread(s))
            p_preds = np.asarray(imageio.volread(p))
            w_preds = np.asarray(imageio.volread(w))

            # repeat predicted frames x4
            s_preds = np.repeat(s_preds, 4, 0)
            p_preds = np.repeat(p_preds, 4, 0)
            w_preds = np.repeat(w_preds, 4, 0)

            # TODO: can't crop until annotations loading is fixed
            # if original length %4 != 0, crop preds
            # if ys_loaded.shape != s_preds.shape:
            #    duration = ys_loaded.shape[0]
            #    s_preds = s_preds[:duration]
            #    p_preds = p_preds[:duration]
            #    w_preds = w_preds[:duration]

            # TODO: can't check until annotations loading is fixed
            # assert ys_loaded.shape == s_preds.shape
            # assert ys_loaded.shape == p_preds.shape
            # assert ys_loaded.shape == w_preds.shape

            training_sparks[video_id] = s_preds
            training_puffs[video_id] = p_preds
            training_waves[video_id] = w_preds
        else:
            training_sparks[video_id] = np.asarray(imageio.volread(s))
            training_puffs[video_id] = np.asarray(imageio.volread(p))
            training_waves[video_id] = np.asarray(imageio.volread(w))

    return training_ys, training_sparks, training_puffs, training_waves


def load_predictions_all_trainings_ids(training_names, epochs, metrics_folder, ids):
    """
    open and process annotations (where sparks have been processed), predicted
    sparks, puffs and waves for a list of training names
    !!! the predictions movies have to be saved in metrics_folder for the given
        training name !!!

    training_names: list of saved training names
    epochs: list of training epochs whose predictions have to be loaded
            corresponding to the training names
    metrics_folder: folder where predictions and annotations are saved,
                    annotations are saved as "[0-9][0-9]_ys.tif"
                    sparks are saved as "<base name>_[0-9][0-9]_sparks.tif"
                    puffs are saved as "<base name>_[0-9][0-9]_puffs.tif"
                    waves are saved as "<base name>_[0-9][0-9]_waves.tif"
    """
    # dicts with "shapes":
    # num trainings (dict) x num videos (dict) x video shape
    ys = {}
    s = {}  # sparks
    p = {}  # puffs
    w = {}  # waves

    for name, epoch in zip(training_names, epochs):
        ys[name], s[name], p[name], w[name] = load_predictions_ids(
            name, epoch, metrics_folder, ids
        )

    return ys, s, p, w


####################### Tools for writing videos on disc #######################


def write_videos_on_disk(
    training_name, video_name, path="predictions", xs=None, ys=None, preds=None
):
    """
    Write all videos on disk
    xs : input video used by network
    ys: segmentation video used in loss function
    preds : all u-net preds [bg preds, sparks preds, puffs preds, waves preds]
            preds should already be normalized between 0 and 1
    """
    if out_name_root is not None:
        out_name_root = training_name + "_" + video_name + "_"
    else:
        out_name_root = video_name + "_"

    logger.debug(f"Writing videos on directory {os.path.abspath(path)} ..")
    os.makedirs(os.path.abspath(path), exist_ok=True)

    if not isinstance(xs, type(None)):
        imageio.volwrite(os.path.join(
            path, out_name_root + "xs.tif"), xs)
    if not isinstance(ys, type(None)):
        imageio.volwrite(os.path.join(
            path, out_name_root + "ys.tif"), np.uint8(ys))
    if not isinstance(preds, type(None)):
        imageio.volwrite(os.path.join(
            path, out_name_root + "sparks.tif"), preds[1])
        imageio.volwrite(os.path.join(
            path, out_name_root + "waves.tif"), preds[2])
        imageio.volwrite(os.path.join(
            path, out_name_root + "puffs.tif"), preds[3])


def write_colored_events_videos_on_disk(
    movie,
    events_mask,
    out_dir,
    movie_fn,
    transparency=50,
    ignore_frames=0,
    white_bg=False,
    instances=False,
    label_contours=False,
    label_mask=None
):
    r"""
    Given a movie and a class segmentation (labels or preds), paste colored
    segmentation on video and save on disk.

    The file will be saved at "out_dir/movie_fn.tif"

    Set white_bg == True to save colored segmentation on white background.

    If instances == True, events_maks is considered as a mask of instances
    otherwise it is considered as a mask of classes.

    If label_contours == True, the contours of the labels will be drawn on the
    original movie, togheter with the colored predictions.

    Color used:
    sparks: green
    puffs: red
    waves: purple
    ignore regions: grey
    """
    if instances:
        colored_movie = add_colored_instances_to_video(movie=movie,
                                                       instances_mask=events_mask,
                                                       transparency=transparency,
                                                       ignore_frames=ignore_frames,
                                                       white_bg=white_bg)
    else:
        if label_contours:
            assert label_mask is not None, \
                "label_mask must be provided when label_contours is True"

        colored_movie = add_colored_classes_to_video(movie=movie,
                                                     classes_mask=events_mask,
                                                     transparency=transparency,
                                                     ignore_frames=ignore_frames,
                                                     white_bg=white_bg,
                                                     label_mask=label_mask)

    imageio.volwrite(os.path.join(out_dir, movie_fn + ".tif"), colored_movie)


############################### csv files tools ################################


def create_csv(filename, positions):
    # N = positions.shape[0]
    with open(filename, "w") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        filewriter.writerow(["frame", "x", "y"])
        # for n in range(N):
        #    filewriter.writerow([positions[n,0], positions[n,1], positions[n,2]])
        for loc in positions:
            filewriter.writerow([loc[0], loc[2], loc[1]])
            logger.info(
                f"Location {[loc[0], loc[2], loc[1]]} written to .csv file")


################## tools for writing sparks locations on disk ##################


def write_spark_locations_on_disk(
    spark_pred, movie, filename, t_detection_sparks, min_r_sparks, ignore_frames=0
):
    # process predictions
    sparks_list = process_spark_prediction(
        pred=spark_pred,
        movie=movie,
        ignore_frames=ignore_frames,
        t_detection=t_detection_sparks,
        min_radius=min_r_sparks,
    )

    logger.info(f"Writing sparks locations to .csv file in file {filename}")
    create_csv(filename, sparks_list)


def write_paired_sparks_on_disk(paired_true, paired_preds, fp, fn, file_path):
    # after computing spark peaks correspondences, write the on a .txt file
    # paired_true: list of coords of annotated sparks paired with a pred
    # paired_preds: list of coords of predicted sparks paired with an annotation
    # fp: list of coords of false positive predicted sparks
    # fn: list of coords of false negative annotated sparks
    # file_path: location + filename of output text file

    with open(file_path, "w") as f:
        f.write(f"{datetime.datetime.now()}\n\n")
        f.write(f"Paired annotations and preds:\n")
        for p_true, p_preds in zip(paired_true, paired_preds):
            f.write(f"{list(map(int, p_true))} {list(map(int, p_preds))}\n")
        f.write(f"\n")
        f.write(f"Unpaired preds (false positives):\n")
        for f_p in fp:
            f.write(f"{list(map(int, f_p))}\n")
        f.write(f"\n")
        f.write(f"Unpaired annotations (false negatives):\n")
        for f_n in fn:
            f.write(f"{list(map(int, f_n))}\n")


def write_colored_sparks_on_disk(
    training_name,
    video_name,
    paired_real,
    paired_pred,
    false_positives,
    false_negatives,
    path="predictions",
    xs=None,
    movie_shape=None,
):
    """
    Write input video with colored paired sparks and text file with sparks
    coordinates on disk.

    Used at model validation (test_function) during training.

    training_name, video_name : used to save output on disk
    paired_real : list of coordinates [t,y,x] of paired annotated sparks
    paired_pred : list of coordinates [t,y,x] of paired predicted sparks
    false_positives : list of coordinates [t,y,x] of wrongly predicted sparks
    false_negatives : list of coordinates [t,y,x] of not found annotated sparks
    path : directory where output will be saved
    xs: input video used by network, if None, save sparks on white background
    movie_shape : if input movie xs is None, provide video shape (t,y,x)
    """

    if not isinstance(xs, type(None)):
        sample_video = xs
    else:
        assert not isinstance(
            movie_shape, type(None)
        ), "Provide movie shape if not providing input movie."
        sample_video = 255 * np.ones(movie_shape)

    # compute colored sparks mask
    transparency = 45
    annotated_video = add_colored_paired_sparks_to_video(
        movie=sample_video,
        paired_true=paired_real,
        paired_preds=paired_pred,
        fp=false_positives,
        fn=false_negatives,
        transparency=transparency,
    )

    # set saved movies filenames
    white_background_fn = "white_BG" if isinstance(xs, type(None)) else ""
    out_name_root = f"{training_name }_{video_name}_{white_background_fn}"

    # save video on disk
    imageio.volwrite(
        os.path.join(
            path, f"{out_name_root}_colored_sparks.tif"), annotated_video
    )

    # write sparks locations to file
    file_path = os.path.join(path, f"{out_name_root}_sparks_location.txt")
    write_paired_sparks_on_disk(
        paired_true=paired_real,
        paired_preds=paired_pred,
        fp=false_positives,
        fn=false_negatives,
        file_path=file_path,
    )


def pair_and_write_sparks_on_video(
    movie,
    coords_true,
    coords_preds,
    min_dist_xy,
    min_dist_t,
    out_path,
    white_background,
    transparency=50,
    training_name=None,
    epoch=None,
    movie_id=None,
):
    # compute annotated and predicted spark peaks correspondences and then save
    # resulting paired coords, false negative, false positive on text file
    # + save colored spark peaks on movie on disk
    # + save text file with summarized parameters
    # training_name, movie_id and epoch are only used for movie's output filename

    # Compute correspondences between annotations and predictions
    paired_true, paired_preds, fp, fn = correspondences_precision_recall(
        coords_true, coords_preds, min_dist_t, min_dist_xy, return_pairs_coords=True
    )
    # Add colored annotations to video
    sample_video = np.copy(movie)
    if white_background:
        sample_video.fill(255)  # the movie will be white

    annotated_video = add_colored_paired_sparks_to_video(
        movie=sample_video,
        paired_true=paired_true,
        paired_preds=paired_preds,
        fp=fp,
        fn=fn,
        transparency=transparency,
    )

    # Write sparks locations to file
    coords_file_path = os.path.join(
        out_path, f"{movie_id}_sparks_location.txt")
    write_paired_sparks_on_disk(
        paired_true=paired_true,
        paired_preds=paired_preds,
        fp=fp,
        fn=fn,
        file_path=coords_file_path,
    )

    # set saved movies filenames
    wb_fn = "_white_backgroud" if white_background else ""

    movie_fn = "colored_sparks" + wb_fn + ".tif"
    if movie_id is not None:
        movie_fn = movie_id + "_" + movie_fn
    if epoch is not None:
        movie_fn = str(epoch) + "_" + movie_fn
    if training_name is not None:
        movie_fn = training_name + "_" + movie_fn

    movie_path = os.path.join(out_path, movie_fn)
    imageio.volwrite(movie_path, annotated_video)

    # write summary file with parameters
    summary_file_path = os.path.join(out_path, f"parameters{wb_fn}.txt")

    with open(summary_file_path, "w") as f:
        f.write(f"{datetime.datetime.now()}\n\n")

        f.write("Phyisiological parameters\n")
        f.write(f"Pixel size: {PIXEL_SIZE} um\n")
        f.write(f"Min distance (x,y): {min_dist_xy} pixels\n")
        f.write(f"Time frame: {TIME_FRAME} ms\n")
        f.write(f"Min distance t: {min_dist_t} pixels\n\n")

        if training_name is not None:
            f.write("Training parameters\n")
            f.write(f"Training name: {training_name}\n")
            f.write(f"Loaded epoch: {epoch}\n")

        f.write("Coloured sparks parameters\n")
        f.write(f"Saved coloured sparks path: {out_path}\n")
        f.write(f"Coloured sparks' transparency: {transparency}\n")
        f.write(
            f"Using white background instead of original movies: {white_background}\n"
        )
