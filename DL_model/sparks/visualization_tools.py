'''
20.10.2022

Script with tools for data visualisation (e.g. plots and Napari)
'''

import itertools
import logging
import math

import numpy as np
# import vispy.color # not working ?!?
from matplotlib import cm
from PIL import Image
from scipy import ndimage as ndi
import vispy

# REMARK
# Come creare un dataframe con vari parametri
# rows <- lista di liste con un valore per ogni colonna
# cols <- lista con i nomi delle colonne
# df = pd.DataFrame(rows, columns=cols)
# -> si possono 'append' datasets con la stessa struttura

# Altre funzioni utile per visualizzare i dati:
# df_values = df[(df[col_name]==value) & ...]
# pd.pivot_table(df_values, values=...,columns=[col_names], index=[...])

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


#################################### Napari ####################################


def get_discrete_cmap(name='gray', lut=16):
    # function to obtain discrete Colormap instance that can be used by Napari

    # create original cmap
    segmented_cmap = cm.get_cmap(name=name, lut=lut)

    # get colors
    colors = segmented_cmap(np.arange(0, segmented_cmap.N))

    # get new discrete cmap
    cmap = vispy.color.Colormap(colors, interpolation='zero')

    return cmap


def get_labels_cmap():
    no_color = [0, 0, 0, 0]
    green_spark = [178/255, 255/255, 102/255, 1]
    red_puff = [255/255, 102/255, 102/255, 1]
    purple_wave = [178/255, 102/255, 255/255, 1]
    grey_ignore = [224/255, 224/255, 224/255, 1]
    black = [1, 1, 1, 1]

    labels_cmap = {
        0: no_color,
        1: green_spark,
        2: purple_wave,
        3: red_puff,
        4: grey_ignore,
        5: black
    }

    return labels_cmap


def set_edges_to_white(input_movie, white, thickness=0):
    if thickness == 0:
        input_movie[0, 0, :] = white
        input_movie[-1, 0, :] = white
        input_movie[0, -1, :] = white
        input_movie[-1, -1, :] = white

        input_movie[:, 0, 0] = white
        input_movie[:, -1, 0] = white
        input_movie[:, 0, -1] = white
        input_movie[:, -1, -1] = white

        input_movie[0, :, 0] = white
        input_movie[-1, :, 0] = white
        input_movie[0, :, -1] = white
        input_movie[-1, :, -1] = white

    else:
        input_movie[:thickness, :thickness, :] = white
        input_movie[-thickness-1:, :thickness, :] = white
        input_movie[:thickness, -thickness-1:, :] = white
        input_movie[-thickness-1:, -thickness-1:, :] = white

        input_movie[:, :thickness, :thickness] = white
        input_movie[:, -thickness-1:, :thickness] = white
        input_movie[:, :thickness, -thickness-1:] = white
        input_movie[:, -thickness-1:, -thickness-1:] = white

        input_movie[:thickness, :, :thickness] = white
        input_movie[-thickness-1:, :, :thickness] = white
        input_movie[:thickness, :, -thickness-1:] = white
        input_movie[-thickness-1:, :, -thickness-1:] = white


def get_annotations_contour(annotations, contour_val=2):
    '''
    Compute annotation's mask contour, for Napari visualisation.
    '''
    # dilate only along x and y
    struct = np.zeros((1, 1+contour_val, 1+contour_val))
    struct[0, 1, :] = 1
    struct[0, :, 1] = 1

    labels_contour = np.zeros_like(annotations)

    # need to dilate and erode each class separately
    for class_nb in range(1, 5):
        class_ys = annotations == class_nb
        class_dilated = ndi.binary_dilation(class_ys, structure=struct)
        class_eroded = ndi.binary_erosion(class_ys, structure=struct)
        class_contour = np.where(np.logical_not(class_eroded.astype(bool)),
                                 class_dilated,
                                 0
                                 )
        labels_contour += class_nb*class_contour.astype(labels_contour.dtype)

    return labels_contour

#################### colored events on movie visualisation #####################


def paste_segmentation_on_video(video, colored_mask):
    # video is a RGB video, list of PIL images
    # colored_mask is a RGBA video, list of PIL images
    for frame, ann in zip(video, colored_mask):
        frame.paste(ann, mask=ann.split()[3])


def add_colored_segmentation_to_video(segmentation, video, color, transparency=50):
    # segmentation is a binary array
    # video is a RGB video, list of PIL images
    # color is a list of 3 RGB elements

    # convert segmentation into a colored mask
    # mask_shape = (*(segmentation.shape), 4)
    # colored_mask = np.zeros(mask_shape, dtype=np.uint8)
    r, g, b = color
    colored_mask = np.stack((r*segmentation, g*segmentation,
                            b*segmentation, transparency*segmentation), axis=-1)
    colored_mask = colored_mask.astype(np.uint8)

    colored_mask = [Image.fromarray(frame).convert('RGBA')
                    for frame in colored_mask]

    paste_segmentation_on_video(video, colored_mask)
    return video


def add_colored_locations_to_video(locs, video, color, transparency=50, radius=4):
    # locs is a list of t,x,y coordinates
    # video is a RGB video, list of PIL images
    # color is a list of 3 RGB elements
    mask_shape = (len(video), video[0].size[1], video[0].size[0], 4)
    colored_mask = np.zeros(mask_shape, dtype=np.uint8)
    for pt in locs:
        colored_mask = color_ball(
            colored_mask, pt, radius, color, transparency)
    colored_mask = [Image.fromarray(frame).convert('RGBA')
                    for frame in colored_mask]

    paste_segmentation_on_video(video, colored_mask)
    return video


def l2_dist(p1, p2):
    # p1 = (t1,y1,x1)
    # p2 = (t2,y2,x2)
    t1, y1, x1 = p1
    t2, y2, x2 = p2
    return math.sqrt(math.pow((t1-t2), 2)+math.pow((y1-y2), 2)+math.pow((x1-x2), 2))


def ball(c, r):
    # r scalar
    # c = (t,y,x)
    # returns coordinates c' around c st dist(c,c') <= r
    t, y, x = c
    t_vect = np.linspace(t-r, t+r, 2*r+1, dtype=int)
    y_vect = np.linspace(y-r, y+r, 2*r+1, dtype=int)
    x_vect = np.linspace(x-r, x+r, 2*r+1, dtype=int)

    cube_idxs = list(itertools.product(t_vect, y_vect, x_vect))
    ball_idxs = [pt for pt in cube_idxs if l2_dist(c, pt) <= r]

    return ball_idxs


def color_ball(mask, c, r, color, transparency=50):
    color_idx = ball(c, r)
    # mask boundaries
    duration, height, width, _ = np.shape(mask)

    for t, y, x in color_idx:
        if 0 <= t and t < duration and 0 <= y and y < height and 0 <= x and x < width:
            mask[t, y, x] = [*color, transparency]

    return mask


def add_colored_paired_sparks_to_video(movie, paired_true, paired_preds, fp, fn,
                                       transparency=50, white_background=False):
    # return a colored movie where paired sparks are color-coded as follows:
    # paired annotations: green
    # paired preds: blue
    # false positive preds: yellow
    # false nevative annotations: red
    # - each input of locations is a list of coordinates
    # - movie can be original sample or white background

    # normalize sample movie and convert to RGB
    sample_video = np.copy(movie)
    sample_video = 255*(sample_video/sample_video.max())
    rgb_video = [Image.fromarray(frame).convert('RGB')
                 for frame in sample_video]

    # Add colored annotations to video
    annotated_video = add_colored_locations_to_video(paired_true,
                                                     rgb_video,
                                                     [0, 255, 0],
                                                     0.8*transparency)
    annotated_video = add_colored_locations_to_video(paired_preds,
                                                     annotated_video,
                                                     [0, 255, 200],
                                                     0.8*transparency)
    annotated_video = add_colored_locations_to_video(fp,
                                                     annotated_video,
                                                     [255, 255, 0],
                                                     transparency)
    annotated_video = add_colored_locations_to_video(fn,
                                                     annotated_video,
                                                     [255, 0, 0],
                                                     transparency)

    annotated_video = [np.array(frame) for frame in annotated_video]
    return annotated_video


def add_colored_classes_to_video(movie, classes_mask, transparency=50,
                                 ignore_frames=0, white_bg=False,
                                 label_mask=None):
    r'''
    given a movie and a segmentation with all classes (sparks, puffs, waves and
    ignore regions), create a movie with the colored events on top
    sparks: green
    puffs: red
    waves: purple
    ignore regions: grey

    if label_mask is provided, then add labels denoted by their contours in addition
    to the transparent prediction mask
    '''
    # dataset params
    classes = ['sparks', 'puffs', 'waves', 'ignore']
    classes_nb = {'sparks': 1, 'puffs': 3, 'waves': 2, 'ignore': 4}

    # normalize sample movie and create color dict
    if white_bg:
        movie.fill(255)
        color_dict = {'sparks': [0, 255, 0],  # green
                      'puffs': [220, 20, 60],  # red
                      'waves': [138, 43, 226],  # purple
                      'ignore': [80, 80, 80]  # gray
                      }

        # if background is white, remove transparency
        transparency = 1000

    else:
        movie = 255*(movie/movie.max())
        color_dict = {'sparks': [178, 255, 102],  # green
                      'puffs': [255, 102, 102],  # red
                      'waves': [178, 102, 255],  # purple
                      'ignore': [224, 224, 224]  # gray
                      }

    # convert video to RGB
    color_movie = [Image.fromarray(frame).convert('RGB')
                   for frame in movie]

    # add colored segmentation to movie
    for event_class in classes:
        # get values of current class
        class_nb = classes_nb[event_class]

        if class_nb in classes_mask:
            binary_preds = classes_mask == class_nb
            color_movie = add_colored_segmentation_to_video(
                segmentation=binary_preds,
                video=color_movie,
                color=color_dict[event_class],
                transparency=transparency)

        # add labels contours if label_mask is provided
        if label_mask is not None:
            label_contours = get_annotations_contour(annotations=label_mask,
                                                     contour_val=2)
            if class_nb in label_mask:
                binary_labels = label_contours == class_nb
                color_movie = add_colored_segmentation_to_video(
                    segmentation=binary_labels,
                    video=color_movie,
                    color=color_dict[event_class],
                    transparency=1000)

    # convert to numpy array and remove first and last frames
    if ignore_frames > 0:
        color_movie = [np.array(frame)
                       for frame in color_movie[ignore_frames:-ignore_frames]]
    else:
        color_movie = [np.array(frame)for frame in color_movie]

    return color_movie


def add_colored_instances_to_video(movie, instances_mask, transparency=50,
                                   ignore_frames=0, white_bg=False):
    r'''
    given a movie and a segmentation with all event instances, create a movie
    with the colored events on top
    each event is assigned a random color
    '''
    # normalize sample movie
    if white_bg:
        movie.fill(255)
        # if background is white, remove transparency
        transparency = 1000

    else:
        movie = 255*(movie/movie.max())

    # convert video to RGB
    color_movie = [Image.fromarray(frame).convert('RGB')
                   for frame in movie]

    # add colored segmentation to movie
    for event_id in range(1, instances_mask.max()+1):
        event_mask = instances_mask == event_id

        # create random color
        color = np.random.randint(0, 255, size=3)

        color_movie = add_colored_segmentation_to_video(segmentation=event_mask,
                                                        video=color_movie,
                                                        color=color,
                                                        transparency=transparency)

    # convert to numpy array and remove first and last frames
    if ignore_frames > 0:
        color_movie = [np.array(frame)
                       for frame in color_movie[ignore_frames:-ignore_frames]]
    else:
        color_movie = [np.array(frame)for frame in color_movie]

    return color_movie


############################## signal extraction ###############################


def create_circular_mask(h, w, center, radius):
    # h : image height
    # w : image width
    # center : center of the circular mask (x_c, y_c) !!
    # radius : radius of the circular mask
    # returns a circular mask of given radius around given center

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius

    return mask


def create_signal_mask(t, h, w, start, stop, center, radius):
    # t : video duration
    # h : video height
    # w : video width
    # start : first frame
    # stop : last frame (not included!)
    # center : center of the circular mask (x_c,y_c)
    # radius : radius of the circular mask
    start = max(0, start)
    stop = min(t, stop)

    frame_mask = create_circular_mask(h, w, center, radius)

    video_mask = np.zeros((t, h, w), dtype=bool)
    video_mask[start:stop] = frame_mask

    return video_mask


def get_spark_signal(video,
                     sparks_labelled,
                     center,
                     radius,
                     context_duration,
                     return_info=False):
    # video:             is the original video sample
    # sparks_labelled:   is a mask containing the segmentation of the
    #                    spark events (1 integer for every event)
    # center:            [t y x] is the center of the selected event to plot
    # radius:            is the radius of the considered region around the
    #                    center the spark for averaging
    # context_duration:  is the number of frames included in the analysis before
    #                    and after the event

    t, y, x = center
    event_idx = sparks_labelled[t, y, x]  # event_idx = 1,2,...,n_sparks

    assert event_idx != 0, (
        "given center does not correspond to any event in the given labelled mask")

    loc = ndi.measurements.find_objects(sparks_labelled)[event_idx-1]

    assert loc[0].start <= t and loc[0].stop > t, "something weird is wrong"

    # get mask representing sparks location (with radius and context)
    start = loc[0].start - context_duration
    stop = loc[0].stop + context_duration

    start = max(0, start)
    stop = min(video.shape[0], stop)
    signal_mask = create_signal_mask(*sparks_labelled.shape,
                                     start, stop,
                                     (x, y), radius)

    frames = np.arange(start, stop)
    signal = np.average(video[start:stop],
                        axis=(1, 2),
                        weights=signal_mask[start:stop])

    if return_info:
        return frames, signal, (y, x), loc[0].start, loc[0].stop

    return frames, signal


def get_spark_2d_signal(video, slices, coords, spatial_context, sigma=2, return_info=False):
    # video : original video
    # slices : slices in the 3 dimensions of a given spark (ROI)
    # coords [t y x]: center of a given spark
    # spatial_context : extend ROI corresponding to spark
    # sigma : for gaussian filtering

    # TODO: add assertion to check that coords are inside slices

    t, y, x = coords
    t_slice, y_slice, x_slice = slices

    y_start = max(0, y_slice.start-spatial_context)
    y_end = min(video.shape[1], y_slice.stop+spatial_context)

    x_start = max(0, x_slice.start-spatial_context)
    x_end = min(video.shape[2], x_slice.stop+spatial_context)

    signal_2d = video[t, y_start:y_end, x_start:x_end]

    # average over 3 frames
    # signal_2d_avg = video_array[t-1:t+1, y_start:y_end, x_start:x_end]
    # signal_2d_avg = np.average(signal_2d_avg, axis=0)

    # smooth signal
    # signal_2d_gaussian = ndimage.gaussian_filter(signal_2d, 2) # Best

    if return_info:
        y_frames = np.arange(y_start, y_end)
        x_frames = np.arange(x_start, x_end)

        return t, y, x, y_frames, x_frames, signal_2d

    return signal_2d  # signal_2d_gaussian
