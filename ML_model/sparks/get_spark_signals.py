# 30.11.2020
# Functions to extract the signal of the sparks present in the annotations
# Run this script to plot a few samples (1D signal and 2D signal)

import numpy as np
import glob
import os
import imageio
import ntpath

from scipy import spatial, optimize, ndimage
from skimage import morphology

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from correspondences import process_spark_prediction

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

    start = max(0,start)
    stop = min(t,stop)

    frame_mask = create_circular_mask(h,w,center,radius)

    video_mask = np.zeros((t,h,w), dtype=bool)
    video_mask[start:stop] = frame_mask

    return video_mask


def get_spark_signal(video,
                     sparks_labelled,
                     center,
                     radius,
                     context_duration,
                     return_info = False):
    # video is the original video sample
    # sparks_labelled is a mask containing the segmentation of the spark events (1 integer for every event)
    # center [t y x] is the center of the selected event to plot
    # radius is the radius of the considered region around the center the spark for averaging
    # context_duration is the number of frames included in the analysis before and after the event

    t,y,x = center
    event_idx = sparks_labelled[t,y,x] # event_idx = 1,2,...,n_sparks

    assert event_idx != 0, (
    "given center does not correspond to any event in the given labelled mask")

    loc = ndimage.measurements.find_objects(sparks_labelled)[event_idx-1]

    assert loc[0].start <= t and loc[0].stop > t, "something weird is wrong"

    # get mask representing sparks location (with radius and context)
    start = max(0,loc[0].start - context_duration)
    stop = min(video.shape[0],loc[0].stop + context_duration)
    signal_mask = create_signal_mask(*sparks_labelled.shape,
                                     start, stop,
                                     (x,y), radius)

    frames = np.arange(start,stop)
    signal = np.average(video[start:stop],
                        axis=(1,2),
                        weights=signal_mask[start:stop])

    if return_info:
        return frames, signal, (y,x), loc[0].start, loc[0].stop

    return frames, signal


def get_spark_2d_signal(video,
                        slices,
                        coords,
                        spatial_context,
                        sigma = 2,
                        return_info = False):
    # video : original video
    # slices : slices in the 3 dimensions of a given spark (ROI)
    # coords [t y x]: center of a given spark
    # spatial_context : extend ROI corresponding to spark
    # sigma : for gaussian filtering

    # TODO: add assertion to check that coords are inside slices

    t,y,x = coords
    t_slice, y_slice, x_slice = slices

    y_start = max(0, y_slice.start-spatial_context)
    y_end = min(video.shape[1], y_slice.stop+spatial_context)

    x_start = max(0, x_slice.start-spatial_context)
    x_end = min(video.shape[2], x_slice.stop+spatial_context)

    signal_2d = video[t, y_start:y_end, x_start:x_end]

    # average over 3 frames
    #signal_2d_avg = video_array[t-1:t+1, y_start:y_end, x_start:x_end]
    #signal_2d_avg = np.average(signal_2d_avg, axis=0)

    # smooth signal
    signal_2d_gaussian = ndimage.gaussian_filter(signal_2d, 2) # Best

    if return_info:
        y_frames = np.arange(y_start, y_end)
        x_frames = np.arange(x_start, x_end)

        return t, y, x, y_frames, x_frames, signal_2d_gaussian

    return signal_2d_gaussian



if __name__ == "__main__":

    # Import .tif events file as numpy array
    data_path = os.path.join("..","data","annotation_masks")
    sample_name = "140708_B_ET-1"

    sparks_name = os.path.join("spark_preds_TEMP",sample_name+"_sparks.tif")
    video_name = os.path.join(data_path,"videos_test",sample_name+".tif")

    sparks_array = np.asarray(imageio.volread(sparks_name))
    video_array = np.asarray(imageio.volread(video_name)).astype('int')

    # Remove first and last frames
    half_overlap = 6 # TODO: get this from code
    sparks_array[:half_overlap] = 0
    sparks_array[-half_overlap:] = 0

    # Get sparks centres from preds
    min_radius = 3 # minimal 'radius' of a valid spark
    t_detection = 0.95 # threshold on spark preds to extract events

    coords_spark_centres = process_spark_prediction(sparks_array,
                                                   t_detection=t_detection,
                                                   min_radius=min_radius)


    # Get sparks ROIs from preds from every spark centre
    min_size = (2 * min_radius) ** sparks_array.ndim

    spark_mask = sparks_array > t_detection
    spark_mask = morphology.remove_small_objects(spark_mask,
                                                 min_size=min_size)

    # Separate events: assign an integer to every connected component
    sparks_labelled, n_sparks = ndimage.measurements.label(spark_mask,
                                                           np.ones((3,3,3),
                                                           dtype=np.int))
    # events labelled with values 1,2,...,n_sparks

    # Compute single ROIs
    sparks_rois = ndimage.measurements.find_objects(sparks_labelled)
    # i-th element correspond to event with index i+1

    # Assign a ROI (slices) to each centre
    slices_spark_centres = []
    for t,y,x in coords_spark_centres:
        event_nb = sparks_labelled[t,y,x]
        slices_spark_centres.append(sparks_rois[event_nb-1])


    # Define radius around event center and context duration
    radius = 3
    context_duration = 30

    # Get 1D signal along time and plot it for a few samples
    plt.rcParams.update({'font.size': 5})

    plt.figure(figsize=(20,10))
    plt.suptitle("Signal around some sample sparks", fontsize=10)

    for idx, centre in enumerate(coords_spark_centres[:10]):
        frames,signal,(y,x),start,stop = get_spark_signal(video_array,
                                                          sparks_labelled,
                                                          centre,radius,
                                                          context_duration,
                                                          return_info=True)

        ax = plt.subplot(2,5,idx+1)
        ax.set_title(f"Location ({x},{y}), frames {start}-{stop}")
        ax.axvspan(start, stop, facecolor='green', alpha=0.25)
        plt.plot(frames,signal,color='darkgreen')


    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.05, right=0.95)
    plt.show()

    # Get 2D signal around center of some sparks and plot them
    spatial_context = 10

    plt.figure(figsize=(20,10))

    if n_sparks == 1:
        plt.suptitle("2D smoothed signal of a sample spark", fontsize=10)
        idx = 1

        t,y,x,y_axis,x_axis,signal_2d = get_spark_2d_signal(video_array,
                                                            slices_spark_centres[idx-1],
                                                            coords_spark_centres[idx-1],
                                                            spatial_context,
                                                            sigma = 2,
                                                            return_info = True)

        x_axis, y_axis = np.meshgrid(x_axis,y_axis)

        ax = plt.subplot(1,1,idx, projection='3d')
        ax.set_title(f"Spark at location ({x},{y}) at frame {t}")
        ax.plot_surface(y_axis, x_axis, signal_2d, cmap=cm.coolwarm, linewidth=0)#, antialiased=False)
    else:
        plt.suptitle("2D smoothed signals of some sample sparks", fontsize=10)
        plt.rcParams.update({'font.size': 5})

        num_plots = min(n_sparks,10)
        for idx in range(1,num_plots+1):
            n_cols = num_plots//2 if num_plots%2 == 0 else (num_plots//2)+1
            t,y,x,y_axis,x_axis,signal_2d = get_spark_2d_signal(video_array,
                                                                slices_spark_centres[idx-1],
                                                                coords_spark_centres[idx-1],
                                                                spatial_context,
                                                                sigma = 2,
                                                                return_info = True)

            x_axis, y_axis = np.meshgrid(x_axis,y_axis)

            ax = plt.subplot(2,n_cols,idx, projection='3d')
            ax.set_title(f"Location ({x},{y}), frame {t}")
            ax.plot_surface(y_axis, x_axis, signal_2d, cmap=cm.coolwarm, linewidth=0)#, antialiased=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9, bottom=0.1, left=0.05, right=0.95)
    plt.show()
