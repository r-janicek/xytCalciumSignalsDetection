# 07.11.2020
# Functions to extract the signal of the puffs present in the annotations
# Run this script to plot a few samples

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy import ndimage


def get_puff_signal(video, puffs_labelled, event_idx, context_duration,
                    mode = 'avg', return_info = False):
    # puffs_labelled : mask containing the segmentation of the puff events
    #                  (1 integer for every event)
    # event_idx :      index of the chosen event (0,1,...,n_puffs-1)
    # return a mask == 1 where the puff 'event_idx' is and == 0 elsewhere
    # (plus a context before and afterwards)

    assert event_idx < puffs_labelled.max(), (
    f"given idx is too large, video contains only {puffs_labelled.max()} events")

    assert mode == 'sum' or  mode == 'avg', "mode must be 'sum' or 'avg'"
    # 'sum' is probably useless

    # extract event mask
    mask = np.where(puffs_labelled == event_idx+1, 1, 0)

    # add regions before and after the event
    nonzero_frames = np.nonzero(mask)[0]
    first_frame = min(nonzero_frames)
    last_frame = max(nonzero_frames)

    start = max(0, first_frame - context_duration)
    stop = min(puffs_labelled.shape[0], last_frame + 1 + context_duration)

    first_mask = mask[first_frame]
    last_mask = mask[last_frame]

    # repeat first mask in frames preceding the event
    mask[start:first_frame] = first_mask
    # repeat last mask in frames following the event
    mask[last_frame:stop] = last_mask

    if mode == 'sum':
        signal = video*mask
        signal = np.sum(signal, axis=(1,2))[start:stop]

    if mode == 'avg':
        signal = np.average(video[start:stop],
                            axis=(1,2),
                            weights=mask[start:stop])

    frames = np.arange(start,stop)

    if return_info:
        return frames, signal, first_frame, last_frame + 1

    return frames, signal


if __name__ == "__main__":

    # Import .tif events file as numpy array
    data_path = os.path.join("..","data","annotation_masks")
    sample_name = "131119_I_ET-1.tif"
    events_name = os.path.join(data_path,"masks_test",sample_name)
    video_name = os.path.join(data_path,"videos_test",sample_name)

    events_array = np.asarray(imageio.volread(events_name)).astype('int')
    video_array = np.asarray(imageio.volread(video_name)).astype('int')


    # only analyse puff events, get rid of the rest
    puffs_array = np.where(events_array==3,1,0)


    # separate events: assign an integer to every connected component
    puffs_labelled, n_puffs = ndimage.measurements.label(puffs_array,
                                                         np.ones((3,3,3),
                                                         dtype=np.int))

    assert n_puffs > 0, "Video does not contain any labelled puffs"

    # define context duration
    context_duration = 50 # NOT WELL-DEFINED WRT PUFF ROIs


    # Plot a sample
    plt.figure(figsize=(20,10))

    if n_puffs == 1:
        plt.suptitle("Signal averaged over a sample puff", fontsize=10)
        idx = 0

        frames, signal, start, stop = get_puff_signal(video_array,
                                                      puffs_labelled,
                                                      idx,context_duration,
                                                      return_info=True)

        ax = plt.subplot(1,1,idx+1)
        ax.set_title(f"Puff at frames {start}-{stop}")
        ax.axvspan(start, stop, facecolor='green', alpha=0.3)
        plt.plot(frames,signal,color='darkgreen')
    else:
        plt.suptitle("Signal averaged over some sample puffs", fontsize=10)
        plt.rcParams.update({'font.size': 6})

        num_plots = min(n_puffs,10)
        for idx in range(num_plots):
            n_cols = num_plots//2 if num_plots%2 == 0 else (num_plots//2)+1
            frames, signal, start, stop = get_puff_signal(video_array,
                                                          puffs_labelled,
                                                          idx,context_duration,
                                                          return_info=True)

            ax = plt.subplot(2,n_cols,idx+1)
            ax.set_title(f"Puff at frames {start}-{stop}")
            ax.axvspan(start, stop, facecolor='green', alpha=0.3)
            plt.plot(frames,signal,color='darkgreen')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9, left=0.05, right=0.95)
    plt.show()
