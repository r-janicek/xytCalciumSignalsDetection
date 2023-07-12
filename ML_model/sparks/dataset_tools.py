"""
Functions needed to preprocess the csv files, create the structure of data in
the spark datasets and compute the weights of the network.
"""


import numpy as np
import torch
from torch import nn
from scipy import ndimage
from scipy.interpolate import interp1d
from PIL import Image

import torch


__all__ = ["numpy_to_mask",
           "final_mask",
           "get_chunks",
           "random_flip",
           "compute_class_weights",
           "compute_class_weights_waves",
           "compute_class_weights_puffs",
           "weights_init",
           "get_times",
           "get_fps",
           "annotations_interpolation",
           "video_spline_interpolation",
           "annotations_mask",
           "annotations_mask_puffs",
           "remove_avg_background",
           "concat_sin_channels",
           "random_flip_channels"]


### functions for data preproccesing ###


def numpy_to_mask(pos, shape):
    # return a numpy array of the same shape of the sample video where the peaks
    # annotated in pos have value 1 and the background has value 0

    mask = np.zeros(shape, dtype=np.int64)
    mask[pos[:, 0], pos[:, 2], pos[:, 1]] = 1
    # x and y indices are inverted in the CSV file

    return mask


def final_mask(mask, radius1=2.5, radius2=3.5, ignore_ind=2): # SLOW
    dt = ndimage.distance_transform_edt(1 - mask)
    new_mask = np.zeros(mask.shape, dtype=np.int64)
    new_mask[dt < radius2] = ignore_ind
    new_mask[dt < radius1] = 1

    return new_mask


def annotations_mask(sparks, waves, radius_sparks = 2.5, radius_ignore = 1, ignore_index = 3):
    # sparks and waves are masks
    # radius_mask = how much to increase sparks size
    # radius_ignore = radius of the uncertainty region
    # return the final annotation mask that can be used as input of the NN
    dt = ndimage.distance_transform_edt(1-sparks)
    sparks[dt < radius_sparks+radius_ignore] = ignore_index
    sparks[dt < radius_sparks] = 1

    annotations = np.where(sparks != 0, sparks, waves)
    dt = ndimage.distance_transform_edt(1-annotations.astype(bool))

    return np.where(np.logical_or(dt==0, dt>radius_ignore), annotations, ignore_index)


def annotations_mask_puffs(sparks, waves, puffs, radius_event = 2.5, radius_ignore = 1, ignore_index = 4):
    # sparks, puffs and waves are masks
    # radius_event = how much to increase sparks and puffs size
    # radius_ignore = radius of the uncertainty region
    # return the final annotation mask that can be used as input of the NN

    empty_dist = np.ones(sparks.shape)*(radius_event+radius_ignore)*2

    if np.count_nonzero(sparks) != 0:
        dt_s = ndimage.distance_transform_edt(1-sparks)
    else:
        dt_s = empty_dist

    if np.count_nonzero(puffs) != 0:
        dt_p = ndimage.distance_transform_edt(3-puffs)
    else:
        dt_p = empty_dist

    if np.count_nonzero(waves) != 0:
        dt_w = ndimage.distance_transform_edt(2-waves)
    else:
        dt_w = empty_dist

    annotations = np.zeros(sparks.shape)
    annotations[np.logical_or(dt_s <= radius_event+radius_ignore,dt_p <= radius_event+radius_ignore)] = 4
    annotations[dt_s <= radius_event] = 1
    annotations[dt_p <= radius_event] = 3
    annotations = np.where(annotations != 0, annotations, waves)
    annotations[np.logical_and(dt_w <= radius_ignore, annotations == 0)] = 4

    return annotations


def get_chunks(video_length, step, duration):
    n_blocks = ((video_length-duration)//(step))+1

    return np.arange(duration)[None,:] + step*np.arange(n_blocks)[:,None]


def random_flip(x, y):

    if np.random.uniform() > 0.5:
        x = x[..., ::-1]
        y = y[..., ::-1]

    if np.random.uniform() > 0.5:
        x = x[..., ::-1, :]
        y = y[..., ::-1, :]

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    return x, y


def random_flip_channels(x, y):

    if np.random.uniform() > 0.5:
        x = x[..., ::-1]
        y = y[..., ::-1]

    if np.random.uniform() > 0.5:
        x = x[..., ::-1, :]
        y = y[..., ::-1, :]

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    return x, y


def remove_avg_background(video):
    # remove average background
    avg = np.mean(video, axis = 0)
    return np.add(video, -avg)


def concat_sin_channels(chunk):
    shape = np.shape(chunk) # 16 x 64 x 512

    x = np.linspace(-np.pi,np.pi,shape[2]) # 512 elem
    y = np.linspace(-np.pi,np.pi,shape[1]) # 64 elem
    t = np.linspace(-np.pi,np.pi,shape[0]) # 16 elem

    # n values
    n_x = [1,2,4,8,32]
    n_y = [1,2,4,8]
    n_t = [1,2,4]

    # sin values
    x_sin = [np.sin(n*x) for n in n_x] # 5 x 512
    y_sin = [np.sin(n*y) for n in n_y] # 4 x 64
    t_sin = [np.sin(n*t) for n in n_t] # 3 x 16

    # new 3D channels
    x_sin_all = [np.broadcast_to(x_sin_n, (shape[0], shape[1], shape[2])) for x_sin_n in x_sin]

    y_sin_all = [np.broadcast_to(y_sin_n, (shape[2], shape[0], shape[1])) for y_sin_n in y_sin]
    y_sin_all = [np.transpose(channel, (1,2,0)) for channel in y_sin_all]

    t_sin_all = [np.broadcast_to(t_sin_n, (shape[1], shape[2], shape[0])) for t_sin_n in t_sin]
    t_sin_all = [np.transpose(channel, (2,0,1)) for channel in t_sin_all]

    return np.asarray([chunk] + x_sin_all + y_sin_all + t_sin_all)


### functions related to U-Net hyperparameters ###


def compute_class_weights(dataset):
    # Assuming there are only 2 classes
    count0 = 0
    count1 = 0

    with torch.no_grad():
        for _,y in dataset:
            count0 += np.count_nonzero(y==0)
            count1 += np.count_nonzero(y==1)
    total = count0 + count1
    weights = np.array([4*total/(2*count0), 0.25*total/(2*count1)])

    return np.float64(weights)


def compute_class_weights_waves(dataset, w0=1, w1=1, w2=2):
    # For 3 classes
    count0 = 0
    count1 = 0
    count2 = 0

    with torch.no_grad():
        for _,y in dataset:
            count0 += np.count_nonzero(y==0)
            count1 += np.count_nonzero(y==1)
            count2 += np.count_nonzero(y==2)

    total = count0 + count1 + count2
    weights = np.array([w0*total/(3*count0), w1*total/(3*count1), w2*total/(3*count2)])

    return np.float64(weights)


def compute_class_weights_puffs(dataset, w0=1, w1=1, w2=1, w3=1):
    # For 4 classes
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0

    with torch.no_grad():
        for _,y in dataset:
            count0 += np.count_nonzero(y==0)
            count1 += np.count_nonzero(y==1)
            count2 += np.count_nonzero(y==2)
            count3 += np.count_nonzero(y==3)

    total = count0 + count1 + count2 + count3

    w0_new = w0*total/(4*count0) if count0 != 0 else 0
    w1_new = w1*total/(4*count1) if count1 != 0 else 0
    w2_new = w2*total/(4*count2) if count2 != 0 else 0
    w3_new = w3*total/(4*count3) if count3 != 0 else 0

    weights = np.array([w0_new, w1_new, w2_new, w3_new])

    return np.float64(weights)


def weights_init(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
        stdv = np.sqrt(2/m.weight.size(1))
        m.weight.data.normal_(m.weight, std=stdv)


### functions for video resampling ###


def get_times(video_path):
    # get times at which video frames where sampled
    description = Image.open(video_path).tag[270][0].split('\r\n')
    description  = [line.split('\t') for line in description]
    description = [[int(i) if i.isdigit() else i for i in line] for line in description]
    description = [d for d in  description if isinstance(d[0], int)]
    return np.array([float(line[1]) for line in description])


def get_fps(video_path):
    # compute estimated video fps value wrt sampling times deltas
    times = get_times(video_path)
    deltas = np.diff(times)
    return 1/np.mean(deltas)


def annotations_interpolation(csv_data, fps, new_fps=150):
    # adapt annotations wrt resampling time
    csv_data_new = np.copy(csv_data)
    csv_data_new[:,0] = np.array(csv_data[:,0]*new_fps/fps, dtype=int)
    return csv_data_new


def video_spline_interpolation(video, video_path, new_fps=150):
    # interpolate video wrt new sampling times
    frames_time = get_times(video_path)
    f = interp1d(frames_time, video, kind='linear', axis=0)
    assert(len(frames_time) == video.shape[0])
    frames_new = np.linspace(frames_time[0], frames_time[-1], int(frames_time[-1]*new_fps))
    return f(frames_new)
