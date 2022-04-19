'''
Classes to create training and testing datasets
'''

import os
import os.path
import glob

import imageio
import pandas as pd
import ntpath

import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
from scipy.ndimage.filters import convolve

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from dataset_tools import (numpy_to_mask, final_mask, get_chunks,
                          annotations_mask, annotations_mask_puffs,
                          get_fps, video_spline_interpolation,
                          remove_avg_background, concat_sin_channels)


__all__ = ["SparkPuffDataset", "SparkPuffTestDataset",
           "MaskDataset", "MaskTestDataset",
           "SparkDataset", "SparkTestDataset"]


basepath = os.path.dirname("__file__")



class SparkPuffDataset(Dataset):

    def __init__(self,
                 base_path=os.path.join(basepath,"..","..",
                                        "data","manual_annotations"),
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 radius_event = 2.5, remove_background = False):

        self.base_path = base_path
        self.files = sorted(glob.glob(os.path.join(self.base_path,
                                                   "video", "*.tif")))

        self.data = [np.asarray(imageio.volread(file)) for file in self.files]

        if remove_background:
            self.data = [remove_avg_background(sample) for sample in self.data]

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))

            self.data = [np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in sample])
                                            for sample in self.data]

        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.data = [convolve(sample, _smooth_filter)
                         for sample in self.data]

        if resampling:
            self.fps = [get_fps(file) for file in self.files]
            self.data = [video_spline_interpolation(video, video_path,
                                                    resampling_rate)
                            for video,video_path in zip(self.data,self.files)]

        # compute chunks indices
        self.step = step
        self.duration = duration

        self.lengths = [video.shape[0] for video in self.data]
        # blocks in each video:
        self.blocks_number = [((length-self.duration)//self.step)+1
                                for length in self.lengths]
        # number of blocks in preceding videos in data:
        self.preceding_blocks = np.roll(np.cumsum(self.blocks_number),1)
        self.tot_blocks = self.preceding_blocks[0]
        self.preceding_blocks[0] = 0

        # spark annotations
        self.csv_files = sorted(glob.glob(os.path.join(self.base_path,
                                                       "sparks", "*.csv")))
        self.csv_data = [pd.read_csv(file).values.astype('int')
                         for file in self.csv_files]
        self.sparks = [numpy_to_mask(csv, video.shape)
                       for csv,video in zip(self.csv_data,self.data)]

        # wave annotations
        self.wave_files = sorted(glob.glob(os.path.join(self.base_path,
                                                        "waves", "*.tif")))
        self.waves = [2*np.asarray(imageio.volread(file)).astype('int')
                      for file in self.wave_files]

        # puff annotations
        self.puff_files = sorted(glob.glob(os.path.join(self.base_path,
                                                        "puffs", "*.tif")))
        self.puffs = [3*np.asarray(imageio.volread(file)).astype('int')
                      for file in self.puff_files]

        # combine sparks, waves and puffs into same array
        self.annotations = [annotations_mask_puffs(s,w,p,radius_event)
                            for s,w,p in zip(self.sparks,self.waves,self.puffs)]

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        #index of video containing chunk idx
        vid_id = np.where(self.preceding_blocks == max([y for y in self.preceding_blocks if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[vid_id][chunks[chunk_id]]

        return chunk, labels



class SparkPuffTestDataset(Dataset):
    # dataset that loads a single video for testing

    def __init__(self,
                 base_path=os.path.join(basepath,"..","..",
                                        "data","manual_annotations"),
                 video_name = "130918_C_ET-1",
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 radius_event = 2.5, remove_background = False):

        self.base_path = base_path
        self.video_name = video_name
        self.file = os.path.join(self.base_path, "video_test", self.video_name + ".tif")

        self.video = imageio.volread(self.file)

        if remove_background:
            self.video = remove_avg_background(self.video)

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],[1/16,1/2,1/16],[1/16,1/16,1/16]))
            self.video = np.asarray([convolve2d(frame, _smooth_filter, mode = 'same', boundary = 'symm') for frame in self.video])

        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.video = convolve(self.video, _smooth_filter)

        if resampling:
            self.fps = get_fps(self.file)
            self.video = video_spline_interpolation(self.video, self.file, resampling_rate)

        self.step = step
        self.duration = duration
        self.length = self.video.shape[0]
        self.pad = 0

        # if necessary, pad empty frames at the end
        if (((self.length-self.duration)/self.step) % 1 != 0):
            self.pad = (self.duration
                       + self.step*(1+(self.length-self.duration)//self.step)
                       - self.length)
            self.video = np.pad(self.video, ((0,self.pad),(0,0),(0,0)),
                                mode='constant', constant_values=0)
            self.length = self.length + self.pad

        self.blocks_number = ((self.length-self.duration)//self.step)+1 # blocks in the video

        # spark annotations
        self.csv_file = os.path.join(self.base_path, "sparks_test", self.video_name + ".csv")
        self.csv_data = pd.read_csv(self.csv_file).values.astype('int')
        self.sparks = numpy_to_mask(self.csv_data, self.video.shape)

        # wave annotations
        self.wave_file = os.path.join(self.base_path, "waves_test", self.video_name + ".tif")
        self.waves = 2*np.asarray(imageio.volread(self.wave_file)).astype(int)

        #puff annotations
        self.puff_file = os.path.join(self.base_path, "puffs_test", self.video_name + ".tif")
        self.puffs = 3*np.asarray(imageio.volread(self.puff_file)).astype(int)

        # combine sparks, waves and puffs into same array
        self.annotations = annotations_mask_puffs(self.sparks, self.waves, self.puffs, radius_event = radius_event)


    def __len__(self):
        return self.blocks_number

    def __getitem__(self, chunk_id):
        chunks = get_chunks(self.length, self.step, self.duration)
        chunk = self.video[chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[chunks[chunk_id]]

        return chunk, labels


class MaskDataset(Dataset):

    def __init__(self, base_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False):

        # base_path is the folder containing the whole dataset (train and test)

        self.base_path = base_path
        self.files = sorted(glob.glob(os.path.join(self.base_path,
                                                   "videos", "*.tif")))

        self.data = [np.asarray(imageio.volread(file)) for file in self.files]
        #self.ignore_index = ignore_index
        #self.radius_ignore = radius_ignore

        if remove_background:
            self.data = [remove_avg_background(video) for video in self.data]

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.data = [np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in video])
                                            for video in self.data]

        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.data = [convolve(video, _smooth_filter) for video in self.data]

        if resampling:
            self.fps = [get_fps(file) for file in self.files]
            self.data = [video_spline_interpolation(video, video_path,
                                                    resampling_rate)
                            for video,video_path in zip(self.data,self.files)]


        # compute chunks indices
        self.step = step
        self.duration = duration
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()

        # import annotation masks
        self.annotations_files = sorted(glob.glob(os.path.join(self.base_path,
                                              "masks", "*.tif")))
        self.annotations = [np.asarray(imageio.volread(f)).astype('int')
                            for f in self.annotations_files]
        #self.annotations = [self.add_ignore_region(video, self.radius_ignore,
        #                    self.ignore_index) for video in self.annotations]


    def compute_chunks_indices(self):
        lengths = [video.shape[0] for video in self.data]
        # blocks in each video :
        blocks_number = [((length-self.duration)//self.step)+1
                         for length in lengths]
        # number of blocks in preceding videos in data :
        preceding_blocks = np.roll(np.cumsum(blocks_number),1)
        tot_blocks = preceding_blocks[0]
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    #def add_ignore_region(self, annotation_mask, radius_ignore, ignore_index):
    #    binary_mask = np.where(np.logical_or(annotation_mask == 0,
    #                           annotation_mask == ignore_index), 1, 0)
    #    dt = ndimage.distance_transform_edt(binary_mask)
    #    annotation_mask[np.logical_and(dt < radius_ignore,
    #                    annotation_mask == 0)] = ignore_index

    #    return annotation_mask

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        #index of video containing chunk idx
        vid_id = np.where(self.preceding_blocks == max([y
                          for y in self.preceding_blocks
                          if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[vid_id][chunks[chunk_id]]

        return chunk, labels


class MaskTestDataset(Dataset): # dataset that load a single video for testing

    def __init__(self, base_path, video_name,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False):#,
                 #ignore_index = 4 , radius_ignore = 2):

        # base_path is the folder containing the whole dataset (train and test)
        # video_name is a video that must be present in the "video_test" folder
        # of the base_path

        # sparks in the masks have already the correct shape (radius and
        # ignore index) !!!

        self.base_path = base_path
        self.video_name = video_name
        self.file = os.path.join(self.base_path, "videos_test",
                                 self.video_name + ".tif")

        self.video = imageio.volread(self.file)
        #self.ignore_index = ignore_index
        #self.radius_ignore = radius_ignore

        if remove_background:
            self.video = remove_avg_background(self.video)

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.video = np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in self.video])

        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.video = convolve(self.video, _smooth_filter)

        if resampling:
            self.fps = get_fps(self.file)
            self.video = video_spline_interpolation(self.video, self.file,
                                                    resampling_rate)

        self.step = step
        self.duration = duration
        self.length = self.video.shape[0]
        self.pad = 0

        # if necessary, pad empty frames at the end
        if (((self.length-self.duration)/self.step) % 1 != 0):
            self.pad = (self.duration
                        + self.step*(1+(self.length-self.duration)//self.step)
                        - self.length)
            self.video = np.pad(self.video,((0,self.pad),(0,0),(0,0)),
                                'constant',constant_values=0)
            self.length = self.length + self.pad

        # blocks in the video :
        self.blocks_number = ((self.length-self.duration)//self.step)+1

        # annotations
        self.annotations_file = os.path.join(self.base_path,
                                             "masks_test",
                                             self.video_name + ".tif")
        self.annotations = np.asarray(imageio.volread(self.annotations_file)).astype(int)


        #self.annotations = self.add_ignore_region(self.annotations,
        #                   self.radius_ignore, self.ignore_index)

    #def add_ignore_region(self, annotation_mask, radius_ignore, ignore_index):
    #    binary_mask = np.where(np.logical_or(annotation_mask == 0,
    #                           annotation_mask == ignore_index), 1, 0)
    #    dt = ndimage.distance_transform_edt(binary_mask)
    #    annotation_mask[np.logical_and(dt < radius_ignore, annotation_mask == 0)] = ignore_index

    #    return annotation_mask

    def __len__(self):
        return self.blocks_number

    def __getitem__(self, chunk_id):
        chunks = get_chunks(self.length, self.step, self.duration)
        chunk = self.video[chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[chunks[chunk_id]]

        return chunk, labels


'''
New dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Annotation filenames are: XX_mask.tif

'''

class IDMaskDataset(Dataset):

    def __init__(self, base_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False):

        # base_path is the folder containing the whole dataset (train and test)

        self.base_path = base_path
        self.files = sorted(glob.glob(os.path.join(self.base_path,
                                                   "videos", "*.tif")))

        self.data = [np.asarray(imageio.volread(file)) for file in self.files]
        #self.ignore_index = ignore_index
        #self.radius_ignore = radius_ignore

        if remove_background:
            self.data = [remove_avg_background(video) for video in self.data]

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.data = [np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in video])
                                            for video in self.data]

        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.data = [convolve(video, _smooth_filter) for video in self.data]

        if resampling:
            self.fps = [get_fps(file) for file in self.files]
            self.data = [video_spline_interpolation(video, video_path,
                                                    resampling_rate)
                            for video,video_path in zip(self.data,self.files)]


        # compute chunks indices
        self.step = step
        self.duration = duration
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()

        # import annotation masks
        self.annotations_files = sorted(glob.glob(os.path.join(self.base_path,
                                              "masks", "*.tif")))
        self.annotations = [np.asarray(imageio.volread(f)).astype('int')
                            for f in self.annotations_files]
        #self.annotations = [self.add_ignore_region(video, self.radius_ignore,
        #                    self.ignore_index) for video in self.annotations]

        # check that video files correspond to annotation files
        assert ([ntpath.split(v)[1][:3] for v in self.files] ==
               [ntpath.split(a)[1][:3] for a in self.annotations_files]), \
               "Video and annotation filenames do not match"


    def compute_chunks_indices(self):
        lengths = [video.shape[0] for video in self.data]
        # blocks in each video :
        blocks_number = [((length-self.duration)//self.step)+1
                         for length in lengths]
        # number of blocks in preceding videos in data :
        preceding_blocks = np.roll(np.cumsum(blocks_number),1)
        tot_blocks = preceding_blocks[0]
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    #def add_ignore_region(self, annotation_mask, radius_ignore, ignore_index):
    #    binary_mask = np.where(np.logical_or(annotation_mask == 0,
    #                           annotation_mask == ignore_index), 1, 0)
    #    dt = ndimage.distance_transform_edt(binary_mask)
    #    annotation_mask[np.logical_and(dt < radius_ignore,
    #                    annotation_mask == 0)] = ignore_index

    #    return annotation_mask

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        #index of video containing chunk idx
        vid_id = np.where(self.preceding_blocks == max([y
                          for y in self.preceding_blocks
                          if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[vid_id][chunks[chunk_id]]

        return chunk, labels


class IDMaskTestDataset(Dataset): # dataset that load a single video for testing

    def __init__(self, base_path, video_name,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False, test = True):#,
                 #ignore_index = 4 , radius_ignore = 2):

        # base_path is the folder containing the whole dataset (train and test)
        # video_name is a video of the form XX_video.tif that must be present in
        # the "video_test" folder of the base_path

        # sparks in the masks have already the correct shape (radius and
        # ignore index) !!!

        self.base_path = base_path
        self.video_name = video_name

        if test == True:
            self.file = os.path.join(self.base_path, "videos_test",
                                     self.video_name + ".tif")
        else:
            self.file = os.path.join(self.base_path, "videos",
                                     self.video_name + ".tif")

        self.video = imageio.volread(self.file)
        #self.ignore_index = ignore_index
        #self.radius_ignore = radius_ignore

        if remove_background:
            self.video = remove_avg_background(self.video)

        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.video = np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in self.video])

        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.video = convolve(self.video, _smooth_filter)

        if resampling:
            self.fps = get_fps(self.file)
            self.video = video_spline_interpolation(self.video, self.file,
                                                    resampling_rate)

        self.step = step
        self.duration = duration
        self.length = self.video.shape[0]
        self.pad = 0

        # if necessary, pad empty frames at the end
        if (((self.length-self.duration)/self.step) % 1 != 0):
            self.pad = (self.duration
                        + self.step*(1+(self.length-self.duration)//self.step)
                        - self.length)
            self.video = np.pad(self.video,((0,self.pad),(0,0),(0,0)),
                                'constant',constant_values=0)
            self.length = self.length + self.pad

        # blocks in the video :
        self.blocks_number = ((self.length-self.duration)//self.step)+1

        # annotations

        if self.video_name[2] != "_":
            self.annotations_name = self.video_name
        else:
            self.annotations_name = self.video_name[:3]+"unet_mask"

        if test == True:
            self.annotations_file = os.path.join(self.base_path,
                                                 "masks_test",
                                                 self.annotations_name + ".tif")
        else:
            self.annotations_file = os.path.join(self.base_path,
                                                 "masks",
                                                  self.annotations_name + ".tif")

        self.annotations = np.asarray(imageio.volread(self.annotations_file)).astype(int)


        #self.annotations = self.add_ignore_region(self.annotations,
        #                   self.radius_ignore, self.ignore_index)

    #def add_ignore_region(self, annotation_mask, radius_ignore, ignore_index):
    #    binary_mask = np.where(np.logical_or(annotation_mask == 0,
    #                           annotation_mask == ignore_index), 1, 0)
    #    dt = ndimage.distance_transform_edt(binary_mask)
    #    annotation_mask[np.logical_and(dt < radius_ignore, annotation_mask == 0)] = ignore_index

    #    return annotation_mask

    def __len__(self):
        return self.blocks_number

    def __getitem__(self, chunk_id):
        chunks = get_chunks(self.length, self.step, self.duration)
        chunk = self.video[chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[chunks[chunk_id]]

        return chunk, labels


'''
Datasets folder are reorganised
'''

class SparkDataset(Dataset):

    def __init__(self, base_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False):

        # base_path is the folder containing the whole dataset (train and test)
        self.base_path = base_path

        # get videos and masks paths
        self.files = sorted(glob.glob(os.path.join(self.base_path,
                                                   "videos", "*[!_mask].tif")))
        self.annotations_files = sorted(glob.glob(os.path.join(self.base_path,
                                              "videos", "*_mask.tif")))

        # check that video filenames correspond to annotation filenames
        assert ((os.path.splitext(v) + "_mask") == os.path.splitext(a)
                for v,a in zip(self.files,self.annotations_files)), \
               "Video and annotation filenames do not match"

        # get videos and masks data
        self.data = [np.asarray(imageio.volread(file)) for file in self.files]
        self.annotations = [np.asarray(imageio.volread(f)).astype('int')
                            for f in self.annotations_files]

        # preprocess videos if necessary
        if remove_background:
            self.data = [remove_avg_background(video) for video in self.data]
        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.data = [np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in video])
                                            for video in self.data]
        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.data = [convolve(video, _smooth_filter) for video in self.data]
        if resampling:
            self.fps = [get_fps(file) for file in self.files]
            self.data = [video_spline_interpolation(video, video_path,
                                                    resampling_rate)
                            for video,video_path in zip(self.data,self.files)]


        # compute chunks indices
        self.step = step
        self.duration = duration
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()


    def compute_chunks_indices(self):
        lengths = [video.shape[0] for video in self.data]
        # blocks in each video :
        blocks_number = [((length-self.duration)//self.step)+1
                         for length in lengths]
        # number of blocks in preceding videos in data :
        preceding_blocks = np.roll(np.cumsum(blocks_number),1)
        tot_blocks = preceding_blocks[0]
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        #index of video containing chunk idx
        vid_id = np.where(self.preceding_blocks == max([y
                          for y in self.preceding_blocks
                          if y <= idx]))[0][0]
        #index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = get_chunks(self.lengths[vid_id],self.step,self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        labels = self.annotations[vid_id][chunks[chunk_id]]

        return chunk, labels

class SparkTestDataset(Dataset): # dataset that load a single video for testing

    def __init__(self, video_path,
                 step = 4, duration = 16, smoothing = False,
                 resampling = False, resampling_rate = 150,
                 remove_background = False, gt_available = True):

        # video_path is the complete path to the video
        # gt_available == True if ground truth annotations is available

        self.gt_available = gt_available

        # get video path and array
        self.video_path = video_path
        self.video = imageio.volread(self.video_path)

        # get mask path and array
        if self.gt_available:
            filename, ext = os.path.splitext(video_path)
            self.mask_path = filename + "_mask" + ext
            self.mask = imageio.volread(self.mask_path)

        # perform some preprocessing on videos, if necessary
        if remove_background:
            self.video = remove_avg_background(self.video)
        if smoothing == '2d':
            _smooth_filter = torch.tensor(([1/16,1/16,1/16],
                                           [1/16,1/2,1/16],
                                           [1/16,1/16,1/16]))
            self.video = np.asarray([convolve2d(frame, _smooth_filter,
                                            mode = 'same', boundary = 'symm')
                                            for frame in self.video])
        if smoothing == '3d':
            _smooth_filter = 1/52*np.ones((3,3,3))
            _smooth_filter[1,1,1] = 1/2
            self.video = convolve(self.video, _smooth_filter)
        if resampling:
            self.fps = get_fps(self.file)
            self.video = video_spline_interpolation(self.video, self.file,
                                                    resampling_rate)

        self.step = step
        self.duration = duration
        self.length = self.video.shape[0]

        # if necessary, pad empty frames at the end
        self.pad = 0
        if (((self.length-self.duration)/self.step) % 1 != 0):
            self.pad = (self.duration
                        + self.step*(1+(self.length-self.duration)//self.step)
                        - self.length)
            self.video = np.pad(self.video,((0,self.pad),(0,0),(0,0)),
                                'constant',constant_values=0)
            self.length = self.length + self.pad

        # blocks in the video :
        self.blocks_number = ((self.length-self.duration)//self.step)+1

    def __len__(self):
        return self.blocks_number

    def __getitem__(self, chunk_id):
        chunks = get_chunks(self.length, self.step, self.duration)
        chunk = self.video[chunks[chunk_id]]
        chunk = chunk / chunk.max()
        chunk = np.float32(chunk)

        if self.gt_available:
            labels = self.mask[chunks[chunk_id]]

            return chunk, labels

        return chunk
