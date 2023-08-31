'''
Classes to create training and testing datasets
'''

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from data_processing_tools import detect_spark_peaks
from in_out_tools import load_annotations_ids, load_movies_ids
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from PIL import Image

__all__ = ["SparkDataset"]


basepath = os.path.dirname("__file__")
logger = logging.getLogger(__name__)


'''
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Class label filenames are: XX_class_label.tif
Event label filenames are: XX_event_label.tif
'''


class SparkDataset(Dataset):

    def __init__(self, base_path, sample_ids, testing,
                 step=4, duration=16, smoothing=False,
                 resampling=False, resampling_rate=150,
                 remove_background='average', temporal_reduction=False,
                 num_channels=1, normalize_video='chunk',
                 only_sparks=False, sparks_type='peaks', ignore_index=4,
                 ignore_frames=0, gt_available=True, inference=None):
        r'''
        Dataset class for SR-calcium releases segmented dataset.

        base_path:          directory where movies and annotation masks are
                            saved
        sample_ids:         list of sample IDs used to create the dataset
        testing:            if True, apply additional processing to data to
                            compute metrics during validation/testing
        step:               step between two chunks extracted from the sample
        duration:           duration of a chunk
        smoothing:          if '2d' or '3d', preprocess movie with simple
                            convolution (probably useless)
        resampling_rate:    resampling rate used if resampling the movies
        remove_background:  if 'moving' or 'average', remove background from
                            input movies accordingly
        temporal_reduction: set to True if using TempRedUNet (sample processed
                            in conv layers before unet)
        num_channels:       >0 if using temporal_reduction, value depends on
                            temporal reduction configuration
        normalize_video:    if 'chunk', 'movie' or 'abs_max' normalize input
                            video accordingly
        only_sparks:        if True, train using only sparks annotations
        sparks_type:        can be 'raw' or 'peaks' (use smaller annotated ROIs)
        ignore_frames:      if testing, used to ignore events in first and last
                            frames
        gt_available:       True if sample's ground truth is available
        inference:          this is used only during inference (not at
                            training) values can be 'overlap', or 'average'.
                            If 'overlap', on overlapping frames take first half
                            from preceding chunk and second half from following
                            chunk. If 'average', compute average on overlapping
                            chunks (TODO: define best average method).
        '''

        # base_path is the folder containing the whole dataset (train and test)
        self.base_path = base_path

        # physiological params (for spark peaks results)
        self.pixel_size = 0.2  # 1 pixel = 0.2 um x 0.2 um
        # min distance in space
        self.min_dist_xy = round(1.8 / self.pixel_size)
        self.time_frame = 6.8  # 1 frame = 6.8 ms
        self.min_dist_t = round(20 / self.time_frame)  # min distance in time

        # dataset parameters
        self.testing = testing
        self.inference = inference
        self.gt_available = gt_available
        self.sample_ids = sample_ids
        self.only_sparks = only_sparks
        self.sparks_type = sparks_type
        self.ignore_index = ignore_index
        self.ignore_frames = ignore_frames

        self.duration = duration
        self.step = step

        # if performing inference, get video name and take note of the padding
        # applied to the movie
        if self.inference is not None:
            # check that dataset contains a single video
            assert len(sample_ids) == 1, \
                f"Dataset set to inference mode, but it contains "\
                f"{len(sample_ids)} samples: {sample_ids}."

            # check that inference mode is valid
            assert self.inference in ['overlap', 'average'], \
                "If testing, select one inference mode from "\
                "'overlap' and 'average'."

            self.video_name = sample_ids[0]
            self.pad = 0

        self.temporal_reduction = temporal_reduction
        if self.temporal_reduction:
            self.num_channels = num_channels

        self.normalize_video = normalize_video
        self.remove_background = remove_background

        # get video samples
        self.data = list(load_movies_ids(data_folder=self.base_path,
                                         ids=sample_ids,
                                         names_available=True,
                                         movie_names="video"
                                         ).values())
        self.data = [torch.from_numpy(movie.astype('int'))
                     for movie in self.data]  # int32

        if inference is not None:
            # need to keep track of movie duration, in case it is shorter than
            # `chunks_duration` and a pad is added
            self.movie_duration = self.data[0].shape[0]

        if self.testing:
            assert self.gt_available, \
                "If testing, ground truth must be available."
            assert len(sample_ids) == 1, \
                f"Dataset set to testing mode, but it contains "\
                f"{len(sample_ids)} samples: {sample_ids}."

        # get annotation masks, if ground truth is available
        if self.gt_available:
            # preprocess annotations if necessary
            assert self.sparks_type in ['peaks', 'small', 'raw'], \
                "Sparks type should be 'peaks', 'small' or 'raw'."

            if self.sparks_type == 'raw':
                # get class label masks
                self.annotations = list(load_annotations_ids(
                    data_folder=self.base_path,
                    ids=sample_ids,
                    mask_names="class_label"
                ).values())
                # no preprocessing

            elif self.sparks_type == 'peaks':
                # TODO: if necessary implement mask that contain only spark peaks
                pass

            elif self.sparks_type == 'small':
                # reduce the size of sparks annotations and replace difference
                # with undefined label (4)

                self.annotations = list(load_annotations_ids(
                    data_folder=self.base_path,
                    ids=sample_ids,
                    mask_names="class_label_small_sparks"
                ).values())

            self.annotations = [torch.from_numpy(mask)
                                for mask in self.annotations]  # int8

            # if testing, load the event label masks too (for peaks detection)
            # and compute the location of the spark peaks

            if self.testing:
                # if testing, the dataset contain a single video

                self.events = list(load_annotations_ids(
                    data_folder=self.base_path,
                    ids=sample_ids,
                    mask_names="event_label"
                ).values())[0]
                self.events = torch.from_numpy(self.events)  # int8

                logger.debug("Computing spark peaks...")
                spark_mask = np.where(self.annotations[0] == 1,
                                      self.events, 0)
                self.coords_true = detect_spark_peaks(movie=self.data[0],
                                                      spark_mask=spark_mask,
                                                      sigma=2,
                                                      max_filter_size=10)
                logger.debug(
                    f"Sample {self.video_name} contains {len(self.coords_true)} sparks.")

        # preprocess videos if necessary
        if self.remove_background == 'average':
            self.data = [self.remove_avg_background(
                video) for video in self.data]

        if smoothing == '2d':
            logger.info("Applying 2d gaussian blur to videos...")
            # apply gaussian blur to each frame of each video in self.data
            from torchvision.transforms import GaussianBlur
            gaussian_blur = GaussianBlur(kernel_size=(3, 3), sigma=1.0)
            self.data = [gaussian_blur(video) for video in self.data]

        if smoothing == '3d':
            _smooth_filter = 1/52*torch.ones((3, 3, 3))
            _smooth_filter[1, 1, 1] = 1/2
            self.data = [convolve(video, _smooth_filter)
                         for video in self.data]

        if resampling:
            self.fps = [self.get_fps(file) for file in self.files]
            self.data = [self.video_spline_interpolation(video, video_path,
                                                         resampling_rate)
                         for video, video_path in zip(self.data, self.files)]

        if self.normalize_video == 'movie':
            self.data = [(video - video.min()) / (video.max() - video.min())
                         for video in self.data]
        elif self.normalize_video == 'abs_max':
            absolute_max = np.iinfo(np.uint16).max  # 65535
            self.data = [(video-video.min())/(absolute_max-video.min())
                         for video in self.data]

        # pad movies shorter than chunk duration with zeros before beginning and
        # after end
        self.data = [self.pad_short_video(video) for video in self.data]
        if self.gt_available:
            self.annotations = [self.pad_short_video(mask,
                                padding_value=ignore_index)
                                for mask in self.annotations]

        # pad movies whose length does not match chunks_duration and step params
        self.data = [self.pad_end_of_video(video) for video in self.data]
        if self.gt_available:
            self.annotations = [self.pad_end_of_video(mask,
                                mask=True, padding_value=ignore_index)
                                for mask in self.annotations]

        # compute chunks indices
        self.lengths, self.tot_blocks, self.preceding_blocks = self.compute_chunks_indices()

        # if using temporal reduction, shorten the annotations duration
        if self.temporal_reduction and self.gt_available:
            self.annotations = [self.shrink_mask(mask)
                                for mask in self.annotations]

        # print("annotations shape", self.annotations[-1].shape)

        # if training with sparks only, set puffs and waves to 0
        if self.only_sparks and self.gt_available:
            logger.info("Removing puff and wave annotations in training set")
            self.annotations = [torch.where(torch.logical_or(mask == 1, mask == 4),
                                            mask, 0) for mask in self.annotations]

    def pad_short_video(self, video, padding_value=0):
        # pad videos shorter than chunk duration with zeros on both sides
        if video.shape[0] < self.duration:
            pad = self.duration - video.shape[0]
            video = F.pad(video, (0, 0, 0, 0, pad//2, pad//2+pad % 2),
                          'constant', value=padding_value)

            assert video.shape[0] == self.duration, "padding is wrong"

            logger.debug("Added padding to short video")

        return video

    def pad_end_of_video(self, video, mask=False, padding_value=0):
        # pad videos whose length does not match with chunks_duration and
        # step params
        length = video.shape[0]
        if (((length-self.duration)/self.step) % 1 != 0):
            pad = (self.duration
                   + self.step*(1+(length-self.duration)//self.step)
                   - length)

            # if testing, store the pad lenght as class attribute
            if self.inference is not None:
                self.pad = pad

            video = F.pad(video, (0, 0, 0, 0, pad//2, pad//2+pad % 2),
                          'constant', value=padding_value)
            length = video.shape[0]
            if not mask:
                logger.debug(
                    f"Added padding of {pad} frames to video with unsuitable duration")

        assert ((length-self.duration) /
                self.step) % 1 == 0, "padding at end of video is wrong"

        return video

    def compute_chunks_indices(self):
        lengths = [video.shape[0] for video in self.data]
        # blocks in each video :
        blocks_number = [((length-self.duration)//self.step)+1
                         for length in lengths]
        blocks_number = torch.as_tensor(blocks_number)
        # number of blocks in preceding videos in data :
        preceding_blocks = torch.roll(torch.cumsum(blocks_number, dim=0), 1)
        tot_blocks = preceding_blocks[0].detach().item()
        preceding_blocks[0] = 0

        return lengths, tot_blocks, preceding_blocks

    def __len__(self):
        return self.tot_blocks

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.__len__() + idx

        # index of video containing chunk idx
        vid_id = torch.where(self.preceding_blocks == max(
            [y for y in self.preceding_blocks if y <= idx]))[0][0]
        # index of chunk idx in video vid_id
        chunk_id = idx - self.preceding_blocks[vid_id]

        chunks = self.get_chunks(self.lengths[vid_id],
                                 self.step,
                                 self.duration)

        chunk = self.data[vid_id][chunks[chunk_id]]

        if self.remove_background == 'moving':
            # remove the background of the single chunk
            # !! se migliora molto i risultati, farlo nel preprocessing che se
            #    no è super lento
            chunk = self.remove_avg_background(chunk)

        if self.normalize_video == 'chunk':
            chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
        # assert chunk.min() >= 0 and chunk.max() <= 1, \
        # "chunk values not normalized between 0 and 1"
        # print("min and max value in chunk:", chunk.min(), chunk.max())

        # print("vid id", vid_id)
        # print("chunk id", chunk_id)
        # print("chunks", chunks[chunk_id])

        if self.gt_available:
            if self.temporal_reduction:
                assert self.lengths[vid_id] % self.num_channels == 0, \
                    "video length must be a multiple of num_channels"
                assert self.step % self.num_channels == 0, \
                    "step must be a multiple of num_channels"
                assert self.duration % self.num_channels == 0, \
                    "duration must be multiple of num_channels"

                masks_chunks = self.get_chunks(self.lengths[vid_id]//self.num_channels,
                                               self.step//self.num_channels,
                                               self.duration//self.num_channels)

                # print("mask chunk", masks_chunks[chunk_id])
                labels = self.annotations[vid_id][masks_chunks[chunk_id]]
            else:
                labels = self.annotations[vid_id][chunks[chunk_id]]

            return chunk, labels

        return chunk

    def remove_avg_background(self, video):
        # remove average background

        if torch.is_tensor(video):
            avg = torch.mean(video, axis=0)
            return torch.add(video, -avg)
        else:
            avg = np.mean(video, axis=0)
            return np.add(video, -avg)

    def get_chunks(self, video_length, step, duration):
        n_blocks = ((video_length-duration)//(step))+1

        return (torch.arange(duration)[None, :]
                + step*torch.arange(n_blocks)[:, None])

    ###################### functions for video resampling ######################

    def get_times(self, video_path):
        # get times at which video frames where sampled
        description = Image.open(video_path).tag[270][0].split('\r\n')
        description = [line.split('\t') for line in description]
        description = [[int(i) if i.isdigit() else i for i in line]
                       for line in description]
        description = [d for d in description if isinstance(d[0], int)]
        return np.array([float(line[1]) for line in description])

    def get_fps(self, video_path):
        # compute estimated video fps value wrt sampling times deltas
        times = self.get_times(video_path)
        deltas = np.diff(times)
        return 1/np.mean(deltas)

    def video_spline_interpolation(self, video, video_path, new_fps=150):
        # interpolate video wrt new sampling times
        frames_time = self.get_times(video_path)
        f = interp1d(frames_time, video, kind='linear', axis=0)
        assert (len(frames_time) == video.shape[0])
        frames_new = np.linspace(frames_time[0],
                                 frames_time[-1],
                                 int(frames_time[-1]*new_fps))

        return f(frames_new)

    ##################### functions for temporal reduction #####################

    def shrink_mask(self, mask):
        # input is an annotation mask with the number of channels of the unet
        # output is a shrinked mask where :
        # {0} -> 0
        # {0, i}, {i} -> i, i = 1,2,3
        # {0, 1, i}, {1, i} -> 1, i = 2,3
        # {0, 2 ,3}, {2, 3} -> 2
        # and each voxel in the output corresponds to 'num_channels' voxels in
        # the input

        assert mask.shape[0] % self.num_channels == 0, \
            "in shrink_mask the duration of the mask is not a multiple of num_channels"

        # get subtensor of duration 'self.num_channels'
        sub_masks = np.split(mask, mask.shape[0]//self.num_channels)

        # print(sub_masks[0].shape)
        # print(len(sub_masks))

        new_mask = []
        # for each subtensor get a single frame
        for sub_mask in sub_masks:
            new_frame = np.array([[self.get_new_voxel_label(sub_mask[:, y, x])
                                   for x in range(sub_mask.shape[2])]
                                  for y in range(sub_mask.shape[1])])
            # print(new_frame.shape)
            new_mask.append(new_frame)

        new_mask = np.stack(new_mask)
        return new_mask

    def get_new_voxel_label(self, voxel_seq):
        # voxel_seq is a vector of 'num_channels' elements
        # {0} -> 0
        # {0, i}, {i} -> i, i = 1,2,3
        # {0, 1, i}, {1, i} -> 1, i = 2,3
        # {0, 2 ,3}, {2, 3} -> 3
        # print(voxel_seq)

        if np.max(voxel_seq == 0):
            return 0
        elif 1 in voxel_seq:
            return 1
        elif 3 in voxel_seq:
            return 3
        else:
            return np.max(voxel_seq)


# define dataset class that will be used for training the UNet-convLSTM model

class SparkDatasetLSTM(SparkDataset):
    '''
    SparkDataset class for UNet-convLSTM model.

    The dataset is adapted in such a way that each chunk is a sequence of
    frames centered around the frame to be predicted.
    The label is the segmentation mask of the central frame.
    '''

    def __init__(self, base_path, sample_ids, testing,
                 duration=16, smoothing=False,
                 resampling=False, resampling_rate=150,
                 remove_background='average', temporal_reduction=False,
                 num_channels=1, normalize_video='chunk',
                 only_sparks=False, sparks_type='peaks', ignore_index=4,
                 gt_available=True, inference=None):
        '''
        step = 1 and ignore_frames = 0 because we need to have a prediction
        for each frame.
        '''

        super().__init__(base_path=base_path, sample_ids=sample_ids,
                         testing=testing, step=1, duration=duration,
                         smoothing=smoothing, resampling=resampling,
                         resampling_rate=resampling_rate, remove_background=remove_background,
                         temporal_reduction=temporal_reduction, num_channels=num_channels,
                         normalize_video=normalize_video, only_sparks=only_sparks,
                         sparks_type=sparks_type, ignore_index=ignore_index,
                         ignore_frames=0, gt_available=gt_available, inference=inference)

    def pad_short_video(self, video, padding_value=0):
        '''
        Instead of padding the video with zeros, pad it with the first
        and last frame of the video.
        '''

        if video.shape[0] < self.duration:
            pad = self.duration - video.shape[0]
            video = F.pad(video, (0, 0, 0, 0, pad//2, pad//2+pad % 2),
                          'replicate')

            assert video.shape[0] == self.duration, "padding is wrong"

            logger.debug("Added padding to short video")

        return video

    def pad_end_of_video(self, video, mask=False, padding_value=0):
        '''
        Pad duration/2 frames at the beginning and at the end of the video
        with the first and last frame of the video.
        '''
        length = video.shape[0]

        # check that duration is odd
        assert self.duration % 2 == 1, "duration must be odd"

        pad = self.duration - 1

        # if testing, store the pad lenght as class attribute
        if self.testing:
            self.pad = pad

        if mask:
            video = video.float()  # cast annotations to float32

        replicate = nn.ReplicationPad3d((0, 0, 0, 0, pad//2, pad//2))
        video = replicate(video[None, :])[0]

        if mask:
            video = video.int()  # cast annotations back to int32

        # check that duration of video is original duration + chunk duration - 1
        assert video.shape[0] == length + self.duration - 1, \
            "padding at end of video is wrong"

        return video

    def __getitem__(self, idx):
        '''
        As opposed to the SparkDataset class, here the label is just the
        middle frame of the chunk.
        '''
        sample = super().__getitem__(idx)

        if self.gt_available:
            # extract middle frame from label
            label = sample[1][self.duration//2]
            # sample = (sample[0], label[None, :, :])
            sample = (sample[0], label)

        return sample
