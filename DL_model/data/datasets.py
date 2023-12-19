"""
Classes to create datasets.

Author: Prisca Dotti
Last modified: 23.10.2023
"""

import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import interp1d
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur

from config import TrainingConfig, config
from data.data_processing_tools import detect_spark_peaks, remove_padding
from utils.in_out_tools import load_annotations_ids, load_movies_ids

__all__ = [
    "SparkDataset",
    "SparkDatasetTemporalReduction",
    "SparkDatasetResampled",
    "SparkDatasetLSTM",
    "SparkDatasetInference",
]


basepath = os.path.dirname("__file__")
logger = logging.getLogger(__name__)


"""
Dataset videos are identified by an ID of the form XX
Video filenames are: XX_video.tif
Class label filenames are: XX_class_label.tif
Event label filenames are: XX_event_label.tif
"""


class SparkDataset(Dataset):
    """
    A PyTorch Dataset class for spark detection.

    Args:
        params (TrainingConfig): A configuration object containing the
            dataset parameters.
        **kwargs: Additional keyword arguments to customize the dataset.

    Keyword Args:
        base_path (str): The base path to the dataset files on disk.
        sample_ids (List[str]): A list of sample IDs to load from disk.
        load_instances (bool): Whether to load instance data from disk.
        movies (List[np.ndarray]): A list of numpy arrays containing the movie
            data.
        labels (List[np.ndarray]): A list of numpy arrays containing the ground
            truth labels.
        instances (List[np.ndarray]): A list of numpy arrays containing the
            instance data.
        stride (int): The stride to use when generating samples from the movie
            data.

    Raises:
        ValueError: If neither `movies` nor `base_path` and `sample_ids` are
        provided.

    Attributes:
        params (TrainingConfig): The configuration object containing the dataset
            parameters.
        window_size (int): The duration of each sample in frames.
        stride (int): The stride to use when generating samples from the movie
            data.
        movies (List[torch.Tensor]): A list of PyTorch tensors containing the
            movie data.
        labels (List[torch.Tensor]): A list of PyTorch tensors containing the
            ground truth labels.
        instances (List[torch.Tensor]): A list of PyTorch tensors containing the
            instance data.
        gt_available (bool): Whether ground truth labels are available for the
            dataset.
        spark_peaks (List[Tuple[int, int]]): A list of tuples containing the
            (t, y, x) coordinates of the spark peaks in each movie.
        original_durations (List[int]): A list of the original durations of each
            movie before padding.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx: int) -> Dict[str, Any]: Returns a dictionary containing
            the data, labels, and metadata for a given sample.
    """

    def __init__(self, params: TrainingConfig, **kwargs) -> None:
        # Get the dataset and training parameters
        self.params = params

        base_path: str = kwargs.get("base_path", "")
        sample_ids: List[str] = kwargs.get("sample_ids", [])

        movies: List[np.ndarray] = kwargs.get("movies", [])

        if base_path and sample_ids:
            # Load data from disk if base_path and sample_ids are provided
            self.base_path = base_path
            self.sample_ids = sample_ids
            load_instances: bool = kwargs.get("load_instances", False)

            ### Get video samples and ground truth ###
            movies = self._load_movies()  # dict of numpy arrays
            labels = self._load_labels()  # list of numpy arrays
            instances = (
                self._load_instances() if load_instances else []
            )  # list of numpy arrays

        elif movies:
            # Otherwise, data is provided directly
            labels: List[np.ndarray] = kwargs.get("labels", [])
            instances: List[np.ndarray] = kwargs.get("instances", [])

            # Create empty attributes
            self.base_path = ""
            self.sample_ids = []

        else:
            raise ValueError(
                "Either movies or base_path and sample_ids must be provided."
            )

        # Store the dataset parameters
        self.window_size = params.data_duration
        self.stride: int = kwargs.get("stride", 0) or params.data_stride

        # Store the movies, labels and instances
        self.movies = [torch.from_numpy(movie.astype(np.int32)) for movie in movies]
        self.labels = [torch.from_numpy(label.astype(np.int8)) for label in labels]
        self.instances = [
            torch.from_numpy(instance.astype(np.int8)) for instance in instances
        ]
        self.gt_available = True if len(labels) == len(movies) else False

        # If instances are available, get the locations of the spark peaks
        if len(self.instances) > 0:
            self.spark_peaks = self._detect_spark_peaks()
        else:
            self.spark_peaks = []

        # Preprocess videos if necessary
        self._preprocess_videos()

        # Store original duration of all movies before padding
        self.original_durations = [movie.shape[0] for movie in self.movies]

        # Adjust videos shape so that it is suitable for the model
        self._adjust_videos_shape()

    ############################## Class methods ###############################

    def __len__(self) -> int:
        total_samples = 0
        for movie in self.movies:
            frames = movie.shape[0]
            # Calculate the number of samples for each movie
            samples_per_movie = (frames - self.window_size) // self.stride + 1
            total_samples += samples_per_movie
        return total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = self.__len__() + idx

        sample_dict = {}

        # Get the movie index and chunk index for the given idx
        movie_idx, chunk_idx = self._get_movie_and_chunk_indices(idx)
        sample_dict["movie_id"] = movie_idx

        # Store the original duration of the movie
        sample_dict["original_duration"] = self.original_durations[movie_idx]

        # Calculate the starting frame within the movie
        start_frame = chunk_idx * self.stride
        end_frame = start_frame + self.window_size

        # Extract the windowed data and labels
        sample_dict["data"] = self.movies[movie_idx][start_frame:end_frame]

        if self.gt_available:
            sample_dict["labels"] = self.labels[movie_idx][start_frame:end_frame]

        # Add the sample ID (string) to the item dictionary, if available
        if self.sample_ids:
            sample_dict["sample_id"] = self.sample_ids[movie_idx]

        return sample_dict

    def get_movies(self) -> Dict[int, np.ndarray]:
        """
        Returns the processed movies as a dictionary.

        Returns:
            dict: A dictionary containing the movies used as input to the model.
        """
        # Remove padding from the movies
        movies_numpy = {
            i: remove_padding(movie, self.original_durations[i]).numpy()
            for i, movie in enumerate(self.movies)
        }
        return movies_numpy

    def get_labels(self) -> Dict[int, np.ndarray]:
        """
        Returns the labels as a dictionary.

        Returns:
            dict: A dictionary containing the original labels used for training
            and testing.
        """
        # Remove padding from the labels
        labels_numpy = {
            i: remove_padding(label, self.original_durations[i]).numpy()
            for i, label in enumerate(self.labels)
        }
        return labels_numpy

    def get_instances(self) -> Dict[int, np.ndarray]:
        """
        Returns the instances as a dictionary.

        Returns:
            dict: A dictionary containing the original instances used for
            training and testing.
        """
        # Raise an error if instances are not available
        if not self.instances:
            raise ValueError("Instances not available for this dataset.")

        # Remove padding from the instances
        instances_numpy = {
            i: instance.numpy() for i, instance in enumerate(self.instances)
        }
        return instances_numpy

    ############################## Private methods #############################

    def _load_movies(self) -> List[np.ndarray]:
        # Load movie data for each sample ID
        movies = load_movies_ids(
            data_folder=self.base_path,
            ids=self.sample_ids,
            names_available=True,
            movie_names="video",
        )

        # Extract and return the movie values as a list
        movies = list(movies.values())

        return movies

    def _load_labels(self) -> List[np.ndarray]:
        # preprocess annotations if necessary
        if self.params.sparks_type == "raw":
            mask_names = "class_label"
        elif self.params.sparks_type == "peaks":
            mask_names = "class_label_peaks"
        elif self.params.sparks_type == "small":
            mask_names = "class_label_small_peaks"
        elif self.params.sparks_type == "dilated":
            mask_names = "class_label_dilated"
        else:
            raise NotImplementedError("Annotation type not supported yet.")

        labels = load_annotations_ids(
            data_folder=self.base_path, ids=self.sample_ids, mask_names=mask_names
        )

        if labels:
            # Extract and return the mask values as a list
            labels = list(labels.values())
        else:
            labels = []

        return labels

    def _load_instances(self) -> List[np.ndarray]:
        # Load single event instances for each sample ID
        instances = load_annotations_ids(
            data_folder=self.base_path,
            ids=self.sample_ids,
            mask_names="event_label",
        )

        if instances:
            # Extract and return the mask values as a list
            instances = list(instances.values())
        else:
            raise ValueError("Instances not available for this dataset.")

        return instances

    def _get_movie_and_chunk_indices(self, idx: int) -> Tuple[int, int]:
        """
        Given an index, returns the movie index and chunk index for the
        corresponding chunk in the dataset.

        Args:
            idx (int): The index of the chunk in the dataset.

        Returns:
            tuple: A tuple containing the movie index and chunk index for the
            corresponding chunk.
        """
        current_idx = 0  # Number of samples seen so far
        for movie_idx, movie in enumerate(self.movies):
            frames, _, _ = movie.shape
            samples_per_movie = (frames - self.window_size) // self.stride + 1
            if idx < current_idx + samples_per_movie:
                # If idx is smaller than the number of samples seen so
                # far plus the number of samples in the current movie,
                # then the sample we're looking for is in the current
                # movie.
                chunk_idx = idx - current_idx  # chunk idx in the movie
                return movie_idx, chunk_idx

            current_idx += samples_per_movie

        # If the index is out of range, raise an error
        raise IndexError(
            f"Index {idx} is out of range for dataset of length {len(self)}"
        )

    def _detect_spark_peaks(self, class_name: str = "sparks") -> List[np.ndarray]:
        # Detect the spark peaks in the instance mask of each movie
        # Remark: can be used for other classes as well
        spark_peaks = []
        for movie, labels, instances in zip(self.movies, self.labels, self.instances):
            spark_mask = np.where(
                labels == config.classes_dict[class_name], instances, 0
            )
            self.coords_true = detect_spark_peaks(
                movie=movie.numpy(),
                instances_mask=spark_mask,
                sigma=config.sparks_sigma_dataset,
                max_filter_size=10,
            )
            spark_peaks.append(self.coords_true)
        return spark_peaks

    def _preprocess_videos(self) -> None:
        """
        Preprocesses the videos in the dataset.
        """
        if self.params.remove_background == "average":
            self.movies = [self._remove_avg_background(movie) for movie in self.movies]

        if self.params.data_smoothing in ["2d", "3d"]:
            n_dims = int(self.params.data_smoothing[0])
            self.movies = [
                self._blur_movie(movie, n_dims=n_dims) for movie in self.movies
            ]

        if self.params.norm_video in ["movie", "abs_max", "std_dev"]:
            self.movies = [
                self._normalize(movie, norm_type=self.params.norm_video)
                for movie in self.movies
            ]

    def _remove_avg_background(self, movie: torch.Tensor) -> torch.Tensor:
        # Remove the average background from the video frames.
        avg = torch.mean(movie, dim=0)
        return movie - avg

    def _blur_movie(self, movie: torch.Tensor, n_dims: int) -> torch.Tensor:
        # Define the kernel size and sigma based on the number of dimensions
        kernel_size = (3,) * n_dims
        sigma = 1.0

        # Apply gaussian blur to the video
        gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return gaussian_blur(movie)

    def _normalize(self, movie: torch.Tensor, norm_type: str) -> torch.Tensor:
        # Normalize the video frames.
        if norm_type == "movie":
            # Normalize each movie separately using its own max and min
            movie = (movie - torch.min(movie)) / (torch.max(movie) - torch.min(movie))
        elif norm_type == "abs_max":
            # Normalize each movie separately using the absolute max of uint16
            absolute_max = np.iinfo(np.uint16).max  # 65535
            movie = (movie - torch.min(movie)) / (absolute_max - torch.min(movie))
        elif norm_type == "std_dev":
            # Normalize each movie separately using its own standard deviation
            movie = (movie - torch.mean(movie)) / torch.std(movie)
        else:
            raise ValueError(f"Invalid norm type: {norm_type}")
        return movie

    def _adjust_videos_shape(self) -> None:
        # Pad videos whose length does not match with chunks_duration and stride
        # params.
        self.movies = [self._pad_extremities_of_video(video) for video in self.movies]
        if self.gt_available:
            self.labels = [
                self._pad_extremities_of_video(mask, padding_value=config.ignore_index)
                for mask in self.labels
            ]

    def _pad_extremities_of_video(
        self, video: torch.Tensor, padding_value: int = 0
    ) -> torch.Tensor:
        """
        Pads videos whose length does not match with chunks_duration and step
        params.

        Args:
        - video (torch.Tensor): The video to pad.
        - padding_value (int): The value to use for padding. Default is 0.

        Returns:
        - The padded video.
        """
        video_duration = video.shape[0]
        chunk_duration = self.params.data_duration
        stride = self.params.data_stride

        # Check if length is shorter than data_duration
        if video_duration < chunk_duration:
            padding_length = chunk_duration - video_duration
        else:
            padding_length = stride * math.ceil(
                (video_duration - chunk_duration) / stride
            ) - (video_duration - chunk_duration)

        if padding_length > 0:
            video = F.pad(
                video,
                (
                    0,
                    0,
                    0,
                    0,
                    padding_length // 2,
                    padding_length // 2 + padding_length % 2,
                ),
                "constant",
                value=padding_value,
            )
            video_duration = video.shape[0]

        assert (
            (video_duration - chunk_duration) / stride
        ) % 1 == 0, "Padding at end of video is wrong."

        return video


class SparkDatasetTemporalReduction(SparkDataset):
    """
    A PyTorch Dataset class for spark detection with temporal reduction.

    This class is a subclass of the `SparkDataset` class and is specifically
    designed to work with deep learning models that use temporal reduction.
    It shrinks the annotation masks and instances to match the reduced temporal
    resolution of the model.

    Args:
        same as SparkDataset

    Raises:
        ValueError: If neither `movies` nor `base_path` and `sample_ids` are
        provided.
        AssertionError: If temporal reduction is not enabled in the parameters.

    Attributes:
        same as SparkDataset

    Methods:
        same as SparkDataset
    """

    def __init__(self, params: TrainingConfig, **kwargs: Any) -> None:
        # check that the temporal reduction is enabled in the parameters
        assert (
            params.temporal_reduction
        ), "Temporal reduction is not enabled in the parameters."

        # call the parent constructor
        super().__init__(params, **kwargs)

        # shrink the labels
        self.labels = [self._shrink_mask(mask) for mask in self.labels]

        # shrink the instances (not implemented yet!)
        if self.instances:
            # raise and error if instances are available
            raise NotImplementedError(
                "Instances are not supported for temporal reduction yet."
            )

    ############################## Private methods #############################

    def _shrink_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Shrink an annotation mask based on the number of channels.

        Args:
            mask (numpy.ndarray): Input annotation mask.

        Returns:
            numpy.ndarray: Shrinked annotation mask.
        """
        assert (
            mask.shape[0] % self.params.num_channels == 0
        ), "Duration of the mask is not a multiple of num_channels."

        # Get tensor of duration 'self.num_channels'
        sub_masks = np.split(mask, mask.shape[0] // self.params.num_channels)
        new_mask = []

        # For each subtensor get a single frame
        for sub_mask in sub_masks:
            new_frame = np.array(
                [
                    [
                        self._get_new_voxel_label(sub_mask[:, y, x])
                        for x in range(sub_mask.shape[2])
                    ]
                    for y in range(sub_mask.shape[1])
                ]
            )
            new_mask.append(new_frame)

        new_mask = np.stack(new_mask)
        return new_mask

    def _get_new_voxel_label(self, voxel_seq: np.ndarray) -> int:
        """
        Get the new voxel label based on the sequence of voxel values.

        Args:
            voxel_seq (numpy.ndarray): Sequence of voxel values
                (num_channels elements).

        Returns:
            int: New voxel label.
        """
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


class SparkDatasetResampled(SparkDataset):
    """
    Dataset class for resampled SR-calcium releases segmented dataset.

    This class extends the `SparkDataset` class and resamples the movies to a
    given frame rate. The original frame rate of the movies is obtained from
    their metadata. The resampled movies, labels, and instances are stored in
    memory.

    Args:
    - params (TrainingConfig): The training configuration.
    - movie_paths (List[str]): A list of paths to the movies (same order as the
        movies in the dataset). This allows to obtain the original frame rate of
        the movies from their metadata.
    - new_fps (int): The frame rate to resample the movies to.
    ... (same as SparkDataset)

    Raises:
        ValueError: If `movie_paths` or `new_fps` are not provided.
    """

    def __init__(
        self,
        params: TrainingConfig,
        **kwargs,
    ) -> None:
        # Verify that movie_paths and new_fps are provided
        if "movie_paths" not in kwargs:
            raise ValueError("movie_paths must be provided")
        if "new_fps" not in kwargs:
            raise ValueError("new_fps must be provided")

        self.movie_paths: List[str] = kwargs["movie_paths"]
        self.new_fps: int = kwargs["new_fps"]

        # Initialize the SparksDataset class
        super().__init__(params=params, **kwargs)

    ############################## Class methods ###############################

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        # Get item from the SparksDataset class and add the original frame rate
        item_dict = super().__getitem__(idx)
        item_dict["original_fps"] = self.original_fps[int(item_dict["movie_id"])]

        return item_dict

    ############################## Private methods #############################

    def _preprocess_videos(self) -> None:
        """
        Preprocesses the videos in the dataset.
        """
        # apply the same preprocessing as in the SparksDataset class
        super()._preprocess_videos()

        # Get the original frame rate of the movies
        self.original_fps = [
            self._get_fps(movie_path) for movie_path in self.movie_paths
        ]

        # Resample the movies to the desired frame rate
        self.movies = [
            self._resample_video(movie, movie_path)
            for movie, movie_path in zip(self.movies, self.movie_paths)
        ]

        # Resample the labels to the desired frame rate
        if self.labels:
            self.labels = [
                self._resample_video(mask, movie_path)
                for mask, movie_path in zip(self.labels, self.movie_paths)
            ]

        # Resample the instances to the desired frame rate
        if self.instances:
            self.instances = [
                self._resample_video(instance, movie_path)
                for instance, movie_path in zip(self.instances, self.movie_paths)
            ]

    ####################### Methods for video resampling #######################

    def _resample_video(self, movie: torch.Tensor, movie_path: str) -> torch.Tensor:
        # Resample the video to the desired frame rate
        return self._video_spline_interpolation(
            movie=movie, movie_path=movie_path, new_fps=self.new_fps
        )

    def _get_fps(self, movie_path: str) -> float:
        """
        Compute estimated video FPS value with respect to sampling time deltas.

        Args:
            movie_path (str): Path to the video.

        Returns:
            float: Estimated FPS value.
        """
        times = self._get_times(movie_path)
        deltas = np.diff(times)
        return float(1.0 / np.mean(deltas))

    def _get_times(self, movie_path: str) -> np.ndarray:
        """
        Get times at which video frames were sampled.

        Args:
            movie_path (str): Path to the video.

        Returns:
            numpy.ndarray: Array of frame times.
        """
        with Image.open(movie_path) as img:
            exif_data = img.getexif()
        description = exif_data[270][0].split("\r\n")
        description = [line.split("\t") for line in description]
        description = [
            [int(i) if i.isdigit() else i for i in line] for line in description
        ]
        description = [d for d in description if isinstance(d[0], int)]
        return np.array([float(line[1]) for line in description])

    def _video_spline_interpolation(
        self, movie: torch.Tensor, movie_path: str, new_fps: int
    ) -> torch.Tensor:
        """
        Interpolate video frames based on new sampling times (FPS).

        Args:
            movie (numpy.ndarray): Input video frames.
            movie_path (str): Path to the video.
            new_fps (int): Desired FPS for the output video.

        Returns:
            numpy.ndarray: Interpolated video frames.
        """
        frames_time = self._get_times(movie_path)
        f = interp1d(frames_time, movie, kind="linear", axis=0)
        assert len(frames_time) == movie.shape[0], (
            "In video_spline_interpolation the duration of the video "
            "is not equal to the number of frames"
        )
        frames_new = np.linspace(
            frames_time[0], frames_time[-1], int(frames_time[-1] * new_fps)
        )

        return f(frames_new)


class SparkDatasetLSTM(SparkDataset):
    """
    SparkDataset class for UNet-convLSTM model.

    The dataset is adapted in such a way that each chunk is a sequence of
    frames centered around the frame to be predicted.
    The label is the segmentation mask of the central frame.
    """

    def __init__(self, params: TrainingConfig, **kwargs: Dict[str, Any]) -> None:
        # step = 1 and ignore_frames = 0 because we need to have a prediction
        # for each frame.
        self.params.data_stride = 1
        self.params.ignore_frames_loss = 0
        super().__init__(params, **kwargs)

    ############################## Class methods ###############################

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        """
        As opposed to the SparkDataset class, here the label is just the
        middle frame of the chunk.
        """
        sample_dict = super().__getitem__(idx)

        if self.gt_available:
            # extract middle frame from label
            sample_dict["labels"] = sample_dict["labels"][
                self.params.data_duration // 2
            ]

        return sample_dict

    ############################## Private methods #############################

    def _pad_short_video(
        self, video: torch.Tensor, padding_value: int = 0
    ) -> torch.Tensor:
        """
        Instead of padding the video with zeros, pad it with the first
        and last frame of the video.
        """
        padding_length = self.params.data_duration - video.shape[0]
        if padding_length:
            video = F.pad(
                video,
                (
                    0,
                    0,
                    0,
                    0,
                    padding_length // 2,
                    padding_length // 2 + padding_length % 2,
                ),
                "replicate",
            )

            assert video.shape[0] == self.params.data_duration, "Padding is wrong"

            # logger.debug("Added padding to short video")

        return video

    def _pad_extremities_of_video(
        self, video: torch.Tensor, padding_value: int = 0
    ) -> torch.Tensor:
        """
        Pad duration/2 frames at the beginning and at the end of the video
        with the first and last frame of the video.
        """
        length = video.shape[0]

        # check that duration is odd
        assert self.params.data_duration % 2 == 1, "duration must be odd"

        pad = self.params.data_duration - 1

        # if video is int32, cast it to float32
        cast_to_float = video.dtype == torch.int32
        if cast_to_float:
            video = video.float()  # cast annotations to float32

        replicate = nn.ReplicationPad3d((0, 0, 0, 0, pad // 2, pad // 2))
        video = replicate(video[None, :])[0]

        if cast_to_float:
            video = video.int()  # cast annotations back to int32

        # check that duration of video is original duration + chunk duration - 1
        assert (
            video.shape[0] == length + self.params.data_duration - 1
        ), "padding at end of video is wrong"

        return video


class SparkDatasetInference(SparkDataset):
    """
    Create a dataset that contains only a single movie for inference.
    It requires either a single movie or a movie path to be provided.
    """

    def __init__(self, params: TrainingConfig, **kwargs) -> None:
        # Check that the arguments are suitable
        movie_path = kwargs.get("movie_path")
        movie = kwargs.get("movie")

        if movie is None and movie_path is None:
            raise ValueError("Either movie or movie_path must be provided.")
        if movie_path:
            # If a movie path is provided, load the movie from disk
            movies = [np.asarray(imageio.volread(movie_path))]
        else:
            movies = [movie]

        # Initialize the SparksDataset class
        super().__init__(
            params=params, movies=movies, labels=[], instances=[], **kwargs
        )
