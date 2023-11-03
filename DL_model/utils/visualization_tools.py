"""
Script with tools for data visualisation (e.g. plots and Napari).

Author: Prisca Dotti
Last modified: 02.11.2023
"""

import itertools
import logging
import math
import random
from ast import Tuple
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import vispy.color
from matplotlib import cm
from PIL import Image
from scipy import ndimage as ndi

from config import config

# NOTES FOR THE FUTURE:
# How to create a DataFrame with various parameters
# rows <- a list of lists with one value for each column
# cols <- a list with column names
# df = pd.DataFrame(rows, columns=cols)
# -> You can 'append' datasets with the same structure

# Other useful functions for data visualization:
# df_values = df[(df[col_name]==value) & ...]
# pd.pivot_table(df_values, values=..., columns=[col_names], index=[...])

__all__ = [
    "get_discrete_cmap",
    "get_labels_cmap",
    "set_edges_to_white",
    "get_annotations_contour",
    "paste_segmentation_on_video",
    "add_colored_segmentation_to_video",
    "ball",
    "color_ball",
    "l2_distance",
    "add_colored_locations_to_video",
    "add_colored_paired_sparks_to_video",
    "add_colored_classes_to_video",
    "add_colored_instances_to_video",
    "create_circular_mask",
    "create_signal_mask",
    "get_spark_signal",
    "get_spark_2d_signal",
]

logger = logging.getLogger(__name__)


#################################### Napari ####################################


def get_discrete_cmap(name: str = "gray", lut: int = 16) -> vispy.color.Colormap:
    """
    This function creates a discrete colormap from an original colormap with a
    specified number of discrete colors. It returns a Colormap instance that can
    be used for visualization in tools like Napari.

    Args:
        name (str): The name of the original colormap.
        lut (int): The number of discrete colors to obtain.

    Returns:
        vispy.color.Colormap: A discrete colormap instance suitable for use
        with Napari.
    """

    # Create the original colormap
    segmented_cmap = cm.get_cmap(name=name, lut=lut)

    # Get the colors
    colors = segmented_cmap(np.arange(0, lut))

    # Create a new discrete colormap
    cmap = vispy.color.Colormap(colors, interpolation="zero")

    return cmap


def get_labels_cmap() -> Dict[int, list]:
    """
    Get a color map (cmap) for visualizing labeled regions.

    Returns:
        dict: A dictionary mapping label IDs to RGBA color values.

    The function returns a cmap that assigns distinct colors to specific labels
    for visual representation. The color values are in RGBA format, with values
    ranging from 0.0 to 1.0.
    """
    no_color = [0, 0, 0, 0]
    green_spark = [178 / 255, 255 / 255, 102 / 255, 1]
    red_puff = [255 / 255, 102 / 255, 102 / 255, 1]
    purple_wave = [178 / 255, 102 / 255, 255 / 255, 1]
    grey_ignore = [224 / 255, 224 / 255, 224 / 255, 1]
    yellow_removed = [255 / 255, 255 / 255, 0 / 255, 1]
    black = [1, 1, 1, 1]

    labels_cmap = {
        0: no_color,
        1: green_spark,
        2: purple_wave,
        3: red_puff,
        4: grey_ignore,
        5: yellow_removed,
        6: black,
    }

    return labels_cmap


def set_edges_to_white(
    input_movie: np.ndarray, white: Union[int, Tuple, List], thickness: int = 0
) -> None:
    """
    Set the edges of an input movie to a specified color (called white for now).

    Args:
        input_movie (numpy.ndarray): Input movie as a NumPy array with shape
            (T, H, W) or (T, H, W, 3) or (T, H, W, 4) for RGB or RGBA movies,
            where T is the number of frames, H is the height of each frame, and
            W is the width of each frame.
        white (tuple or list): int, RGB or RGBA color value to set the edges to, e.g.,
            (255, 255, 255) for white.
        thickness (int): Thickness of the edge region to set to the white color.

    If thickness is 0, it sets the corners and edges of the input_movie to the
    pecified white color. If thickness is greater than 0, it sets a border of the
    specified thickness to the white color.
    """

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
        input_movie[-thickness - 1 :, :thickness, :] = white
        input_movie[:thickness, -thickness - 1 :, :] = white
        input_movie[-thickness - 1 :, -thickness - 1 :, :] = white

        input_movie[:, :thickness, :thickness] = white
        input_movie[:, -thickness - 1 :, :thickness] = white
        input_movie[:, :thickness, -thickness - 1 :] = white
        input_movie[:, -thickness - 1 :, -thickness - 1 :] = white

        input_movie[:thickness, :, :thickness] = white
        input_movie[-thickness - 1 :, :, :thickness] = white
        input_movie[:thickness, :, -thickness - 1 :] = white
        input_movie[-thickness - 1 :, :, -thickness - 1 :] = white


def get_annotations_contour(
    annotations: np.ndarray, contour_val: int = 2
) -> np.ndarray:
    """
    Compute the contour of annotation masks for Napari visualization.

    Args:
        annotations (numpy.ndarray): Input annotation mask with integer class labels.
        contour_val (int): Number of pixels to dilate the contour.

    Returns:
        numpy.ndarray: Contour mask with the same shape as input annotations.

    This function computes the contour of annotation masks for visualization in
    Napari. It dilates and erodes each class separately to obtain the contour and
    combines them into a single contour mask.
    """
    # Create a structuring element for dilation and erosion
    # (dilate only along x and y)
    struct = np.zeros((1, 1 + contour_val, 1 + contour_val))
    struct[0, 1, :] = 1
    struct[0, :, 1] = 1

    # Initialize the contour mask
    labels_contour = np.zeros_like(annotations)

    # Compute the contour for each class separately
    for class_nb in range(1, config.num_classes + 2):
        class_ys = annotations == class_nb
        class_dilated = ndi.binary_dilation(class_ys, structure=struct)
        class_eroded = ndi.binary_erosion(class_ys, structure=struct)
        class_contour = np.where(
            np.logical_not(class_eroded.astype(bool)), class_dilated, 0
        )
        labels_contour += class_nb * class_contour.astype(labels_contour.dtype)

    return labels_contour


#################### colored events on movie visualisation #####################


def paste_segmentation_on_video(
    video: List[Image.Image], colored_mask: List[Image.Image]
) -> None:
    """
    Paste colored segmentation masks on video frames.

    Args:
        video (list of PIL.Image): List of RGB video frames.
        colored_mask (list of PIL.Image): List of RGBA colored segmentation masks.

    This function pastes colored segmentation masks on each video frame using the
    provided alpha channel as a mask.
    """
    # Ensure that the lengths of video and colored_mask lists match
    if len(video) != len(colored_mask):
        raise ValueError(
            "The number of video frames must match the number of colored masks."
        )

    # Iterate through each frame and colored mask
    for frame, ann in zip(video, colored_mask):
        # Ensure that the colored mask has an alpha channel (RGBA format)
        if len(ann.split()) < 4:
            raise ValueError("Colored mask must have an alpha channel (RGBA format).")

        # Paste the colored mask on the video frame using the alpha channel as a mask
        frame.paste(ann, mask=ann.split()[3])


def add_colored_segmentation_to_video(
    segmentation: np.ndarray,
    video: List[Image.Image],
    color: Tuple[int, int, int],
    transparency: int = 50,
) -> List[Image.Image]:
    """
    This function adds a colored segmentation overlay to each frame in a video
    using the specified color and transparency.

    Args:
        segmentation (numpy.ndarray): Binary segmentation mask.
        video (list of PIL.Image): List of RGB video frames.
        color (tuple of int): RGB color tuple.
        transparency (int): Transparency level for the overlay (0-255).

    Returns:
        list of PIL.Image: List of RGB video frames with segmentation overlay.
    """
    # Ensure the color is in RGB format
    if len(color) != 3:
        raise ValueError(
            "Color must be a tuple of 3 RGB values (e.g., (255, 0, 0) for red)."
        )

    # Create a colored RGBA mask from the binary segmentation
    r, g, b = color
    colored_mask = np.stack(
        (
            r * segmentation,
            g * segmentation,
            b * segmentation,
            transparency * segmentation,
        ),
        axis=-1,
    )
    colored_mask = colored_mask.astype(np.uint8)

    # Convert the colored mask to a list of RGBA PIL images
    colored_mask = [Image.fromarray(frame).convert("RGBA") for frame in colored_mask]

    # Overlay the colored segmentation on each frame in the video
    paste_segmentation_on_video(video, colored_mask)

    return video


def ball(center: Tuple[int, ...], radius: int) -> List[Tuple[int, ...]]:
    """
    Generate a list of coordinates within an N-dimensional ball shape centered
    at the specified center.

    Args:
        center (tuple): Center coordinates (N values).
        radius (int): Radius of the hyperball.

    Returns:
        list of tuples: List of coordinates within the hyperball.
    """
    dimensions = len(center)
    ranges = [
        np.linspace(c - radius, c + radius, 2 * radius + 1, dtype=int) for c in center
    ]
    indices = list(itertools.product(*ranges))
    ball_indices = [pt for pt in indices if l2_distance(center, pt) <= radius]

    return ball_indices


def color_ball(
    mask: np.ndarray,
    center: Tuple[int, ...],
    radius: int,
    color: Tuple[int, ...],
    transparency: float = 0.5,  # original was 50
) -> np.ndarray:
    """
    Color a region within an N-dimensional hyperball shape in an RGBA mask with the
    specified color and transparency.

    Args:
        mask (numpy.ndarray): RGBA mask to be colored.
        center (tuple): Center coordinates (N values) of the hyperball.
        radius (int): Radius of the hyperball.
        color (tuple): RGB color tuple (e.g., (255, 0, 0) for red).
        transparency (int): Transparency level for the colored region (0-255).
    """
    indices = ball(center, radius)

    # Get mask dimensions (N + 1 dimensions for RGBA)
    shape = mask.shape

    for index in indices:
        valid_index = True
        for i, dim in enumerate(index):
            if not (0 <= dim < shape[i]):
                valid_index = False
                break

        if valid_index:
            mask[tuple(index)] = [*color, transparency]

    return mask


def l2_distance(point1: Sequence[int], point2: Sequence[int]) -> float:
    """
    Calculate the L2 Euclidean distance between two points in N-dimensional space.

    Args:
        point1 (tuple or list): Coordinates of the first point (N values).
        point2 (tuple or list): Coordinates of the second point (N values).

    Returns:
        float: L2 Euclidean distance between the two points.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensionality")

    squared_distance = sum((x - y) ** 2 for x, y in zip(point1, point2))
    return math.sqrt(squared_distance)


def add_colored_locations_to_video(
    locations: List[Tuple[int, int, int]],
    video: List[Image.Image],
    color: Tuple[int, int, int],
    transparency: float = 0.5,  # original was 50
    radius: int = 4,
) -> List[Image.Image]:
    """
    This function adds colored circular markers to specified (t, x, y) locations in
    a video using the specified color, transparency, and radius (used for sparks).

    Args:
        locations (list of tuple): List of (t, x, y) coordinates.
        video (list of PIL.Image): List of RGB video frames (list of PIL images).
        color (tuple of int): RGB color tuple.
        transparency (int): Transparency level for the markers (0-255).
        radius (int): Radius of the circular markers.

    Returns:
        list of PIL.Image: List of RGB video frames with colored markers.
    """
    # Ensure the color is in RGB format
    if len(color) != 3:
        raise ValueError(
            "Color must be a tuple of 3 RGB values (e.g., (255, 0, 0) for red)."
        )

    # Create an RGBA mask for markers
    mask_shape = (len(video), video[0].size[1], video[0].size[0], 4)
    colored_mask = np.zeros(mask_shape, dtype=np.uint8)

    # Draw circular markers at specified locations
    for pt in locations:
        colored_mask = color_ball(colored_mask, pt, radius, color, transparency)
    colored_mask = [Image.fromarray(frame).convert("RGBA") for frame in colored_mask]

    paste_segmentation_on_video(video, colored_mask)
    return video


def add_colored_paired_sparks_to_video(
    movie: np.ndarray,
    paired_true: List[Tuple[int, int, int]],
    paired_preds: List[Tuple[int, int, int]],
    fp: List[Tuple[int, int, int]],
    fn: List[Tuple[int, int, int]],
    transparency: int = 50,
) -> List[np.ndarray]:
    """
    This function takes an original video, along with lists of paired true, paired
    predicted, false positive, and false negative spark coordinates. It adds color-coded
    annotations to the video frames and returns the resulting colored video.

    Args:
        movie (numpy.ndarray): Original video frames.
        paired_true (list): List of paired true spark coordinates.
        paired_preds (list): List of paired predicted spark coordinates.
        fp (list): List of false positive spark coordinates.
        fn (list): List of false negative spark coordinates.
        transparency (int): Transparency level for colored annotations (0-255).

    Returns:
        list of numpy.ndarray: Colored video frames.
    """
    # Normalize the sample movie and convert to RGB
    sample_video = np.copy(movie)
    sample_video = 255 * (sample_video / sample_video.max())
    rgb_video = [Image.fromarray(frame).convert("RGB") for frame in sample_video]

    # Color-coded annotations
    annotated_video = add_colored_locations_to_video(
        locations=paired_true,
        video=rgb_video,
        color=(0, 255, 0),
        transparency=0.8 * transparency,
    )  # green
    annotated_video = add_colored_locations_to_video(
        locations=paired_preds,
        video=annotated_video,
        color=(0, 255, 200),
        transparency=0.8 * transparency,
    )  # cyan
    annotated_video = add_colored_locations_to_video(
        locations=fp,
        video=annotated_video,
        color=(255, 255, 0),
        transparency=transparency,
    )  # yellow
    annotated_video = add_colored_locations_to_video(
        locations=fn,
        video=annotated_video,
        color=(255, 0, 0),
        transparency=transparency,
    )  # red

    annotated_video = [np.array(frame) for frame in annotated_video]
    return annotated_video


def add_colored_classes_to_video(
    movie: np.ndarray,
    classes_mask: np.ndarray,
    transparency: int = 50,
    ignore_frames: int = 0,
    white_bg: bool = False,
    label_mask: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    This function takes an original video, a segmentation mask with class
    labels, and optional labelled segmentation mask. It adds color-coded class
    annotations to the video frames and returns the resulting colored video.
    If a label mask is provided, add their contours to the resulting video.

    Args:
        movie (numpy.ndarray): Original video frames.
        classes_mask (numpy.ndarray): Segmentation mask with class labels.
        transparency (int): Transparency level for colored annotations (0-255).
        ignore_frames (int): Number of frames to ignore from the beginning and end.
        white_bg (bool): Whether the background should be white.
        label_mask (numpy.ndarray): Optional label mask for adding label contours.

    Returns:
        list of numpy.ndarray: Colored video frames.
    """
    color_movie = np.copy(movie)
    # Normalize sample movie and create a color dictionary
    if white_bg:
        classes_dict = {  # Dataset parameters
            "sparks": {"nb": 1, "color": [0, 255, 0]},  # Green
            "puffs": {"nb": 3, "color": [220, 20, 60]},  # Red
            "waves": {"nb": 2, "color": [138, 43, 226]},  # Purple
            "ignore": {"nb": 4, "color": [80, 80, 80]},  # Gray
        }
        color_movie.fill(255)
        transparency = 1000  # Remove transparency if the background is white
    else:
        classes_dict = {  # Dataset parameters
            "sparks": {"nb": 1, "color": [178, 255, 102]},  # Green
            "puffs": {"nb": 3, "color": [255, 102, 102]},  # Red
            "waves": {"nb": 2, "color": [178, 102, 255]},  # Purple
            "ignore": {"nb": 4, "color": [224, 224, 224]},  # Gray
        }
        color_movie = 255 * (color_movie / color_movie.max())

    # Convert video to RGB
    color_movie = [Image.fromarray(frame).convert("RGB") for frame in movie]

    # Add colored segmentation to movie
    for class_info in classes_dict.values():
        class_nb = class_info["nb"]
        color = class_info["color"]

        if class_nb in classes_mask:
            binary_preds = classes_mask == class_nb
            color_movie = add_colored_segmentation_to_video(
                segmentation=binary_preds,
                video=color_movie,
                color=color,
                transparency=transparency,
            )

        # Add label contours if label_mask is provided
        if label_mask is not None:
            label_contours = get_annotations_contour(
                annotations=label_mask, contour_val=2
            )
            if class_nb in label_mask:
                binary_labels = label_contours == class_nb
                color_movie = add_colored_segmentation_to_video(
                    segmentation=binary_labels,
                    video=color_movie,
                    color=color,
                    transparency=1000,
                )
    # Convert to numpy array and remove first and last frames
    if ignore_frames > 0:
        color_movie = [
            np.array(frame) for frame in color_movie[ignore_frames:-ignore_frames]
        ]
    else:
        color_movie = [np.array(frame) for frame in color_movie]

    return color_movie


def add_colored_instances_to_video(
    movie: np.ndarray,
    instances_mask: np.ndarray,
    transparency: int = 50,
    ignore_frames: int = 0,
    white_bg: bool = False,
) -> List[np.ndarray]:
    """
    This function takes an original video and a segmentation mask with event
    instances. It assigns a random color to each event instance and adds the
    colored instances to the video frames, returning the resulting colored video.

    Args:
        movie (numpy.ndarray): Original video frames.
        instances_mask (numpy.ndarray): Segmentation mask with event instances.
        transparency (int): Transparency level for colored instances (0-255).
        ignore_frames (int): Number of frames to ignore from the beginning and end.
        white_bg (bool): Whether the background should be white.

    Returns:
        list of numpy.ndarray: Colored video frames.
    """
    color_movie = np.copy(movie)

    # Normalize sample movie
    if white_bg:
        color_movie.fill(255)
        transparency = 1000  # Remove transparency if the background is white
    else:
        color_movie = 255 * (color_movie / color_movie.max())

    # Convert video to RGB
    color_movie = [Image.fromarray(frame).convert("RGB") for frame in color_movie]

    # Add colored segmentation to movie
    for event_id in range(1, instances_mask.max() + 1):
        event_mask = instances_mask == event_id

        # Create a random color for each event
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        color_movie = add_colored_segmentation_to_video(
            segmentation=event_mask,
            video=color_movie,
            color=color,
            transparency=transparency,
        )

    # Convert to numpy array and remove first and last frames
    if ignore_frames > 0:
        color_movie = [
            np.array(frame) for frame in color_movie[ignore_frames:-ignore_frames]
        ]
    else:
        color_movie = [np.array(frame) for frame in color_movie]

    return color_movie


############################## signal extraction ###############################


def create_circular_mask(
    h: int, w: int, center: Tuple[int, int], radius: int
) -> np.ndarray:
    """
    Create a circular mask of given radius around the specified center.

    Args:
    - h (int): Image height.
    - w (int): Image width.
    - center (tuple): Center of the circular mask (x_c, y_c).
    - radius (int): Radius of the circular mask.

    Returns:
    - mask (numpy.ndarray): Circular mask with True values inside the circle.
    """
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def create_signal_mask(
    t: int, h: int, w: int, start: int, stop: int, center: Tuple[int, int], radius: int
) -> np.ndarray:
    """
    Create a video mask with a circular region of interest (ROI) specified by
    center and radius.

    Args:
    - t (int): Video duration.
    - h (int): Video height.
    - w (int): Video width.
    - start (int): First frame containing ROI.
    - stop (int): First frame not containing ROI.
    - center (tuple): Center of the circular mask (x_c, y_c).
    - radius (int): Radius of the circular mask.

    Returns:
    - video_mask (numpy.ndarray): Video mask with a circular region of interest.
    """
    start = max(0, start)
    stop = min(t, stop)

    frame_mask = create_circular_mask(h, w, center, radius)

    video_mask = np.zeros((t, h, w), dtype=bool)
    video_mask[start:stop] = frame_mask

    return video_mask


def get_spark_signal(
    video: np.ndarray,
    sparks_labelled: np.ndarray,
    center: Tuple[int, int, int],
    radius: int,
    context_duration: int,
    return_info: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Tuple[int, int], int, int],
]:
    """
    Compute the signal of a spark event given its center and radius.

    Args:
    - video (numpy.ndarray): The original video sample.
    - sparks_labelled (numpy.ndarray): A mask containing the segmentation of
        spark events.
    - center (tuple): The center of the selected event to analyze (t, y, x).
    - radius (int): The radius of the region of interest around the center.
    - context_duration (int): The number of frames included in the analysis before
        and after the event.
    - return_info (bool, optional): Whether to return additional information.

    Returns:
    - frames (numpy.ndarray): Array of frame indices.
    - signal (numpy.ndarray): Averaged signal within the region of interest.
    - event_location (tuple): The location of the event (y, x).
    - event_start (int): The start frame of the event.
    - event_stop (int): The stop frame of the event.
    """
    t, y, x = center
    event_idx = sparks_labelled[t, y, x]  # event_idx = 1,2,...,n_sparks

    assert (
        event_idx != 0
    ), "Given center does not correspond to any event in the labeled mask"

    loc = ndi.measurements.find_objects(sparks_labelled)[event_idx - 1]

    # assert loc[0].start <= t and loc[0].stop > t, "Something is wrong"

    # Get mask representing spark's location (with radius and context)
    start = loc[0].start - context_duration
    stop = loc[0].stop + context_duration

    start = max(0, start)
    stop = min(video.shape[0], stop)

    t, h, w = sparks_labelled.shape
    signal_mask = create_signal_mask(
        t=t, h=h, w=w, start=start, stop=stop, center=(x, y), radius=radius
    )

    frames = np.arange(start, stop)
    signal = np.average(video[start:stop], axis=(1, 2), weights=signal_mask[start:stop])

    if return_info:
        return frames, signal, (y, x), loc[0].start, loc[0].stop

    return frames, signal


def get_spark_2d_signal(
    video: np.ndarray,
    slices: Tuple[slice, slice, slice],
    coords: Tuple[int, int, int],
    spatial_context: int,
    sigma: Optional[int] = None,
    return_info: bool = False,
) -> Union[Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray], np.ndarray,]:
    """
    Compute the 2D signal of a spark event given its slices and coordinates.

    Args:
    - video (numpy.ndarray): The original video.
    - slices (tuple): Slices in the 3 dimensions of a given spark (t_slice, y_slice, x_slice).
    - coords (tuple): Center of the given spark (t, y, x).
    - spatial_context (int): How much ROI corresponding to the spark gets extended.
    - sigma (int, optional): Standard deviation for Gaussian filtering.
    - return_info (bool, optional): Whether to return additional information.

    Returns:
    - signal_2d (numpy.ndarray): The 2D signal of the spark event.
    - Other information if return_info is True.
    """
    t, y, x = coords
    t_slice, y_slice, x_slice = slices

    y_start = max(0, y_slice.start - spatial_context)
    y_end = min(video.shape[1], y_slice.stop + spatial_context)

    x_start = max(0, x_slice.start - spatial_context)
    x_end = min(video.shape[2], x_slice.stop + spatial_context)

    signal_2d = video[t, y_start:y_end, x_start:x_end]

    # Average over 3 frames
    # signal_2d = video[t-1:t+2, y_start:y_end, x_start:x_end]
    # signal_2d = np.average(signal_2d, axis=0)

    # Smooth signal if sigma is provided
    if sigma is not None:
        signal_2d = ndi.gaussian_filter(signal_2d, sigma)

    if return_info:
        y_frames = np.arange(y_start, y_end)
        x_frames = np.arange(x_start, x_end)

        return t, y, x, y_frames, x_frames, signal_2d

    return signal_2d
