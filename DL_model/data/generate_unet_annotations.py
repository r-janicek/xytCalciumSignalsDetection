"""
Use this script to create .tif masks that can be used during U-Net training as 
sample annotations. This script generates annotations with sparks of type "peaks",
"small", and "dilated".
Not necessary to run this script if you want to use the raw annotations.

Input: .tif masks containing ROIs (Regions of Interest) of events with values
from 1 to 4.

Output: 
If sparks type is "peaks" or "small", .tif masks where puffs (3), waves (2), and
areas to be ignored (4) remain unchanged, but the sparks (1) are indicated by their
centers. 
If sparks type is "dilated", all events get an ignored region of different size.

Updates:
- 18.01.2021: Generated new annotations where the ignore_region for sparks
              is much larger (1 --> 3).
- 31.08.2021: Created annotations with ignore_region == 1 for the new videos
              [12-14; 18-20; 29-30; 38-46].
- 19.10.2021: Copied the .py file to a new folder to process the new annotations
              for training using Miguel's smoothed videos instead of the original
              videos.
- 07.02.2022: Generated annotations for corrected videos [13, 22, 34-35] and
              videos added to the training [30, 32].
- 23.02.2022: Fixed a bug (imported videos with integer values) and improved
              non-maxima suppression. The process is repeated using the original
              videos.
- 22.03.2022: Regenerated annotations with sigma = 2 for safety.
- 28.03.2022: Regenerated annotations using the correct version of non-maxima
              suppression (the Gaussian filter is now applied to the dilated
              version of the annotation mask).
- 21.10.2022: Merged this code with the one for creating U-Net annotations with
              peaks. From now on, use a single script to generate U-Net annotations.
- 26.10.2022: Included an option to create masks with "small sparks" (using the
              percentile).

Remarks:
- 01.03.2022: This code is now adapted for PC232.

Author: Prisca Dotti
Last modified: 11.10.2023
"""

import argparse
import os

import imageio
import numpy as np

from data.data_processing_tools import (
    annotate_sparks_with_peaks,
    apply_ignore_regions_to_events,
    reduce_sparks_size,
)
from utils.in_out_tools import load_annotations_ids, load_movies_ids


def main():
    parser = argparse.ArgumentParser(description="Generate u-net segmentations")
    parser.add_argument(
        "sparks",
        choices=["dilated", "peaks", "small"],
        help="Type of sparks that will result in the annotations (dilated, peaks, small)",
    )
    parser.add_argument(
        "--sample_ids",
        "-ids",
        nargs="+",
        default=[],
        help="Select sample IDs for which .tif segmentation will be generated",
    )
    args = parser.parse_args()

    print(f"Creating annotations with sparks of type '{args.sparks}'")

    # If no movie ID is provided, process all dataset samples
    if not args.sample_ids:
        args.sample_ids = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "27",
            "28",
            "29",
            "30",
            "32",
            "33",
            "34",
            "35",
            "36",
            "38",
            "39",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46",
        ]

    print(f"Processing samples {args.sample_ids}")

    raw_data_directory = os.path.join("..", "data", "raw_data_and_processing")
    old_mask_folder = os.path.join(
        raw_data_directory, "manual_corr_separated_event_masks", "final_masks"
    )

    # Select if using smooth movies as base for sparks detection
    use_smooth_movies = True

    if use_smooth_movies:
        video_folder = os.path.join(raw_data_directory, "smoothed_movies")
    else:
        video_folder = os.path.join(raw_data_directory, "original_movies")

    out_folder = os.path.join(raw_data_directory, f"unet_masks_{args.sparks}_sparks")
    os.makedirs(out_folder, exist_ok=True)

    # Compute new annotation masks
    for id in args.sample_ids:
        print(f"Processing mask {id}...")

        old_mask = load_annotations_ids(
            data_folder=old_mask_folder, ids=[id], mask_names="class_label"
        )[id]
        print(f"\tOld values: {np.unique(old_mask)}")

        if args.sparks == "peaks":
            if use_smooth_movies:
                video = load_movies_ids(
                    data_folder=video_folder,
                    ids=[id],
                    names_available=True,
                    movie_names="smoothed_video",
                )[id]
            else:
                video = load_movies_ids(data_folder=video_folder, ids=[id])[id]
            print(f"\tVideo shape: {video.shape}")

            # parameters for peaks annotation
            radius_event, radius_ignore = 3, 1

            mask = annotate_sparks_with_peaks(
                video=video,
                labels_mask=old_mask,
                peak_radius=radius_event,
                ignore_radius=radius_ignore,
            )[1]

            out_path = os.path.join(out_folder, f"{id}_class_label_peaks.tif")

        elif args.sparks == "dilated":
            # Values for sparks, waves, puffs, respectively
            radius_ignore_list, erosion_list = [1, 3, 2], [0, 1, 1]

            mask = apply_ignore_regions_to_events(
                mask=old_mask,
                ignore_radii=radius_ignore_list,
                apply_erosion=erosion_list,
            )

            out_path = os.path.join(out_folder, f"{id}_class_label_dilated.tif")

        elif args.sparks == "small":
            if use_smooth_movies:
                video = load_movies_ids(
                    data_folder=video_folder,
                    ids=[id],
                    names_available=True,
                    movie_names="smoothed_video",
                )[id]
            else:
                video = load_movies_ids(data_folder=video_folder, ids=[id])[id]
            print(f"\tVideo shape {video.shape}")

            events_mask = load_annotations_ids(
                data_folder=old_mask_folder, ids=[id], mask_names="event_label"
            )[id]

            k = 75  # Choose value k of percentile
            sigma = 2  # Used for smoothing original

            mask = reduce_sparks_size(
                movie=video,
                class_mask=old_mask,
                event_mask=events_mask,
                sigma=sigma,
                k=k,
            )

            out_path = os.path.join(out_folder, f"{id}_class_label_small_peaks.tif")
        else:
            raise ValueError(f"Unknown sparks type: {args.sparks}")

        print(f"\tNew values: {np.unique(mask)}")
        imageio.volwrite(out_path, np.uint8(mask))


if __name__ == "__main__":
    main()
