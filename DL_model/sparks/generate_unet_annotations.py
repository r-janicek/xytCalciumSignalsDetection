'''
21.01.2021

Usare questo script per creare delle masks .tif che possono essere usate
durante il training della u-net come annotazioni dei samples.

Input:  masks .tif contenenti le ROIs degli eventi con valori da 1 a 4

Output: masks .tif dove puffs, waves e zone da ignorare rimangono invariati, ma
        gli sparks sono indicati dal loro centro

UPDATES:
18.01.2021  Generati nuove annotations dove la ignore_region per gli sparks è
            molto più grande (1 --> 3)
31.08.2021  Generate annotazioni con ignore_region == 1 per i nuovi video
            [12-14; 18-20; 29-30; 38-46]
19.10.2021  Copiato .py file in una nuova cartella per processare le nuove
            annotazioni per il training usando gli smoothed video di Miguel
            invece dei video originali
07.02.2022  Generato annotazioni per video corretti [13,22,34-35] e video
            aggiunti al training [30,32]
23.02.2022  Corretto bug (video importato con valori interi) e perfezionato
            nonmaxima_suppression. Procedimento di nuovo utilizzando i video
            originali.
22.03.2022  Generato nuovamente annotazioni con sigma = 2 per sicurezza.
28.03.2022  Generato nuovamente annotazioni usando la versione corretta di
            nonmaxima_suppression (il gaussian filter è ora applicato alla
            versione dilated dell'annotation mask).
21.10.2022  Merged questo codice con quello per creare le unet annotations con i
            peaks: d'ora in poi usare un solo script per generare le annotazioni
            della UNet.
26.10.2022  Includo opzione per creare masks con "small sparks" (usando il
            percentile).

REMARKS:
01.03.2022  Questo codice ora è adattato a PC232.

'''

import os
import argparse
import imageio
import numpy as np

from data_processing_tools import (get_new_mask,
                                   get_new_mask_raw_sparks,
                                   reduce_sparks_size)
from in_out_tools import load_movies_ids, load_annotations_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate u-net segmentations")

    parser.add_argument("sparks", choices=['raw','peaks','small'],
        help="type of sparks that will result in the annotations (raw/peaks)")

    parser.add_argument("--sample_ids", "-ids", nargs='+', default=[],
        help="select sample ids for which .tif segmentation will be generated")

    args = parser.parse_args()
    print(f"Creating annotations with sparks of type <<{args.sparks}>>")

    # if no movie ID is provided, process all dataset samples
    if not args.sample_ids:
    # list of all movie IDs
        args.sample_ids = ["01","02","03","04","05","06","07","08","09",
                           "10","11","12","13","14","15","16","17","18","19",
                           "20","21","22","23","24","25","27","28","29",
                           "30","32","33","34","35","36","38","39",
                           "40","41","42","43","44","45","46"
                           ]

    print(f"Processing samples {args.sample_ids}")

    raw_data_directory = os.path.join("..", "data", "raw_data_and_processing")
    old_mask_folder = os.path.join(raw_data_directory,
                                   "manual_corr_separated_event_masks",
                                   "final_masks")

    # select if using smooth movies as base for sparks detection
    use_smooth_movies = True

    if use_smooth_movies:
        video_folder = os.path.join(raw_data_directory,"smoothed_movies")
    else:
        video_folder = os.path.join(raw_data_directory,"original_movies")


    # get params
    if args.sparks == 'peaks':
        out_folder = os.path.join(raw_data_directory,"unet_masks")
        os.makedirs(out_folder, exist_ok=True)

        # events paramenters
        radius_event = 3
        radius_ignore = 1
        #radius_ignore = 3
        ignore_index = 4

    elif args.sparks == 'raw':
        out_folder = os.path.join(raw_data_directory,"unet_masks_raw_sparks")
        os.makedirs(out_folder, exist_ok=True)

        # events paramenters
        radius_ignore_sparks = 1
        radius_ignore_puffs = 2
        radius_ignore_waves = 3
        ignore_index = 4

    elif args.sparks == 'small':
        out_folder = os.path.join(raw_data_directory,"unet_masks_small_sparks")
        os.makedirs(out_folder, exist_ok=True)

        k = 75 # choose value k of percentile
        sigma = 2 # used for smoothing original


    # compute new annotation masks
    for id in args.sample_ids:
        print("Processing mask "+id+"...")

        old_mask = load_annotations_ids(data_folder=old_mask_folder,
                                        ids=[id], mask_names='class_label')[id]
        print("\tOld values:", np.unique(old_mask))

        if args.sparks == 'peaks':
            if use_smooth_movies:
                video = load_movies_ids(data_folder=video_folder,
                                        ids=[id], names_available=True,
                                        movie_names='smoothed_video')[id]
            else:
                video = load_movies_ids(data_folder=video_folder,
                                        ids=[id])[id]
            print("\tVideo shape", video.shape)

            mask = get_new_mask(video=video, mask=old_mask,
                                radius_event=radius_event,
                                radius_ignore=radius_ignore,
                                ignore_index=ignore_index,
                                sigma=2)

            out_path = os.path.join(out_folder,
                                    id+"_class_label.tif")

        elif args.sparks == 'raw':
            mask = get_new_mask_raw_sparks(mask=old_mask,
                                radius_ignore_sparks=radius_ignore_sparks,
                                radius_ignore_puffs=radius_ignore_puffs,
                                radius_ignore_waves=radius_ignore_waves,
                                ignore_index=ignore_index
                                )

            out_path = os.path.join(out_folder,
                                    id+"_class_label_raw_sparks.tif")


        elif args.sparks == 'small':
            if use_smooth_movies:
                video = load_movies_ids(data_folder=video_folder,
                                        ids=[id], names_available=True,
                                        movie_names='smoothed_video')[id]
            else:
                video = load_movies_ids(data_folder=video_folder,
                                        ids=[id])[id]
            print("\tVideo shape", video.shape)

            events_mask = load_annotations_ids(data_folder=old_mask_folder,
                                               ids=[id],
                                               mask_names='event_label')[id]

            mask = reduce_sparks_size(movie=video,
                                      class_mask=old_mask,
                                      event_mask=events_mask,
                                      sigma=sigma)

            out_path = os.path.join(out_folder,
                                    id+"_class_label_small_sparks.tif")

        print("\tNew values:", np.unique(mask))
        imageio.volwrite(out_path, np.uint8(mask))
