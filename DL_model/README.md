# Detection and Classification of Local Ca²⁺ Release Events in Cardiomyocytes Using 3D-UNet Neural Network

This repository contains the code and resources related to the detection and classification of local Ca²⁺ release events in cardiomyocytes using a 3D-UNet neural network. Below is an overview of the directory organization:

## Directory Structure

- **config_files/**: Contains .ini files where parameters specific to each training can be specified. This allows for resuming of training or performing inference on specific models.

- **data/**:
  - **sparks_dataset/**: Contains dataset samples as .tif files. Input movies are named "NN_video.tif," class segmentation labels are named "NN_class_label.tif," and masks with event instances are named "NN_event_label.tif" (NN is the sample ID).
  - **data_processing_tools.py**: Includes methods for processing dataset samples and annotations.
  - **datasets.py**: The dataset class and its variants are defined here.
  - **generate_unet_annotations.py**: Use this script to process raw movies and labels, saving variations of the dataset suitable for training (e.g., with Ca²⁺ sparks only annotated by their peaks).

- **evaluation/**:
  - **script1_output/**, **script2_output/**, ...: Output of various scripts, each script has a corresponding directory.
  - **metric_tools.py**: Contains methods for computing metrics on UNet's outputs.

- **models/**:
  - **saved_models/**: Stores model architectures and optimizer parameters.
  - **nnUNet/**: This directory contains the implementation of nnUNet, which was used for experimental purposes but not in the final project.
  - **UNet/**: The U-Net architecture used in the final implementation.
  - **unetOpenAI/**: OpenAI's UNet implementation used for experiments but not included in the final project.
  - **architectures.py**: Contains variations of U-Net implementations used for experiments.
  - **new_unet.py**: An alternative implementation of U-Net explored during experiments.
  - **saved_models.zip.001**, **saved_models.zip.002**: Unzip and combine these two files to get the trained final model.

- **notebooks/**:
  - **inference (interactive).ipynb**: Interactive version of the inference script, mainly used for debugging.
  - **matlab inference.ipynb**: A script showing how to use the method used for the Matlab GUI to run a sample in a saved U-Net model, either using a path or a numpy array as input.
  - **plot and analyze detected events.ipynb**: Used for plotting and analyzing detected events.
  - **training (interactive).ipynb**: Interactive version of the training script, mainly used for debugging.
  - **save processed movies on disk.ipynb**: Notebook to save on disk annotated videos with colored segmentation masks or instances on top of them. It also allows to save a stacked version of the results with descriptive text on top of each movie.

- **raw_notebooks/**: This directory contains notebooks used for experiments, figure generation, and similar purposes. Scripts in this directory may not be fully cleaned up.

- **requirements/**: Specifications (.txt and .yml) of libraries necessary to run the code.

- **utils/**:
  - **LovaszSoftmax/**: Implementation of the Lovasz-Softmax loss.
  - **custom_losses.py**: Implementation of other loss functions used for experimental purposes.
  - **in_out_tools.py**: Utilities for loading and saving files to/from disk.
  - **training_inference_tools.py**: Tools used for training the U-Net and performing various approaches to inference.
  - **training_script_utils.py**: Functions that wrap up the code of the training and inference scripts for improved readability.
  - **visualization_tools.py**: Methods for printing, plotting, and visualizing data.

- **config.py**: Contains two classes - `ProjectConfig` for constants in the project (e.g., event types, parameters, directories) and `TrainingConfig` for training-specific parameters loaded from a config.ini file.

- **run_inference.py**: Use this script to run inference using a saved model.

- **run_training.py**: Use this script to train a model (requires a config.ini file as a terminal argument).

## Getting Started

Before using this code, make sure you have all the necessary libraries and dependencies installed as specified in the **requirements/** directory.

For training a model, use **run_training.py** with a configuration file. For inference with a saved model, use **run_inference.py**.

- `config.py` is the file where all the important parameters of the considered dataset have to be specified
- `config_files/[...].ini` are the files specific to each training (so with different hyperparameters, loss function, number of training epochs, etc.) --> this file has to be either specified when running the training from the terminal (using `run_training.py`, example: ...TODO... ), or hardcoded in the interactive notebook `training (interactive).ipynb` (at line ...TODO...).


