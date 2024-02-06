# Graphical user interface implementing deep learning model to detect calcium events in fluorescent xyt image series.
Please refer to paper:   for more information.
## Usage
Graphical user interface was created using Matlab App Designer as standalone application both for MacOs and Windows.
It contains menu with items: open file and settings.
In settings there are possibilities to choose filters which are applied to loaded image series in sequence, default pixel size and parameters of deep learning model ((names and codes of classes), path to conda environment used during detection).

Main parts are two tabs: 1. detection of calcium events and 2. events browser
typical workflow
- load image series (TIFF) using menu: open file
- detect cell mask, manualy correct if needed
- detect calcium events
- in calcium events tab browse through all detected events, and manually correct fits if necessary (describe how)
- save results (excel file)

