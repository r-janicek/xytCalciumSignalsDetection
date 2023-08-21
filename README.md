# Graphical user interface implementing deep learning model to detect calcium events in fluorescent xyt image series.

menu: 
    - settings (parameters of analysis)
    - choose filters applied to image series after loading
    - set parameters of deep learning model (names and codes of classes), path to conda environment used during detection
    - set default pixel sizes
    
typical workflow
- load image series (TIFF) using menu: open file
- detect cell mask, correct if needed
- detect calcium events
- in calcium events tab browse through all detected events, and manually correct fits if necessary (describe how)
- save results in excel file
