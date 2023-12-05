
from os import listdir
import os.path
import itertools
import glob

import numpy as np
from imageio import imread

__all__ = ['load_volume']

image_file_extensions = [".png", ".tif", ".bmp", ".jpg", ".jpeg", ".tga"]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def tiffread(fname):
    """Read a multi-page TIFF image to a three-dimensional array."""
    from PIL import Image
    img = Image.open(fname)
    
    res = []
    offsets = []
    frame = 0
    try:
        for frame in itertools.count():
            img.seek(frame)
            aux = np.asarray(img)
            if aux.ndim == 0:
                if img.mode == 'I;16':
                    aux = np.fromstring(img.tostring(), np.uint16)
                    aux = np.reshape(aux, img.size[::-1])
                elif img.mode == 'I;16S':
                    aux = np.fromstring(img.tostring(), np.int16)
                    aux = np.reshape(aux, img.size[::-1])
                else:
                    raise ValueError("unknown pixel mode")
            res.append(aux)
    except EOFError:
        pass
    
    return np.asarray(res)


def nrrdread(fname):
    import nrrd
    return nrrd.read(fname)[0]


def search_volume_in_dir(path):
    
    filenames = (f for f in listdir(path) if os.path.isfile(os.path.join(path, f)))
    grouped = itertools.groupby(filenames, key=lambda x: os.path.splitext(x)[1].lower())
    grouped = {k: list(v) for k, v in grouped if k in image_file_extensions}
    largest_group = max(grouped.values(), key=len)
    return sort_nicely([os.path.join(path, f) for f in largest_group])


def load_volume(path):
    
    if os.path.isfile(path):
        if path.lower().endswith('.nrrd'):
            # we need to transpose to match the output of tiffread()
            img = nrrdread(path)
            if img.ndim == 4:
                img = np.transpose( img, (0,3,2,1) )
            else:
                img = np.transpose( img, (2,1,0) )

            return img

        else:
            return tiffread(path)
    elif os.path.isdir(path):
        filenames = search_volume_in_dir(path)
    else:
        # Check if path is a pattern.
        filenames = sort_nicely(glob.glob(path))
    
    if len(filenames) == 0:
        raise ValueError("no image files found in {}".format(path))
    return np.array(list(map(imread, filenames)))
