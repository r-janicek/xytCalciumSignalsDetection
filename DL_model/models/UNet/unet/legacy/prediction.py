"""
Prediction
"""
from itertools import product
import logging
import numpy as np
from . import patch_utils

__all__ = ["predict_in_blocks", "predict_at_once"]

logger = logging.getLogger(__name__)


def ceil_div(a, b):
    
    return -(-a // b)


def centers(full, out_shape):
    
    start = out_shape // 2
    stop = full - (out_shape - out_shape // 2)
    
    count = ceil_div(stop - start, out_shape) + 1
    
    return np.int_(np.round(np.linspace(start, stop, count)))


def _check_and_prepare_image(image, ndims, image_format):
    
    if image_format not in ['NCHW', 'NHWC']:
        raise ValueError("image_format not in ['NCHW', 'NHWC']")
    
    if image.ndim < ndims + 1:
        # Add channel
        if image_format == 'NCHW':
            image = image[None]
        else:
            image = image[..., None]
    
    if image.ndim != ndims + 1:
        raise ValueError("invalid dimensions for the image; it should be {} or {}".format(ndims, ndims+1))
    
    return image


def predict_in_blocks(unet, image, hint_block_shape,
                      output_function=None,
                      image_format='NCHW',
                      pad_mode="reflect",
                      verbose=True):
    """
    Returns the prediction of the U-Net for the given `image`. Processes the
    image in overlapping blocks of the given size. If the image is not very
    large, `predict_at_once` might be more convenient and faster. Use this
    function when `predict_at_once` gives memory errors.
    """
    
    # TODO: This function can be simplified modifying patch_utils.get_patch
    # to work with images with NCHW format.
    
    ndims = unet.config.ndims
    image = _check_and_prepare_image(image, ndims, image_format)
    
    if output_function is None:
        output_function = unet.forward
    
    if image_format == 'NCHW':
        # Move channels to the end
        image = np.transpose(image, tuple(range(1, ndims+1)) + (0, ))
    
    # From this point, image has format NHWC
    image_shape = image.shape[:ndims] # Remove the channels in the shape
    in_shape, block_shape = unet.config.in_out_shape(hint_block_shape)
    
    if any(i < j for i, j in zip(image_shape, block_shape)):
        raise ValueError("image_shape {} is smaller than the block_shape {}; try a smaller value of hint_block_shape".format(image_shape, block_shape))
    
    grid = map(centers, image_shape, block_shape)
    
    num_channels = unet.config.num_classes
    # The result also has format NHWC
    result = np.zeros(image_shape + (num_channels,), dtype=np.float32)
    
    for c in product(*grid):
        if verbose:
            logger.info("Center {}...".format(c))
        
        patch_x = patch_utils.get_patch(image, in_shape, c, mode=pad_mode)
        
        patch_x = np.transpose(patch_x, (ndims, ) + tuple(range(0, ndims)))
        patch_o = output_function(patch_x[None, ...])[0]
        
        # Transpose the output so that the channels come at the end.
        patch_o = np.transpose(patch_o, tuple(range(1, ndims + 1)) + (0,))
        patch_utils.get_patch(result, block_shape, c)[:] = patch_o
    
    # Change format of the result if required
    if image_format == 'NCHW':
        result = np.transpose(result, (ndims, ) + tuple(range(0, ndims)))
    
    return result


def predict_at_once(unet, image,
                    output_function=None,
                    image_format='NCHW',
                    pad_mode="reflect"):
    """
    Returns the prediction of the U-net for the given `image`. This is faster
    than `predict_in_blocks`, but it requires a large amount of memory. If the
    image is too large, use `predict_in_blocks`.
    """
    
    ndims = unet.config.ndims
    image = _check_and_prepare_image(image, ndims, image_format)
    
    if output_function is None:
        output_function = unet.forward
    
    if image_format == 'NHWC':
        image = np.transpose(image, (ndims, ) + tuple(range(0, ndims)))
    
    # From this point image has format NCHW.
    image_shape = image.shape[1:]
    in_pad, out_pad = unet.config.in_out_pad_widths(image_shape)
    pad_image = np.pad(image, [(0, 0)] + in_pad, pad_mode)
    
    # Compute the output
    out = output_function(pad_image[None, ...])[0]
    
    # Cut the extra padding from the output.
    out_slices = (slice(None),) + tuple(slice(i, -j if j > 0 else None) for i, j in out_pad)
    result = out[out_slices]
    
    # Change format if required.
    if image_format == 'NHWC':
        result = np.transpose(result, tuple(range(1, ndims+1)) + (0, ))
    
    return result
