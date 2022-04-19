
import numpy as np

def _box_in_bounds(box, image_shape):

    newbox = []
    pad_width = []

    for box_i, shape_i in zip(box, image_shape):

        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

        newbox.append(newbox_i)
        pad_width.append(pad_width_i)

    needs_padding = any(i != (0, 0) for i in pad_width)

    return newbox, pad_width, needs_padding

def get_patch(image, patch_shape, center, mode='constant'):

    if mode == 'reflect':
        # We need to deal with even patch shapes when the mode is reflect
        correction_slice = tuple(slice(None, None if sh & 1 else -1) for sh in patch_shape)
        patch_shape = tuple(sh | 1 for sh in patch_shape)

    box = [(i-ps//2, i-ps//2+ps) for i, ps in zip(center, patch_shape)]

    box, pad_width, needs_padding = _box_in_bounds(box, image.shape)
    slices = tuple(slice(i[0], i[1]) for i in box)

    patch = image[slices]

    if needs_padding:
        if len(pad_width) < patch.ndim:
            pad_width.append((0, 0))
        patch = np.pad(patch, pad_width, mode=mode)

    if mode == 'reflect':
        patch = patch[correction_slice]

    return patch
