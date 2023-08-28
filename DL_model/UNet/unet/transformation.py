
from itertools import chain, product, permutations
from functools import partial
import logging

import numpy as np

logger = logging.getLogger(__name__)

def cube_group_action(transpose, mirror, arr):
    """
    `transpose` is a permutation of the axes (0, 1, 2...)
    
    `mirror` is a sequence of values in {0, 1} indicating mirroring in those
    axes after transposition
    """
    
    transpose = list(transpose) + list(range(len(transpose), arr.ndim))
    arr = arr.transpose(*transpose)
    mirror_slices = tuple(slice(None, None, -1 if i else None) for i in mirror)
    return arr[mirror_slices]

def all_transformations(ndim):
    transpositions = permutations(range(ndim))
    mirrors = product(*([0, 1],) * ndim)
    args = list(product(transpositions, mirrors))
    return [partial(cube_group_action, *arg) for arg in args]

def d4_transformations(ndim):
    extra_dims = ndim - 2
    transpositions = [tuple(range(extra_dims)) + i for i in permutations([extra_dims, extra_dims + 1])]
    mirrors = [((0,) * extra_dims) + i for i in product([0, 1], [0, 1])]
    args = list(product(transpositions, mirrors))
    return [partial(cube_group_action, *arg) for arg in args]
