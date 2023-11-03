
import logging

import numpy as np
from scipy import ndimage

from . import srng
from . import patch_utils
from .utils import BinCounter

__all__ = ["Sampler", "PatchImportanceSampler"]

logger = logging.getLogger(__name__)


class Sampler(object):
    
    def __init__(self,
                 datasets,
                 minibatch_size,
                 rng=None):
        
        self.rng = rng or srng.RNG()
        self.minibatch_size = minibatch_size
        
        self.num_samples = len(datasets[0])
        self.iters_per_epoch = self.num_samples // self.minibatch_size
        
        self.datasets = datasets
        
        self._indices = None
        self._indices_epoch = None
    
    def __getitem__(self, index):
        return self.get_minibatch(index)
    
    @staticmethod
    def _array_or_list(elements):
        if len(elements) > 0 and isinstance(elements[0], np.ndarray):
            return np.array(elements)
        else:
            return elements
    
    def get_minibatch(self, index):
        
        epoch, i = divmod(index, self.iters_per_epoch)
        
        self._shuffle(epoch)
        
        minibatch_elements = self._indices[i * self.minibatch_size : (i + 1) * self.minibatch_size]
        
        # Required for H5 compatibility.
        minibatch = tuple(self._array_or_list([d[e] for e in minibatch_elements]) for d in self.datasets)
        
        return minibatch
    
    def _shuffle(self, epoch):
        
        if self._indices_epoch == epoch:
            return
        
        self._indices = np.arange(self.num_samples)
        self.rng.shuffle(epoch, self._indices)
        self._indices_epoch = epoch
    

class PatchImportanceSampler(object):
    
    class CenterImportanceSampler(object):
        
        def __init__(self, importance, rng):
            
            self.img_size = importance.size
            self.img_shape = importance.shape
            
            self.indices = np.arange(self.img_size)
            importance = np.float_(importance.flatten())
            importance = np.cumsum(importance)
            importance /= importance[-1]
            
            self.importance = importance
            self.rng = rng
        
        def sample_center(self, n):
            aux = self.rng.uniform(n)
            index = self.indices[np.searchsorted(self.importance, aux, side='right')]
            center = np.unravel_index(index, dims=self.img_shape)
            
            return center
    
    def __init__(self, unet_config,
                 training_x, training_y,
                 in_patch_shape, out_patch_shape,
                 loss_weights=None,
                 sampling_weights=None,
                 transformations=[lambda x: x],
                 mask_func=np.isnan,
                 border_mode='reflect',
                 rng=None):
        
        if loss_weights is None:
            loss_weights = [np.ones(ty_i.shape) for ty_i in training_y]
        
        if sampling_weights is None:
            sampling_weights = loss_weights
        
        # Discard elements with all sampling weights set to 0.
        training_x, training_y, loss_weights, sampling_weights = \
            self._filter((np.any(i != 0) for i in sampling_weights),
                         training_x, training_y, loss_weights, sampling_weights)
        
        # for ty_i in training_y:
        #     if any([a > b for a, b in zip(patch_shape, ty_i.shape)]):
        #         raise ValueError("The patch_shape {} is larger than the shape {} of one of the training images.".format(patch_shape, ty_i.shape))
        
        if not (len(training_x) == len(training_y) == len(sampling_weights) == len(loss_weights)):
            raise ValueError("The length of `training_x`, `training_y`, `sampling_weights` and `loss_weights` must be equal.")
        
        if not all([i.shape[:unet_config.ndims] == j.shape == k.shape == m.shape for i, j, k, m in zip(training_x, training_y, sampling_weights, loss_weights)]):
            raise ValueError("The shape of `training_x`, `training_y`, `sampling_weights` and `loss_weights` must be equal.")
        
        self.rng = rng or srng.RNG()
        
        num_classes = unet_config.num_classes
        if num_classes > 1:
            self.training_counter = BinCounter(num_classes + 1)
            self.weighted_counter = BinCounter(num_classes + 1)
        
        self.unet_config = unet_config
        self.training_x = training_x
        self.training_y = training_y
        
        self.in_patch_shape = in_patch_shape
        self.out_patch_shape = out_patch_shape
        self.transformations = transformations
        self.mask_func = mask_func
        self.border_mode = border_mode
        
        self.iters_per_epoch = len(self.training_x) * len(self.transformations)
        self._indices = None
        self._indices_epoch = None
        
        self.loss_weights = loss_weights
        self.update_sampling_weights(sampling_weights)
    
    @staticmethod
    def _filter(mask, *args):
        return tuple(zip(*[values
                           for mask_i, *values in zip(mask, *args)
                           if mask_i]))
    
    def update_sampling_weights(self, sampling_weights):
        """
        Updates sampling weights. Make sure to pad them before calling this function.
        """
        
        # Compute patch importance summing the importances of pixels contained
        # in every patch.
        patch_importances = (ndimage.uniform_filter(weights_i,
                                                    self.out_patch_shape,
                                                    mode='constant')
                                for weights_i in sampling_weights)
        
        # Set to zero the importances of the output patches that are outside the
        # bounds of training_y.
        patch_importances = [self.zero_margin(i, self.out_patch_shape)
                                for i in patch_importances]
        
        samplers = [self.CenterImportanceSampler(i, self.rng)
                        for i in patch_importances]
        
        # origin needs to be corrected if patch size is even
        fixed_origin = [-1 if p % 2 == 0 else 0 for p in self.out_patch_shape]
        
        # Probability of sampling every pixel is the average of the importances
        # of the patches that contain that pixel.
        pix_sampling_prob = [ndimage.uniform_filter(i, self.out_patch_shape,
                                                    mode='constant', origin=fixed_origin)
                                for i in patch_importances]
        
        # compute normalization weights
        norm_weights = [ps_i.size / np.sum(ps_i)
                            for ps_i in pix_sampling_prob]
        
        sampling_correction_factor = [lw / (1e-9 + nw * ps)
                                        for lw, nw, ps in zip(self.loss_weights, norm_weights, pix_sampling_prob)]
        
        self._patch_importances = patch_importances
        self._sampling_weights = sampling_weights
        self._pix_sampling_prob = pix_sampling_prob
        self.sampling_correction_factor = sampling_correction_factor
        self.samplers = samplers
    
    @staticmethod
    def zero_margin(importance, patch_shape):
        margin1 = tuple(ps // 2 for ps in patch_shape)
        margin2 = tuple(ps - ps // 2 - 1 for ps in patch_shape)
        slices = tuple(slice(m1, -m2 if m2 != 0 else None) for m1, m2 in zip(margin1, margin2))
        res = np.zeros_like(importance)
        res[slices] = importance[slices]
        
        return res
    
    def get_minibatch(self, index):
        
        patch_x, patch_y, patch_w = self.sample_patch_xyw(index)
        patch_y = np.copy(patch_y)
        patch_w = patch_w.astype(np.float32)
        
        mask = self.mask_func(patch_y)
        patch_y[mask] = 0
        patch_w[mask] = 0
        
        if not np.issubdtype(patch_y.dtype, float):
            # classification?
            self.training_counter.update(patch_y)
            self.weighted_counter.update(patch_y, patch_w)
        
        # Transform to the right shape
        patch_x = patch_x[None, None, ...]
        patch_y = patch_y[None, ...]
        patch_w = patch_w[None, ...]
        
        return patch_x, patch_y, patch_w
    
    def sample_patch_xyw(self, index):
        
        patch_x = self.sample_patch(index, self.training_x, self.in_patch_shape)
        patch_y = self.sample_patch(index, self.training_y, self.out_patch_shape)
        patch_w = self.sample_patch(index, self.sampling_correction_factor, self.out_patch_shape)
        
        return patch_x, patch_y, patch_w
    
    def sample_patch(self, index, arrays, shape):
        
        if len(arrays) != len(self.samplers):
            raise ValueError("len(arrays)(={}) != len(self.samplers)(={})".format(len(arrays), len(self.samplers)))
        
        sample_idx, transform_idx = self._sample_transform_indices(index)
        
        current_arr = arrays[sample_idx]
        center = self.samplers[sample_idx].sample_center(index)
        transform = self.transformations[transform_idx]
        
        patch = patch_utils.get_patch(current_arr, shape, center, mode=self.border_mode)
        patch = transform(patch)
        
        return patch
    
    def _sample_transform_indices(self, index):
        epoch, epoch_index = divmod(index, self.iters_per_epoch)
        
        self._shuffle(epoch)
        
        element = self._indices[epoch_index]
        transform_idx, sample_idx  = divmod(element, len(self.samplers))
        
        return sample_idx, transform_idx
    
    def _shuffle(self, epoch):
        
        if self._indices_epoch == epoch:
            return
        
        self._indices = np.arange(self.iters_per_epoch)
        self.rng.shuffle(epoch, self._indices)
        self._indices_epoch = epoch
    
