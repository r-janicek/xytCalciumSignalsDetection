
"""
Some standard setups for U-Net training.
"""

from itertools import chain, product, permutations
import logging

import numpy as np

from . import network, trainer, samplers
from .utils import BinCounter, labels_to_probabilities, has_cuda
from .transformation import all_transformations, d4_transformations

import torch
from torch.autograd import Variable
import torch.optim as optim

logger = logging.getLogger(__name__)

def invfreq_lossweights(labels, num_classes):
    
    bc = BinCounter(num_classes + 1, labels)
    class_weight = 1.0 / (num_classes * bc.frequencies)[:num_classes]
    class_weight = np.hstack([class_weight, 0])
    weights = class_weight[labels]
    return weights

def basic_setup(unet,
                training_x, training_y,
                hint_patch_shape,
                loss_weights=None,
                augment_data=[lambda x: x],
                learning_rate=1e-4,
                save_path=None,
                save_every=None):
    """
    Basic setup for the U-Net. This function preprocess the traininig data,
    computes patch importances, creates a sampler and a solver, and returns a
    trainer.
    
    The sampler is `PatchImportanceSampler`. The importances are computed based
    on the relative frequency of every class with `invfreq_lossweights`. In case
    of regression with real values in `training_y`, uniform importances are
    used instead.
    
    The solver is Adam.
    
    `hint_patch_shape` is a lower bound to the output patch shape. The real
    input and output patch shapes will be computed from it according to the
    U-Net configuration (using `unet.config.in_out_shape`).
    
    Notes:
    
    - Training data must be larger than the patch size.
    
    - This setup is NOT valid if the training data consists of the final patches
    aimed for training.
    """
    
    ndims = unet.config.ndims
    num_classes = unet.config.num_classes
    in_patch_shape, out_patch_shape = unet.config.in_out_shape(hint_patch_shape)
    margin = unet.config.margin()
    
    # Print some info.
    logger.info("Input patch shape: {}".format(in_patch_shape))
    if out_patch_shape != hint_patch_shape:
        logger.warning("The hint_patch_shape {} is not a valid output patch shape for the given architecture".format(hint_patch_shape))
        logger.warning("\tThe output patch shape will be set to the closest larger valid shape")
    logger.info("Output patch shape: {}".format(out_patch_shape))
    logger.info("Margin: {}".format(margin))
    
    # Check that output shapes are larger than out_patch_shape
    for i, ty_i in enumerate(training_y):
        if any(i < j for i, j in zip(ty_i.shape, out_patch_shape)):
            raise ValueError("training_y[{}].shape {} is smaller than out_patch_shape {}; try setting a smaller hint_patch_shape when calling `basic_setup`".format(i, ty_i.shape, out_patch_shape))
    
    if isinstance(unet, network.UNetClassifier):
        mask_func = lambda x: x == num_classes
        
        if loss_weights is None:
            loss_weights = [invfreq_lossweights(ty_i, num_classes) for ty_i in training_y]
    elif isinstance(unet, network.UNetRegressor):
        mask_func = np.isnan
    else:
        raise NotImplementedError
    
    if augment_data is "none":
        transformations = [lambda x: x]
    elif augment_data == "d4":
        transformations = d4_transformations(ndims)
    elif augment_data == "all":
        transformations = all_transformations(ndims)
    else:
        raise ValueError("Unknown value '{}' for `augment_data`".format(augment_data))
    
    # Sampler, solver and trainer
    logger.info("Creating sampler...")
    sampler = samplers.PatchImportanceSampler(unet.config,
                                           training_x, training_y,
                                           in_patch_shape, out_patch_shape,
                                           loss_weights=loss_weights,
                                           transformations=transformations,
                                           mask_func=mask_func)
    
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(unet.parameters(), lr=learning_rate)
    
    def training_step(niter, sampler, unet, optimizer):
        
        # Get the minibatch
        x, y, w = sampler.get_minibatch(niter)
        
        y2 = y
        
        # Convert to pytorch
        x2 = Variable(torch.from_numpy(np.ascontiguousarray(x)))
        y2 = Variable(torch.from_numpy(np.ascontiguousarray(y)))
        w2 = Variable(torch.from_numpy(np.ascontiguousarray(w)))
        
        if has_cuda:
            x2 = x2.cuda()
            y2 = y2.cuda()
            w2 = w2.cuda()
        
        optimizer.zero_grad()
        loss = unet.loss(x2, y2, w2)
        loss.backward()
        optimizer.step()
        
        return {"loss": float(loss.data.cpu().numpy())}
    
    logger.info("Creating trainer...")
    unet_trainer = trainer.Trainer(lambda niter : training_step(niter, sampler, unet, optimizer),
                                   save_every=save_every or sampler.iters_per_epoch,
                                   save_path=save_path,
                                   managed_objects=trainer.managed_objects({"network": unet,
                                                                            "optimizer": optimizer}),
                                   test_function=None,
                                   test_every=None)
    
    return unet_trainer, sampler, optimizer
