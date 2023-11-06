"""
Script containing custom loss functions for training.

Author: Prisca Dotti
Last modified: 23.10.2023
"""

import logging
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DL_model.utils.LovaszSoftmax import lovasz_losses

__all__ = [
    "FocalLoss",
    "LovaszSoftmax",
    "LovaszSoftmax3d",
    "SumFocalLovasz",
    "SoftDiceLoss",
    "MySoftDiceLoss",
    "Dice_CELoss",
]

################################## FOCAL LOSS ##################################
"""
https://github.com/kornia/kornia
"""
# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

# add ignore index to focal loss implementation


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(type(labels))
        )

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype)
        )

    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one."
            " Got: {}".format(num_classes)
        )

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: List[float] = [],
    gamma: float = 2.0,
    reduction: str = "none",
    ignore_index: int = 9999,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha (list of floats): Weighting factor :math:`\alpha \in [0, 1]` for each class. Default: [] (no alpha weighting).
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Return:
        torch.Tensor: the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError(
            "Invalid input shape, we expect BxCx*. Got: {}".format(input.shape)
        )

    if input.size(0) != target.size(0):
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(
                input.size(0), target.size(0)
            )
        )

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(
            "Expected target size {}, got {}".format(out_size, target.size())
        )

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(
                input.device, target.device
            )
        )

    # create mask for ignore_index
    ignore_mask = torch.where(target == ignore_index, 0, 1)
    target_copy = torch.clone(target).detach()  # do not modify original target
    # convert target to torch.int64
    target_copy = target_copy.to(torch.int64)
    temp_ignored_index = 0  # the corresponding pts will be ignored
    target_copy[target_copy == ignore_index] = temp_ignored_index
    num_ignored = torch.count_nonzero(ignore_mask)

    if len(alpha) == 0:
        alpha_mask = torch.Tensor([1.0] * input.shape[1])
    else:
        # create alpha mask that will multiply focal loss
        assert len(alpha) == input.shape[1], "alpha does not contain a weight per class"
        alpha_mask = torch.zeros(target.shape)
        for idx, alpha_t in enumerate(alpha):
            alpha_mask[target == idx] = alpha_t

    alpha_mask = alpha_mask.to(target.device)

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps
    # input_soft: torch.Tensor = input + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target_copy, num_classes=input.shape[1], device=input.device, dtype=input.dtype
    )

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)
    focal = -weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    # remove loss values in ignored points
    loss_tmp = loss_tmp * ignore_mask * alpha_mask

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        # loss_old = torch.mean(loss_tmp)
        loss = torch.sum(loss_tmp) / num_ignored
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        alpha: List[float] = [],
        gamma: float = 2.0,
        reduction: str = "none",
        ignore_index: int = 9999,
        eps: float = 1e-8,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: list = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            input,
            target,
            self.alpha,
            self.gamma,
            self.reduction,
            self.ignore_index,
            self.eps,
        )


############################# LOVASZ-SOFTMAX LOSS ##############################
"""
from https://github.com/bermanmaxim/LovaszSoftmax
"""


def lovasz_softmax_3d(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: Union[str, List[int]] = "present",
    per_image: bool = False,
    ignore: Union[None, int] = None,
) -> torch.Tensor:
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, D, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, D, H, W].
      labels: [B, D, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """

    if per_image:
        loss = lovasz_losses.nanmean(
            lovasz_losses.lovasz_softmax_flat(
                *flatten_probas_3d(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
        # Convert loss to tensor
        loss = torch.tensor(loss)
    else:
        loss = lovasz_losses.lovasz_softmax_flat(
            *flatten_probas_3d(probas, labels, ignore), classes=classes
        )
    return loss


def flatten_probas_3d(
    probas: torch.Tensor, labels: torch.Tensor, ignore: Union[None, int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 4:
        # assumes output of a sigmoid layer
        B, D, H, W = probas.size()
        probas = probas.view(B, 1, D, H, W)
    B, C, D, H, W = probas.size()
    probas = (
        probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    )  # B * D * H * W, C = P, C
    labels = labels.contiguous().view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class LovaszSoftmax3d(nn.Module):
    """
    Criterion that computes Lovasz-Softmax loss on 3-dimensional samples.
    """

    def __init__(
        self,
        classes: str = "present",
        per_image: bool = False,
        ignore: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ! probas are logits, not probabilities !
        return lovasz_softmax_3d(
            probas.exp(), labels, self.classes, self.per_image, self.ignore
        )


class LovaszSoftmax(nn.Module):
    """
    Criterion that computes Lovasz-Softmax loss on 2-dimensional samples.
    """

    def __init__(
        self,
        classes: str = "present",
        per_image: bool = False,
        ignore: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ! probas are logits, not probabilities !
        return lovasz_losses.lovasz_softmax(
            probas.exp(), labels, self.classes, self.per_image, self.ignore
        )


################### FOCAL & LOVASZ-SOFTMAX LOSS COMBINATIONS ###################


class SumFocalLovasz(nn.Module):
    """
    Criterion that sums Focal and Lovasz-Softmax loss.
    ignore:     ignore label
    alpha:      focal loss weighting factors for each class
    gamma:      focal loss focusing parameter
    w:          wegthing factor between focal loss and lovasz-softmax loss
    reduction:  focal_loss redution
    """

    def __init__(
        self,
        classes: str = "present",
        per_image: bool = False,
        ignore: int = 9999,
        alpha: List[float] = [],
        gamma: float = 2.0,
        reduction: str = "none",
        eps: float = 1e-8,
        w: float = 0.5,
    ) -> None:
        super(SumFocalLovasz, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.w = w

        self.focal_loss = FocalLoss(
            reduction=self.reduction,
            ignore_index=self.ignore,
            alpha=self.alpha,
            gamma=self.gamma,
            eps=self.eps,
        )
        self.lovasz_softmax = LovaszSoftmax3d(
            classes=self.classes, per_image=self.per_image, ignore=self.ignore
        )

    def forward(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.w * self.focal_loss(probas, labels) + (
            1 - self.w
        ) * self.lovasz_softmax(probas, labels)

        return loss


################################## DICE LOSS ##################################

"""
Code from https://github.com/MIC-DKFZ/nnUNet
"""


def sum_tensor(
    inp: torch.Tensor, axes: np.ndarray, keepdim: bool = False
) -> torch.Tensor:
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(
    net_output: torch.Tensor,
    gt: torch.Tensor,
    axes: Sequence[int] = [],
    mask: Optional[torch.Tensor] = None,
    square: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if len(axes) == 0:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1
        )
        fp = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1
        )
        fn = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1
        )
        tn = torch.stack(
            tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1
        )

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    if len(axes) > 0:
        tp = sum_tensor(tp, np.array(axes), keepdim=False)
        fp = sum_tensor(fp, np.array(axes), keepdim=False)
        fn = sum_tensor(fn, np.array(axes), keepdim=False)
        tn = sum_tensor(tn, np.array(axes), keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Optional[nn.Module] = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
    ) -> None:
        """ """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MySoftDiceLoss(SoftDiceLoss):
    # Remark: I am not sure if this requires logits or probabilities !!!!
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return 1 + super().forward(x, y, loss_mask)


################################## DICE + CE LOSS #############################

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:39:03 2020

@author: Negin
(modified Negin's code to work with 3D data)
"""


class Dice_CELoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor],
        ignore_index: int = 0,
        #  ignore_first=True,  apply_softmax=True
    ) -> None:
        super(Dice_CELoss, self).__init__()
        self.eps = 1
        # self.ignore_first = ignore_first
        # self.apply_softmax = apply_softmax
        # self.CE = nn.CrossEntropyLoss()
        self.ignore_index = ignore_index
        self.NLLLoss = nn.NLLLoss(
            ignore_index=self.ignore_index,
            weight=weight,  # .to(device, non_blocking=True)
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE_loss = self.CE(input, target)
        CE_loss = self.NLLLoss(input, target)

        # if self.apply_softmax:
        #     input = input.softmax(dim=1)

        # UNet's last layer is a log_softmax layer
        input = input.exp()

        # remove ignored ROIs from input
        # for all 4 classes in the input, set values where target == ignore_index to 0
        input_clean = torch.zeros_like(input)
        for i in range(input.shape[1]):
            input_clean[:, i] = torch.where(target == self.ignore_index, 0, input[:, i])

        # remove ignored ROIs from target
        target_clean = torch.where(target == self.ignore_index, 0, target)

        target_one_hot = F.one_hot(
            target_clean.long(), num_classes=input_clean.shape[1]
        ).permute(0, 4, 1, 2, 3)

        # ignoring background -> necessary because of ignored ROIs
        input_clean = input_clean[:, 1:]
        target_one_hot = target_one_hot[:, 1:]

        intersection = torch.sum(target_one_hot * input_clean, dim=(2, 3, 4))
        cardinality = torch.sum(target_one_hot + input_clean, dim=(2, 3, 4))

        dice = (2 * intersection + self.eps) / (cardinality + self.eps)

        dice = torch.mean(torch.sum(dice, dim=1) / input_clean.size(dim=1))
        # input.size(dim=1) is the number of classes

        loss = 0.8 * CE_loss - 0.2 * torch.log(dice)
        return loss


### ORIGINAL DEFINITIONS OF FUNCTIONS ###

# """
# https://github.com/kornia/kornia
# """
# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

# add ignore index to focal loss implementation


# def focal_loss(
#         input: torch.Tensor,
#         target: torch.Tensor,
#         alpha: float,
#         gamma: float = 2.0,
#         reduction: str = 'none',
#         eps: float = 1e-8) -> torch.Tensor:
#     r"""Criterion that computes Focal loss.

#     According to :cite:`lin2018focal`, the Focal loss is computed as follows:

#     .. math::

#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

#     Where:
#        - :math:`p_t` is the model's estimated probability for each class.

#     Args:
#         input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
#         target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
#         alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
#         reduction (str, optional): Specifies the reduction to apply to the
#          output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#          'mean': the sum of the output will be divided by the number of elements
#          in the output, 'sum': the output will be summed. Default: 'none'.
#         eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

#     Return:
#         torch.Tensor: the computed loss.

#     Example:
#         >>> N = 5  # num_classes
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
#         >>> output.backward()
#     """
#     if not isinstance(input, torch.Tensor):
#         raise TypeError("Input type is not a torch.Tensor. Got {}"
#                         .format(type(input)))

#     if not len(input.shape) >= 2:
#         raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
#                          .format(input.shape))

#     if input.size(0) != target.size(0):
#         raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
#                          .format(input.size(0), target.size(0)))

#     n = input.size(0)
#     out_size = (n,) + input.size()[2:]
#     if target.size()[1:] != input.size()[2:]:
#         raise ValueError('Expected target size {}, got {}'.format(
#             out_size, target.size()))

#     if not input.device == target.device:
#         raise ValueError(
#             "input and target must be in the same device. Got: {} and {}" .format(
#                 input.device, target.device))

#     # compute softmax over the classes axis
#     input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(
#         target, num_classes=input.shape[1],
#         device=input.device, dtype=input.dtype)

#     # compute the actual focal loss
#     weight = torch.pow(-input_soft + 1., gamma)

#     focal = -alpha * weight * torch.log(input_soft)
#     loss_tmp = torch.sum(target_one_hot * focal, dim=1)

#     if reduction == 'none':
#         loss = loss_tmp
#     elif reduction == 'mean':
#         loss = torch.mean(loss_tmp)
#     elif reduction == 'sum':
#         loss = torch.sum(loss_tmp)
#     else:
#         raise NotImplementedError("Invalid reduction mode: {}"
#                                   .format(reduction))
#     return loss


# class FocalLoss(nn.Module):
#     r"""Criterion that computes Focal loss.

#     According to :cite:`lin2018focal`, the Focal loss is computed as follows:

#     .. math::

#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

#     Where:
#        - :math:`p_t` is the model's estimated probability for each class.

#     Args:
#         alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
#         reduction (str, optional): Specifies the reduction to apply to the
#          output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#          'mean': the sum of the output will be divided by the number of elements
#          in the output, 'sum': the output will be summed. Default: 'none'.
#         eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

#     Shape:
#         - Input: :math:`(N, C, *)` where C = number of classes.
#         - Target: :math:`(N, *)` where each value is
#           :math:`0 ≤ targets[i] ≤ C−1`.

#     Example:
#         >>> N = 5  # num_classes
#         >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
#         >>> criterion = FocalLoss(**kwargs)
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = criterion(input, target)
#         >>> output.backward()
#     """

#     def __init__(self, alpha: float, gamma: float = 2.0,
#                  reduction: str = 'none', eps: float = 1e-8) -> None:
#         super(FocalLoss, self).__init__()
#         self.alpha: float = alpha
#         self.gamma: float = gamma
#         self.reduction: str = reduction
#         self.eps: float = eps

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


# def binary_focal_loss_with_logits(
#         input: torch.Tensor,
#         target: torch.Tensor,
#         alpha: float = .25,
#         gamma: float = 2.0,
#         reduction: str = 'none',
#         eps: float = 1e-8) -> torch.Tensor:
#     r"""Function that computes Binary Focal loss.

#     .. math::

#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

#     where:
#        - :math:`p_t` is the model's estimated probability for each class.

#     Args:
#         input (torch.Tensor): input data tensor with shape :math:`(N, 1, *)`.
#         target (torch.Tensor): the target tensor with shape :math:`(N, 1, *)`.
#         alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`. Default: 0.25.
#         gamma (float): Focusing parameter :math:`\gamma >= 0`. Default: 2.0.
#         reduction (str, optional): Specifies the reduction to apply to the. Default: 'none'.
#         eps (float): for numerically stability when dividing. Default: 1e-8.

#     Returns:
#         torch.tensor: the computed loss.

#     Examples:
#         >>> num_classes = 1
#         >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
#         >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
#         >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
#         >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
#         tensor(4.6052)
#     """

#     if not isinstance(input, torch.Tensor):
#         raise TypeError("Input type is not a torch.Tensor. Got {}"
#                         .format(type(input)))

#     if not len(input.shape) >= 2:
#         raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
#                          .format(input.shape))

#     if input.size(0) != target.size(0):
#         raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
#                          .format(input.size(0), target.size(0)))

#     probs = torch.sigmoid(input)
#     target = target.unsqueeze(dim=1)
#     loss_tmp = -alpha * torch.pow((1. - probs), gamma) * target * torch.log(probs + eps) \
#                - (1 - alpha) * torch.pow(probs, gamma) * (1. - target) * torch.log(1. - probs + eps)
#     loss_tmp = loss_tmp.squeeze(dim=1)

#     if reduction == 'none':
#         loss = loss_tmp
#     elif reduction == 'mean':
#         loss = torch.mean(loss_tmp)
#     elif reduction == 'sum':
#         loss = torch.sum(loss_tmp)
#     else:
#         raise NotImplementedError("Invalid reduction mode: {}"
#                                   .format(reduction))
#     return loss


# class BinaryFocalLossWithLogits(nn.Module):
#     r"""Criterion that computes Focal loss.

#     According to :cite:`lin2017focal`, the Focal loss is computed as follows:

#     .. math::

#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

#     where:
#        - :math:`p_t` is the model's estimated probability for each class.

#     Args:
#         alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
#         gamma (float): Focusing parameter :math:`\gamma >= 0`.
#         reduction (str, optional): Specifies the reduction to apply to the
#          output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
#          'mean': the sum of the output will be divided by the number of elements
#          in the output, 'sum': the output will be summed. Default: 'none'.

#     Shape:
#         - Input: :math:`(N, 1, *)`.
#         - Target: :math:`(N, 1, *)`.

#     Examples:
#         >>> N = 1  # num_classes
#         >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
#         >>> loss = BinaryFocalLossWithLogits(**kwargs)
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = loss(input, target)
#         >>> output.backward()
#     """

#     def __init__(self, alpha: float, gamma: float = 2.0,
#                  reduction: str = 'none') -> None:
#         super(BinaryFocalLossWithLogits, self).__init__()
#         self.alpha: float = alpha
#         self.gamma: float = gamma
#         self.reduction: str = reduction
#         self.eps: float = 1e-8

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return binary_focal_loss_with_logits(
#             input, target, self.alpha, self.gamma, self.reduction, self.eps)

# """
# Created on Thu Dec 24 16:39:03 2020
# @author: Negin
# """

# class Dice_CELoss(nn.Module):
#     def __init__(self, ignore_first=True, apply_softmax=True):
#         super(Dice_CELoss, self).__init__()
#         self.eps = 1
#         self.ignore_first = ignore_first
#         self.apply_softmax = apply_softmax
#         self.CE = nn.CrossEntropyLoss()

#     def forward(self, input, target):

#         CE_loss = self.CE(input, target)

#         if self.apply_softmax:
#             input = input.softmax(dim=1)

#         target_one_hot = F.one_hot(target.long(), num_classes=input.shape[1]).permute(0,3,1,2)

#         if self.ignore_first:
#             input = input[:, 1:]
#             target_one_hot = target_one_hot[:, 1:]


#         intersection= torch.sum(target_one_hot*input,dim=(2,3))
#         cardinality= torch.sum(target_one_hot+input,dim=(2,3))


#         dice=(2*intersection+self.eps)/(cardinality+self.eps)

#         dice = torch.mean(torch.sum(dice, dim=1)/input.size(dim=1))

#         loss = 0.8*CE_loss-0.2*torch.log(dice)
#         return loss
