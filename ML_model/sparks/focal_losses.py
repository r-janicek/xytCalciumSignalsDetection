'''
https://github.com/kornia/kornia
'''

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

# add ignore index to focal loss implementation

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: float = 1e-6) -> torch.Tensor:
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
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: list = None,
        gamma: float = 2.0,
        reduction: str = 'none',
        ignore_index: int = 9999,
        eps: float = 1e-8,) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha (float): Weighting factor :math:`\alpha` with shape `(C)`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
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
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))


    # create mask for ignore_index
    #print("IGNORE INDEX", ignore_index)
    ignore_mask = torch.where(target == ignore_index, 0, 1)
    #print("IGNORE MASK\n", ignore_mask, "\n", ignore_mask.shape)
    target_copy = torch.clone(target).detach() # do not modify original target
    temp_ignored_index = 0 # the corresponding pts will be ignored
    target_copy[target_copy == ignore_index] = temp_ignored_index
    #print("TARGET COPY", target_copy)
    num_ignored = torch.count_nonzero(ignore_mask)
    #print("NUM IGNORED", num_ignored)


    if alpha == None:
        alpha_mask = 1.
    else:
        # create alpha mask that will multiply focal loss
        assert len(alpha) == input.shape[1], "alpha does not contain a weight per class"
        alpha_mask = torch.zeros(target.shape)
        for idx, alpha_t in enumerate(alpha):
            alpha_mask[target==idx] = alpha_t
        #print(alpha_mask)

    alpha_mask = alpha_mask.to(target.device)

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps
    #input_soft: torch.Tensor = input + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target_copy, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    #print("TARGET ONE HOT", target_one_hot, target_one_hot.shape)


    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)
    focal = - weight * torch.log(input_soft)
    #print("ALPHA MASK", alpha_mask)
    #print("FOCAL", focal)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    #print("LOSS TMP SHAPE", loss_tmp.shape)
    # remove loss values in ignored points
    loss_tmp = loss_tmp * ignore_mask * alpha_mask
    #print("LOSS TMP SHAPE", loss_tmp.shape)
    #print("LOSS TMP", loss_tmp)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        #loss_old = torch.mean(loss_tmp)
        loss = torch.sum(loss_tmp)/num_ignored
        #print(loss, loss_old)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


''' original version
def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
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
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss
'''


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
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
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

    def __init__(self, alpha: list = None, gamma: float = 2.0,
                 reduction: str = 'none', ignore_index: int = 9999,
                 eps: float = 1e-8) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: list = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction,
                          self.ignore_index, self.eps)


''' original version
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
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
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

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none', eps: float = 1e-8) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
'''


def binary_focal_loss_with_logits(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float = .25,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): input data tensor with shape :math:`(N, 1, *)`.
        target (torch.Tensor): the target tensor with shape :math:`(N, 1, *)`.
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`. Default: 0.25.
        gamma (float): Focusing parameter :math:`\gamma >= 0`. Default: 2.0.
        reduction (str, optional): Specifies the reduction to apply to the. Default: 'none'.
        eps (float): for numerically stability when dividing. Default: 1e-8.

    Returns:
        torch.tensor: the computed loss.

    Examples:
        >>> num_classes = 1
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(4.6052)
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    probs = torch.sigmoid(input)
    target = target.unsqueeze(dim=1)
    loss_tmp = -alpha * torch.pow((1. - probs), gamma) * target * torch.log(probs + eps) \
               - (1 - alpha) * torch.pow(probs, gamma) * (1. - target) * torch.log(1. - probs + eps)
    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss



class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2017focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, 1, *)`.
        - Target: :math:`(N, 1, *)`.

    Examples:
        >>> N = 1  # num_classes
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, self.eps)
