"""
Script containing the architectures tried in the experiments.

Author: Prisca Dotti
Last modified: 23.10.2023
"""

import logging
from typing import List, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from config import TrainingConfig
from models import UNet as unet
from models.UNet.unet.network import crop_and_merge

### temporal reducion unet ###

logger = logging.getLogger(__name__)

__all__ = ["TempRedUNet", "UNetConvLSTM", "ConvLSTM", "ConvLSTMCell", "UNetPadWrapper"]


class UNetPadWrapper(unet.UNetClassifier):
    def __init__(self, config: unet.UNetConfig) -> None:
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2]
        w = x.shape[-1]
        steps = self.config.steps

        # Calculate the required padding for both height and width:
        h_pad = 2**steps - h % 2**steps if h % 2**steps != 0 else 0
        w_pad = 2**steps - w % 2**steps if w % 2**steps != 0 else 0

        # Pad the input tensor:
        x = F.pad(
            x,
            (w_pad // 2, w_pad // 2 + w_pad % 2, h_pad // 2, h_pad // 2 + h_pad % 2),
        )

        # Apply the forward pass:
        x = super().forward(x)

        # Remove the padding:
        crop_h_start = h_pad // 2
        crop_h_end = -(h_pad // 2 + h_pad % 2) if h_pad > 0 else None
        crop_w_start = w_pad // 2
        crop_w_end = -(w_pad // 2 + w_pad % 2) if w_pad > 0 else None
        x = x[..., crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return x


class TempRedUNet(unet.UNet):
    def __init__(self, unet_config: unet.UNetConfig) -> None:
        super().__init__(unet_config)

        padding = {"valid": 0, "same": unet_config.dilation}[unet_config.border_mode]

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3, 3),
            dilation=1,
            padding=padding,
        )
        self.conv2 = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            dilation=1,
            padding=(padding, 0, 0),
        )
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=4,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            dilation=1,
            padding=(padding, 0, 0),
        )

        self.unet = unet.UNetClassifier(unet_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2]
        w = x.shape[-1]
        steps = super().config.steps

        # Calculate the required padding for both height and width:
        h_pad = 2**steps - h % 2**steps if h % 2**steps != 0 else 0
        w_pad = 2**steps - w % 2**steps if w % 2**steps != 0 else 0

        # Pad the input tensor:
        x = F.pad(
            x,
            (w_pad // 2, w_pad // 2 + w_pad % 2, h_pad // 2, h_pad // 2 + h_pad % 2),
        )

        # Apply the forward pass:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.unet(x)

        # Remove the padding:
        crop_h_start = h_pad // 2
        crop_h_end = -(h_pad // 2 + h_pad % 2) if h_pad > 0 else None
        crop_w_start = w_pad // 2
        crop_w_end = -(w_pad // 2 + w_pad % 2) if w_pad > 0 else None
        x = x[..., crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return x


### unet with convLSTM ###


# define convLSTM
class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        elif isinstance(kernel_size, tuple):
            padding = tuple([k // 2 for k in kernel_size])
        else:
            raise ValueError(
                "Invalid kernel_size. Must be of type int or tuple but was {}".format(
                    type(kernel_size)
                )
            )

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # self.conv.weight.data.normal_(0, 0.01)

    def forward(
        self,
        input: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = hidden_state

        # concatenate along channel axis
        combined = torch.cat((input, hidden), dim=1)

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.tanh(cc_g)

        cell = f * cell + i * g
        hidden = o * F.tanh(cell)

        return hidden, cell

    def _init_hidden(
        self, batch_size: int, image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )

    def __repr__(self) -> str:
        return "ConvLSTMCell(input_channels={}, hidden_channels={}, kernel_size={})".format(
            self.input_channels, self.hidden_channels, self.kernel_size
        )


class ConvLSTM(nn.Module):

    """
    Adapted from: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

    Parameters:
        input_channels: Number of channels in input
        hidden_channels: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple: (layer_output, last_state_list)
            0 - layer_output is the list of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        num_layers: int,
        batch_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_channels=cur_input_dim,
                    hidden_channels=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: List[Tuple[torch.Tensor, torch.Tensor]] = [],
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if len(hidden_state) == 0:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        # Input shape: (b, c, t, h, w)
        # Hidden layer shape: (b, c, h, w)

        # permute input shape to (b, t, c, h, w)
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input=cur_layer_input[:, t, :, :, :], hidden_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        layer_output = layer_output_list[-1]

        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4)

        # permute output to (b, c, t, h, w)
        layer_output = layer_output.permute(0, 2, 1, 3, 4)

        return layer_output, last_state_list

    def _init_hidden(
        self, batch_size: int, image_size: Tuple[int, int]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i]._init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(
        kernel_size: Union[int, Tuple[int, int]]
    ) -> None:
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(
        param: Union[int, List[int]], num_layers: int
    ) -> List[int]:
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class UNetConvLSTM(unet.UNet):
    # for the moment just copy the parent class
    def __init__(self, config: unet.UNetConfig, bidirectional: bool = False) -> None:
        super().__init__(config)

        # check that the number of dimensions is 3
        assert config.ndims == 2, "UNetConvLSTM is only implemented for tyx inputs"

        # add convLSTM layers
        n_channels1 = self.down_path[-1].layers[0].out_channels
        self.convLSTM1 = ConvLSTM(
            input_channels=n_channels1,
            hidden_channels=[n_channels1],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
        )

        n_channels2 = self.up_path[-1].layers[0].out_channels
        self.convLSTM2 = ConvLSTM(
            input_channels=n_channels2,
            hidden_channels=[n_channels2],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
        )

        self.bidirectional = bidirectional
        if bidirectional:
            # create convolutional layer to combine the two directions
            self.conv_bidirectional = nn.Conv2d(
                in_channels=n_channels1, out_channels=n_channels1, kernel_size=(1, 1)
            )

        # the final layer must be 2d
        # self.final_layer = nn.Conv2d(in_channels=self.config.first_layer_channels,
        #                              out_channels=self.config.num_classes,
        #                              kernel_size=1)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    # define wrappers to compute 2D layers on each timestep of the 3D input

    def _wrapper_conv_layer(
        self, layer: Type[nn.Module], x: torch.Tensor
    ) -> torch.Tensor:
        b, _, t, h, w = x.shape
        out_shape = (b, layer.layers[0].out_channels, t, h, w)
        output = torch.zeros(out_shape, device=x.device)

        for i in range(t):
            output[:, :, i, :, :] = layer(x[:, :, i, :, :])

        return output

    def _wrapper_down_maxpool(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        denom = self.max_pool.kernel_size
        out_shape = (b, c, t, h // denom, w // denom)
        output = torch.zeros(out_shape, device=x.device)

        for i in range(t):
            output[:, :, i, :, :] = self.max_pool(x[:, :, i, :, :])

        return output

    def _wrapper_upconv_layer(
        self, layer: Type[nn.Module], x: torch.Tensor
    ) -> torch.Tensor:
        b, _, t, h, w = x.shape
        coeff = layer.stride[0]
        out_shape = (b, layer.out_channels, t, h * coeff, w * coeff)
        output = torch.zeros(out_shape, device=x.device)

        for i in range(t):
            output[:, :, i, :, :] = layer(x[:, :, i, :, :])

        return output

    def _wrapper_conv_bidirectional(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = x.shape
        out_shape = (b, self.conv_bidirectional.out_channels, t, h, w)
        output = torch.zeros(out_shape, device=x.device)

        for i in range(t):
            output[:, :, i, :, :] = self.conv_bidirectional(x[:, :, i, :, :])

        return output

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x is B 1 T H W
        x = self._wrapper_conv_layer(self.down_path[0], x)

        down_outputs = [x]
        for unet_layer in self.down_path[1:]:
            x = self._wrapper_down_maxpool(x)
            x = self._wrapper_conv_layer(unet_layer, x)
            down_outputs.append(x)

        if not self.bidirectional:
            x, _ = self.convLSTM1(x)
        else:
            x_forward, _ = self.convLSTM1(x)
            x_backward, _ = self.convLSTM1(torch.flip(x, [2]))
            # combine the two directions using a convolutional layer
            x = self._wrapper_conv_bidirectional(
                x_forward + torch.flip(x_backward, [2])
            )

        for upconv_layer, unet_layer, down_output in zip(
            self.up_path[::2], self.up_path[1::2], down_outputs[-2::-1]
        ):
            x = self._wrapper_upconv_layer(upconv_layer, x)

            if not self.config.remove_skip_connections:
                x = crop_and_merge(down_output, x)
            else:
                aux = torch.zeros_like(down_output)
                x = crop_and_merge(aux, x)

            x = self._wrapper_conv_layer(unet_layer, x)

        if not self.bidirectional:
            x, _ = self.convLSTM2(x)
        else:
            x_forward, _ = self.convLSTM2(x)
            x_backward, _ = self.convLSTM2(torch.flip(x, [2]))
            # combine the two directions summing them
            x = x_forward + torch.flip(x_backward, [2])

        # extract middle frame to get the final output
        x = x[:, :, x.shape[2] // 2, :, :]  # B F H W

        x = self.final_layer(x)

        # from regressor to classifier
        x = self.logsoftmax(x)

        return x
