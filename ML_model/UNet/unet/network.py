
import numpy as np

import torch
import torch.nn as nn

from .layers import *

__all__ = ["UNet", 
           "UNetConfig", 
           "UNetClassifier", 
           "UNetRegressor"]


def crop_slices(shape1, shape2):

    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2)
                    for sh1, sh2 in zip(shape1, shape2)]
    return slices


def crop_and_merge(tensor1, tensor2):

    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)


class UNetConfig:

    def __init__(self,
                 steps=4,
                 first_layer_channels=64,
                 num_classes=2,
                 num_input_channels=1,
                 two_sublayers=True,
                 ndims=2,
                 border_mode='valid',
                 remove_skip_connections=False,
                 batch_normalization=False,
                 scaled_convolution=False):

        if border_mode not in ['valid', 'same']:
            raise ValueError("`border_mode` not in ['valid', 'same']")

        self.steps = steps
        self.first_layer_channels = first_layer_channels
        self.num_input_channels = num_input_channels
        self.num_classes = num_classes
        self.two_sublayers = two_sublayers
        self.ndims = ndims
        self.border_mode = border_mode
        self.remove_skip_connections = remove_skip_connections
        self.batch_normalization = batch_normalization
        self.scaled_convolution = scaled_convolution

        border = 4 if self.two_sublayers else 2
        if self.border_mode == 'same':
            border = 0
        self.first_step = lambda x: x - border
        self.rev_first_step = lambda x: x + border
        self.down_step = lambda x: (x - 1) // 2 + 1 - border
        self.rev_down_step = lambda x: (x + border) * 2
        self.up_step = lambda x: (x * 2) - border
        self.rev_up_step = lambda x: (x + border - 1) // 2 + 1

    # def __getstate__(self):
    #     return [self.steps, self.first_layer_channels, self.num_classes, self.two_sublayers, self.ndims, self.border_mode]

    # def __setstate__(self, state):
    #     return self.__init__(*state)

    # def __repr__(self):
    #     return "{0.__class__.__name__!s}(steps={0.steps!r}, first_layer_channels={0.first_layer_channels!r}, " \
    #             "num_classes={0.num_classes!r}, num_input_channels={0.num_input_channels!r}, "\
    #             "two_sublayers={0.two_sublayers!r}, ndims={0.ndims!r}, "\
    #             "border_mode={0.border_mode!r})".format(self)

    def out_shape(self, in_shape):
        """
        Return the shape of the output given the shape of the input
        """

        shapes = self.feature_map_shapes(in_shape)
        return shapes[-1][1:]

    def feature_map_shapes(self, in_shape):

        def _feature_map_shapes():

            shape = np.asarray(in_shape)
            yield (self.num_input_channels,) + tuple(shape)

            shape = self.first_step(shape)
            yield (self.first_layer_channels,) + tuple(shape)

            for i in range(self.steps):
                shape = self.down_step(shape)
                channels = self.first_layer_channels * 2 ** (i + 1)
                yield (channels,) + tuple(shape)

            for i in range(self.steps):
                shape = self.up_step(shape)
                channels = self.first_layer_channels * 2 ** (self.steps - i - 1)
                yield (channels,) + tuple(shape)

            yield (self.num_classes,) + tuple(shape)

        return list(_feature_map_shapes())

    def _out_step(self):
        
        _, out_shape_0 = self.in_out_shape((0,))
        _, out_shape_1 = self.in_out_shape((out_shape_0[0] + 1,))
        return out_shape_1[0] - out_shape_0[0]

    def in_out_shape(self, out_shape_lower_bound,
                     given_upper_bound=False):
        """
        Compute the best combination of input/output shapes given the
        desired lower bound for the shape of the output
        """
        
        if given_upper_bound:
            out_shape_upper_bound = out_shape_lower_bound
            out_step = self._out_step()
            out_shape_lower_bound = tuple(i - out_step + 1 for i in out_shape_upper_bound)

        shape = np.asarray(out_shape_lower_bound)

        for i in range(self.steps):
            shape = self.rev_up_step(shape)

        # Compute correct out shape from minimum shape
        out_shape = np.copy(shape)
        for i in range(self.steps):
            out_shape = self.up_step(out_shape)

        # Best input shape
        for i in range(self.steps):
            shape = self.rev_down_step(shape)

        shape = self.rev_first_step(shape)
        
        return tuple(shape), tuple(out_shape)
    
    def margin(self):
        """
        Return the size of the margin lost around the input images as a
        consequence of the sequence of convolutions and max-poolings.
        """
        in_shape, out_shape = self.in_out_shape((0,))
        return (in_shape[0] - out_shape[0]) // 2

    def in_out_pad_widths(self, out_shape_lower_bound):

        in_shape, out_shape = self.in_out_shape(out_shape_lower_bound)

        in_pad_widths = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                            for sh_i, sh_o in zip(out_shape_lower_bound, in_shape)]
        out_pad_widths = [((sh_o - sh_i) // 2, (sh_o - sh_i - 1) // 2 + 1)
                            for sh_i, sh_o in zip(out_shape_lower_bound, out_shape)]

        return in_pad_widths, out_pad_widths


class UNetLayer(nn.Module):

    def __init__(self, num_channels_in, num_channels_out,
                 conv_layer_class, bn_layer_class,
                 two_sublayers=True, border_mode='valid'):

        super().__init__()

        padding = {'valid': 0, 'same': 1}[border_mode]

        relu = nn.ReLU()
        conv1 = conv_layer_class(num_channels_in, num_channels_out,
                                 kernel_size=3, padding=padding)
        bn1 = bn_layer_class(num_channels_out)

        if not two_sublayers:
            layers = [conv1, relu, bn1]
        else:
            conv2 = conv_layer_class(num_channels_out, num_channels_out,
                                     kernel_size=3, padding=padding)
            bn2 = bn_layer_class(num_channels_out)
            layers = [conv1, relu, bn1, conv2, relu, bn2]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def _layer_classes(config):
    ndims = config.ndims
    bn = config.batch_normalization
    sc = config.scaled_convolution
    
    conv_class, convtransp_class = {
        (2, False): (nn.Conv2d, nn.ConvTranspose2d),
        (3, False): (nn.Conv3d, nn.ConvTranspose3d),
        (2, True): (ScaledConv2d, ScaledConvTranspose2d),
        (3, True): (ScaledConv3d, ScaledConvTranspose3d)
    }[ndims, sc]
    
    maxpool_class = {
        2: nn.MaxPool2d,
        3: nn.MaxPool3d
    }[ndims]
    
    bn_class = {
        (2, False): Identity,
        (3, False): Identity,
        (2, True): nn.BatchNorm2d,
        (3, True): nn.BatchNorm3d
    }[ndims, bn]
    
    return conv_class, convtransp_class, maxpool_class, bn_class


class UNet(nn.Module):
    """The U-Net."""

    def __init__(self, unet_config=None):
        super().__init__()

        if unet_config is None:
            unet_config = UNetConfig()
        self.config = unet_config

        ConvLayer, ConvTransposeLayer, MaxPool, BatchNorm = _layer_classes(self.config)
        self.max_pool = MaxPool(2)

        def build_ulayer(cin, cout):
            return UNetLayer(
                cin, cout,
                conv_layer_class=ConvLayer,
                bn_layer_class=BatchNorm,
                two_sublayers=self.config.two_sublayers,
                border_mode=self.config.border_mode,
            )

        flc = self.config.first_layer_channels
        layer1 = build_ulayer(self.config.num_input_channels,
                              flc)

        # Down layers.
        down_layers = [layer1]
        for i in range(1, self.config.steps + 1):
            lyr = build_ulayer(flc * 2**(i - 1),
                               flc * 2**i)

            down_layers.append(lyr)

        # Up layers
        up_layers = []
        for i in range(self.config.steps - 1, -1, -1):
            # Up-convolution
            upconv = ConvTransposeLayer(in_channels=flc * 2**(i+1),
                                        out_channels=flc * 2**i,
                                        kernel_size=2,
                                        stride=2)
            lyr = build_ulayer(flc * 2**(i + 1),
                               flc * 2**i)

            up_layers.extend([upconv, lyr])

        final_layer = ConvLayer(in_channels=flc,
                                out_channels=self.config.num_classes,
                                kernel_size=1)

        self.down_path = nn.Sequential(*down_layers)
        self.up_path = nn.Sequential(*up_layers)
        self.final_layer = final_layer

    def forward(self, x, return_feature_maps=False):
        
        x = self.down_path[0](x)
        
        feature_maps = []
        
        down_outputs = [x]
        for unet_layer in self.down_path[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)
        
        feature_maps.extend(down_outputs)
        
        for upconv_layer, unet_layer, down_output in zip(self.up_path[::2], self.up_path[1::2], down_outputs[-2::-1]):
            x = upconv_layer(x)
            
            if not self.config.remove_skip_connections:
                x = crop_and_merge(down_output, x)
            else:
                aux = torch.zeros_like(down_output)
                x = crop_and_merge(aux, x)
            x = unet_layer(x)
            feature_maps.append(x)

        x = self.final_layer(x)
        feature_maps.append(x)
        
        if return_feature_maps:
            return x, feature_maps
        
        return x


class UNetClassifier(UNet):
    """
    Wrapper class for the UNet used as a classifier.
    """

    def __init__(self, unet_config=None):
        super().__init__(unet_config)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def linear_output(self, x):
        return super().forward(x)
    
    def forward(self, x):
        x = self.linear_output(x)
        x = self.logsoftmax(x)
        return x
    
    def probability_output(self, x):
        return torch.exp(self(x))


class UNetRegressor(UNet):
    """
    Wrapper class for the UNet used as a regressor.
    """

    def __init__(self, unet_config=None):
        super().__init__(unet_config)

    def forward(self, x):
        return super().forward(x)
