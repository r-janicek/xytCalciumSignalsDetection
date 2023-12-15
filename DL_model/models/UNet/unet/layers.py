
import numpy as np
from torch import nn

__all__ = [
    "Identity",
    "ScaledConv2d",
    "ScaledConvTranspose2d",
    "ScaledConv3d",
    "ScaledConvTranspose3d"]


class Identity(nn.Module):
    
    def __init__(self, *_, **__):
        super().__init__()
    
    def forward(self, x):
        return x


def _scaled_conv_weight_init(module):
    nn.init.normal_(module.weight, mean=0, std=1)
    nn.init.constant_(module.bias, 0.0)


class ScaledConv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, normalized=False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        fan_in = in_channels * kernel_size ** 2
        self.wscale = np.sqrt(2) / np.sqrt(fan_in)
        
        self.normalized = normalized
        
        # Weight initialization
        _scaled_conv_weight_init(self)
    
    def forward(self, input):
        scaled_weights = self.weight * self.wscale
        x = nn.functional.conv2d(
            input, scaled_weights, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
        
        if self.normalized:
            x = nn.functional.normalize(x, dim=1)
        
        return x


class ScaledConvTranspose2d(nn.ConvTranspose2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, normalized=False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation
        )
        
        fan_in = in_channels * kernel_size ** 2
        self.wscale = np.sqrt(2) / np.sqrt(fan_in)
        
        self.normalized = normalized
        
        # Weight initialization
        _scaled_conv_weight_init(self)
    
    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input, output_size, self.stride,
            self.padding, self.kernel_size
        )
        
        scaled_weights = self.weight * self.wscale
        x = nn.functional.conv_transpose2d(
            input, scaled_weights, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation
        )
        
        if self.normalized:
            x = nn.functional.normalize(x, dim=1)
        
        return x


class ScaledConv3d(nn.Conv3d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, normalized=False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        fan_in = in_channels * kernel_size ** 3
        self.wscale = np.sqrt(2) / np.sqrt(fan_in)
        
        self.normalized = normalized
        
        # Weight initialization
        _scaled_conv_weight_init(self)
    
    def forward(self, input):
        scaled_weights = self.weight * self.wscale
        x = nn.functional.conv2d(
            input, scaled_weights, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
        
        if self.normalized:
            x = nn.functional.normalize(x, dim=1)
        
        return x


class ScaledConvTranspose3d(nn.ConvTranspose3d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, normalized=False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation
        )
        
        fan_in = in_channels * kernel_size ** 3
        self.wscale = np.sqrt(2) / np.sqrt(fan_in)
        
        self.normalized = normalized
        
        # Weight initialization
        _scaled_conv_weight_init(self)
    
    def forward(self, input, output_size=None):
        output_padding = self._output_padding(
            input, output_size, self.stride,
            self.padding, self.kernel_size
        )
        
        scaled_weights = self.weight * self.wscale
        x = nn.functional.conv_transpose2d(
            input, scaled_weights, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation
        )
        
        if self.normalized:
            x = nn.functional.normalize(x, dim=1)
        
        return x
