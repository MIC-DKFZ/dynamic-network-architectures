from typing import Type, Union, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.layer_norm import LayerNorm
from dynamic_network_architectures.building_blocks.regularization import DropPath


class ConvNextBlock(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 channels: int,
                 dw_conv_kernel_size: Union[int, List[int], Tuple[int, ...]] = 7,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6
                 ):
        r"""
        Adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

        ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()
        dw_conv_kernel_size = maybe_convert_scalar_to_list(conv_op, dw_conv_kernel_size)
        self.channels = channels
        dw_conv_kernel_size = (dw_conv_kernel_size, ) if not isinstance(dw_conv_kernel_size, (list, tuple)) else dw_conv_kernel_size
        self.dwconv = conv_op(channels, channels, kernel_size=dw_conv_kernel_size,
                                padding=[(i - 1) // 2 for i in dw_conv_kernel_size], groups=channels, stride=1)  # depthwise conv
        self.norm = LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, 4 * channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((channels)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        orig_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) if len(x.shape) == 4 else x.permute(0, 2, 3, 4, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) if len(x.shape) == 4 else x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = orig_x + self.drop_path(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        """
        we have 3 convs in here. We ignore the activations. We do not change feature map sizes
        """
        return 3 * np.prod(input_size) * self.channels
