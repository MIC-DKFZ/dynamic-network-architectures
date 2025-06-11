import torch
import torch.nn as nn
import math
from dynamic_network_architectures.building_blocks.helper import get_matching_pool_op, convert_dim_to_conv_op, get_matching_batchnorm

def _upsample_like(x, dim, size):
    return nn.Upsample(size=size, mode='bilinear' if dim == 2 else 'trilinear', align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, dim, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()
        
        self.conv_s1 = convert_dim_to_conv_op(dim)(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = get_matching_batchnorm(dimension=dim)(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, dim, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.dim = dim
        self.height = height
        self.dilated = dilated
        self._make_layers(dim, height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, self.dim, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, dim, height, in_ch, mid_ch, out_ch, dilated=False):
        dim = self.dim
        self.add_module('rebnconvin', REBNCONV(dim, in_ch, out_ch))
        self.add_module('downsample', get_matching_pool_op(dimension=dim, pool_type="max")(2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', REBNCONV(dim, out_ch, mid_ch))
        self.add_module(f'rebnconv1d', REBNCONV(dim, mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(dim, mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(dim, mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(dim, mid_ch, mid_ch, dilate=dilate))
