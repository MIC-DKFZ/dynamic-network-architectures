from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0

__author__ = ["Stefano Petraccini"]
__email__ = ["stefano.petraccini@studio.unibo.it"]

class RSUBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        mid_ch,
        out_ch,
        depth,
        conv_op: Type[_ConvNd],
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.conv_op = conv_op
        self.rebnconvin = ConvDropoutNormReLU(
            conv_op, in_ch, out_ch, 3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
        )
        pool_cls = nn.MaxPool3d if conv_op == nn.Conv3d else nn.MaxPool2d
        self.pool = pool_cls(2, stride=2, ceil_mode=True)
        # Encoder
        self.encoders = nn.ModuleList()
        self.encoders.append(
            ConvDropoutNormReLU(
                conv_op, out_ch, mid_ch, 3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            )
        )
        for _ in range(1, depth):
            self.encoders.append(
                ConvDropoutNormReLU(
                    conv_op, mid_ch, mid_ch, 3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                )
            )
        # Decoder
        self.decoders = nn.ModuleList()
        for _ in range(depth-1):
            self.decoders.append(
                ConvDropoutNormReLU(
                    conv_op, mid_ch * 2, mid_ch, 3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                )
            )
        self.decoders.append(
            ConvDropoutNormReLU(
                conv_op, mid_ch + out_ch, out_ch, 3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            )
        )

    def forward(self, x):
        sizes = [x.shape[-3:] if x.dim() == 5 else x.shape[-2:]]
        x_in = self.rebnconvin(x)
        enc_feats = [self.encoders[0](x_in)]
        x = self.pool(enc_feats[0])
        sizes.append(x.shape[-3:] if x.dim() == 5 else x.shape[-2:])
        # Encoder path
        for i in range(1, self.depth):
            enc = self.encoders[i](x)
            enc_feats.append(enc)
            if i < self.depth - 1:
                x = self.pool(enc)
                sizes.append(x.shape[-3:] if x.dim() == 5 else x.shape[-2:])
            else:
                x = enc
        # Decoder path
        for i in range(self.depth-1, 0, -1):
            mode = 'trilinear' if x.dim() == 5 else 'bilinear'
            x = nn.functional.interpolate(x, size=sizes[i], mode=mode, align_corners=False)
            enc_up = nn.functional.interpolate(enc_feats[i-1], size=x.shape[-3:] if x.dim() == 5 else x.shape[-2:], mode=mode, align_corners=False)
            x = torch.cat([x, enc_up], dim=1)
            x = self.decoders[self.depth-1-i](x)
        mode = 'trilinear' if x.dim() == 5 else 'bilinear'
        x = nn.functional.interpolate(x, size=sizes[0], mode=mode, align_corners=False)
        x_in_up = nn.functional.interpolate(x_in, size=x.shape[-3:] if x.dim() == 5 else x.shape[-2:], mode=mode, align_corners=False)
        x = torch.cat([x, x_in_up], dim=1)
        x = self.decoders[-1](x)
        return x + x_in_up


class U2Net(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        mid_channels: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        rsu_depths: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(mid_channels, int):
            mid_channels = [mid_channels] * n_stages
        if isinstance(rsu_depths, int):
            rsu_depths = [rsu_depths] * n_stages

        self.deep_supervision = deep_supervision
        self.conv_op = conv_op

        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = input_channels
        for i in range(n_stages):
            self.encoder.append(
                RSUBlock(
                    in_ch, mid_channels[i], features_per_stage[i], rsu_depths[i],
                    conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, conv_bias
                )
            )
            in_ch = features_per_stage[i]

        # Decoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(1, n_stages)):
            self.decoder.append(
                RSUBlock(
                    features_per_stage[i] + features_per_stage[i-1],
                    mid_channels[i-1],
                    features_per_stage[i-1],
                    rsu_depths[i-1],
                    conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, conv_bias
                )
            )

        self.final = conv_op(features_per_stage[0], num_classes, 1)

    def forward(self, x):
        skips = []
        sizes = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            sizes.append(x.shape[-3:] if x.dim() == 5 else x.shape[-2:])
            pool = nn.functional.max_pool3d if x.dim() == 5 else nn.functional.max_pool2d
            x = pool(x, 2, 2, ceil_mode=True)
        x = skips.pop()
        for idx, dec in enumerate(self.decoder):
            mode = 'trilinear' if x.dim() == 5 else 'bilinear'
            x = nn.functional.interpolate(x, size=sizes[-(idx+2)], mode=mode, align_corners=False)
            x = torch.cat([x, skips.pop()], dim=1)
            x = dec(x)
        x = self.final(x)
        return x

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == "__main__":
    # Esempio 2D
    model2d = U2Net(
        input_channels=3,
        n_stages=4,
        features_per_stage=[64, 128, 256, 512],
        mid_channels=[32, 64, 128, 256],
        num_classes=2,
        rsu_depths=[2, 3, 4, 4],
        conv_op=nn.Conv2d,
        conv_bias=False,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs=None,
        deep_supervision=True,
    )
    x2d = torch.randn(1, 3, 256, 256)
    out2d = model2d(x2d)
    print("2D Output shape:", out2d.shape)

    # Esempio 3D
    model3d = U2Net(
        input_channels=1,
        n_stages=4,
        features_per_stage=[16, 32, 64, 128],
        mid_channels=[8, 16, 32, 64],
        num_classes=2,
        rsu_depths=[2, 3, 4, 4],
        conv_op=nn.Conv3d,
        conv_bias=False,
        norm_op=nn.BatchNorm3d,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs=None,
        deep_supervision=False,
    )
    x3d = torch.randn(1, 1, 32, 64, 64)
    out3d = model3d(x3d)
    print("3D Output shape:", out3d.shape)