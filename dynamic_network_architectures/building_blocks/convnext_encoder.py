import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.init import trunc_normal_
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.convnext_block import ConvNextBlock
from dynamic_network_architectures.building_blocks.layer_norm import LayerNorm
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, \
    get_matching_instancenorm
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class ConvNextEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 block: Type[ConvNextBlock] = ConvNextBlock,
                 return_skips: bool = False,
                 stem: str = 'default',
                 drop_path_rate: float = 0.
                 ):
        """
        Careful! This class behaves differently than the ResidualEncoder in that the first skip is returned by the stem
        (if available). This is because we have an expansion of 4 in the convnext block which means we may not want to
        use it at full feature map resolution (use stem instead)!
        If stem is nn.Module, it MUST produce features_per_stage[0] output channels!

        If stem is None, the default stem from the original implementation is used: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
        (conv with kernel size 4 + stride 4 -> layernorm)
        """
        super().__init__()
        print('INITIALIZATION')
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        self.stages = []
        if stem == 'default':
            # default stem ignores kernel_sizes[0]!
            stem = nn.Sequential(
                conv_op(input_channels, features_per_stage[0], kernel_size=kernel_sizes[0], stride=strides[0],
                        padding=[(i - 1) // 2 for i in maybe_convert_scalar_to_list(conv_op, kernel_sizes[0])]),
                LayerNorm(features_per_stage[0], eps=1e-6, data_format="channels_first")
            )
            stem.__setattr__('compute_conv_feature_map_size', lambda input_size: features_per_stage[0] * np.prod([i / j] for i, j in zip(input_size, maybe_convert_scalar_to_list(conv_op, strides[0]))))
        elif stem == 'stacked_convs':
            stem = StackedConvBlocks(n_blocks_per_stage[0], conv_op, input_channels, features_per_stage[0], kernel_sizes[0], strides[0], True,
                                     get_matching_instancenorm(conv_op), {'eps': 1e-5, 'affine': True},
                                     None, {}, nn.LeakyReLU, {'inplace': True})
        else:
            raise ValueError('unknown input for stem')
        self.stages.append(stem)
        input_channels = features_per_stage[0]

        # now build the network
        # start at 1 because stem is the first stage here
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(n_blocks_per_stage[1:]))]
        cur = 0
        for s in range(1, n_stages):
            # downsample in convnext is done by conv -> layernorm
            downsample_op = nn.Sequential(
                conv_op(input_channels, features_per_stage[s], kernel_size=strides[s], stride=strides[s]),
                LayerNorm(features_per_stage[s], data_format='channels_first')
            )
            convnext_blocks = nn.Sequential(*[
                block(conv_op, features_per_stage[s], kernel_sizes[s], dp_rates[cur + j]) for j in range(n_blocks_per_stage[s])
            ])
            cur += n_blocks_per_stage[s]
            self.stages.append(nn.Sequential(downsample_op, convnext_blocks))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*self.stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (self.conv_op, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = self.stages[0].compute_conv_feature_map_size(input_size)
        input_size = [i // j for i, j in zip(input_size, self.strides[0])]

        for s in range(1, len(self.stages)):
            # each stage consists of a downsampling operation followed by a block
            output_shape = [i // j for i, j in zip(input_size, self.strides[s])]
            output += np.prod(output_shape) * self.stages[s][0][0].out_channels
            input_size = output_shape
            for block in self.stages[s][1]:
                output += block.compute_conv_feature_map_size(input_size)
        return output


class ConvNextEncoder_standardconvblockstart(ConvNextEncoder):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 block: Type[ConvNextBlock] = ConvNextBlock,
                 return_skips: bool = False,
                 stem: str = 'default',
                 drop_path_rate: float = 0.
                 ):
        """
        Careful! This class behaves differently than the ResidualEncoder in that the first skip is returned by the stem
        (if available). This is because we have an expansion of 4 in the convnext block which means we may not want to
        use it at full feature map resolution (use stem instead)!
        If stem is nn.Module, it MUST produce features_per_stage[0] output channels!

        If stem is None, the default stem from the original implementation is used: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
        (conv with kernel size 4 + stride 4 -> layernorm)
        """
        super().__init__(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage, block, return_skips, stem, drop_path_rate)

        # we need to exchange stages[0]

        if stem == 'default':
            # default stem ignores kernel_sizes[0]!
            stem = nn.Sequential(
                conv_op(input_channels, features_per_stage[0], kernel_size=kernel_sizes[0], stride=strides[0],
                        padding=[(i - 1) // 2 for i in maybe_convert_scalar_to_list(conv_op, kernel_sizes[0])]),
                LayerNorm(features_per_stage[0], eps=1e-6, data_format="channels_first")
            )
            stem.__setattr__('compute_conv_feature_map_size', lambda input_size: features_per_stage[0] * np.prod([i / j] for i, j in zip(input_size, maybe_convert_scalar_to_list(conv_op, strides[0]))))
        elif stem == 'stacked_convs':
            stem = StackedConvBlocks(n_blocks_per_stage[0], conv_op, input_channels, features_per_stage[0], kernel_sizes[0], strides[0], True,
                                     get_matching_instancenorm(conv_op), {'eps': 1e-5, 'affine': True},
                                     None, {}, nn.LeakyReLU, {'inplace': True})
        else:
            raise ValueError('unknown input for stem')
        self.stages[0] = stem


if __name__ == '__main__':
    torch.set_num_threads(16)
    data = torch.rand((1, 3, 128, 128, 128))

    model = ConvNextEncoder(3, 4, [4, 8, 16, 32], nn.Conv3d, [3, 7, 7, 7], [1, 2, 2, 2], [2, 2, 2, 2],
                            return_skips=True, stem='stacked_convs')
    out = model(data)
    import hiddenlayer as hl

    g = hl.build_graph(model, data,
                       transforms=None)
    g.save("network_architecture.pdf")
    del g

    model = ConvNextEncoder(3, 6, [32, 64, 128, 256, 320, 320], nn.Conv3d, [3, 7, 7, 7, 7, 7], [1, 2, 2, 2, 2, 2], [2, 3, 3, 5, 4, 3],
                            return_skips=True, stem='stacked_convs')
    input_shape = (128, 128, 128)
    print(model.compute_conv_feature_map_size(input_shape))  # 363044864

    model = ConvNextEncoder(3, 7, [32, 64, 128, 256, 512, 512, 512], nn.Conv2d, [3, 7, 7, 7, 7, 7, 7], [1, 2, 2, 2, 2, 2, 2], [2, 3, 3, 3, 3, 3, 3],
                            return_skips=True, stem='stacked_convs')
    input_shape = (512, 512)
    print(model.compute_conv_feature_map_size(input_shape))  # 97058816

