from torch import nn
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.RSU2 import RSUEncoder, RSUDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp

#TODO: va fixata la deep supervision
class U2Net(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        num_classes: int, 
        n_stages: int,
        features_per_stage: list,
        conv_op,
        kernel_sizes: list,
        strides: list,
        n_conv_per_stage: int,
        conv_bias: bool,
        norm_op,
        norm_op_kwargs: dict,
        dropout_op,
        dropout_op_kwargs: dict,
        nonlin, ########## TODO: ideally we want to be able to define different nonlinearities for both RSUBlock and "general" nonlinearity (the one used by nnUNet)
        nonlin_kwargs: dict,
        deep_supervision: bool = True,
        return_skips: bool = True,
        nonlin_first: bool = False,
        pool: str = "max",
        depth_per_stage: list = None,
        **kwargs
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = RSUEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips,
            nonlin_first,
            pool,
            depth_per_stage = depth_per_stage
        )
        self.decoder = RSUDecoder(self.encoder, num_classes, deep_supervision) 
        
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
    
    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )