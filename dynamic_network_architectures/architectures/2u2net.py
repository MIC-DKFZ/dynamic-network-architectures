from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.RSU2 import RSUEncoder, RSUDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

__author__ = ["Stefano Petraccini"]
__email__ = ["stefano.petraccini@studio.unibo.it"]

class U2Net(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        blocks_nonlin: Union[None, Type[torch.nn.Module]] = None,
        blocks_nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        return_skips: bool = True,
        nonlin_first: bool = False,
        pool: str = "max",
        depth_per_stage: Union[int, List[int], Tuple[int, ...]] = None,
    ):
        super().__init__()
        # If we don't have a specific nonlinearity for blocks, use the default nonlin
        if blocks_nonlin is None:
            blocks_nonlin = nonlin
            blocks_nonlin_kwargs = nonlin_kwargs 

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
            blocks_nonlin,  # Passa blocks_nonlin
            blocks_nonlin_kwargs,
            return_skips,
            nonlin_first,
            pool,
            depth_per_stage=depth_per_stage
        )
        self.decoder = RSUDecoder(
            self.encoder,
            num_classes,
            deep_supervision = deep_supervision
        )

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