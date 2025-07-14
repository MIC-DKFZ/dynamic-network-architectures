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
    """
    U2Net architecture: nested u-structure for salient object detection.

    Parameters
    ----------
    input_channels : int
        Number of input channels (e.g., 1 for grayscale, 3 for RGB).
    n_stages : int
        Number of stages in the encoder.
    features_per_stage : Union[int, List[int], Tuple[int, ...]]
        Number of features (channels) per stage. Can be a single integer for uniform features or
        a list/tuple of integers for different features per stage.
    conv_op : Type[_ConvNd]
        Type of convolution operation to use (e.g., nn.Conv2d, nn.Conv3d).
    kernel_sizes : Union[int, List[int], Tuple[int, ...]]
        Kernel sizes to use for the convolutional layers.
    strides : Union[int, List[int], Tuple[int, ...]]
        Strides for the convolutional layers.
    n_conv_per_stage : Union[int, List[int], Tuple[int, ...]]
        Number of convolutional layers per stage. Can be a single integer for uniform number of convolutions or
        a list/tuple of integers for different numbers of convolutions per stage.
    num_classes : int
        Number of output classes for the segmentation task.
    conv_bias : bool, default=False
        If True, adds a learnable bias to the convolution layers.
    norm_op : Union[None, Type[nn.Module]], default=None
        Type of normalization to use (e.g., nn.BatchNorm2d, nn.InstanceNorm2d).
        If None, no normalization is applied.
    norm_op_kwargs : dict, default=None
        Additional arguments for the normalization operation.
    dropout_op : Union[None, Type[_DropoutNd]], default=None
        Type of dropout to use (e.g., nn.Dropout2d, nn.Dropout3d).
        If None, no dropout is applied.
    dropout_op_kwargs : dict, default=None
        Additional arguments for the dropout operation.
    nonlin : Union[None, Type[torch.nn.Module]], default=None
        Type of nonlinearity to use (e.g., nn.ReLU, nn.LeakyReLU).
        If None, no nonlinearity is applied.
    nonlin_kwargs : dict, default=None
        Additional arguments for the nonlinearity.
    blocks_nonlin : Union[None, Type[torch.nn.Module]], default=None
        Specific nonlinearity for RSU blocks. If None, uses the same as nonlin.
    blocks_nonlin_kwargs : dict, default=None
        Additional arguments for the RSU block nonlinearity.
    deep_supervision : bool, default=False
        If True, returns intermediate outputs for deep supervision during training.
    return_skips : bool, default=True
        If True, returns intermediate feature maps (skips) from the encoder for U-Net-like architectures.
    nonlin_first : bool, default=False
        If True, applies nonlinearity before normalization.
    pool : str, default="max"
        Type of pooling to use ("max" or "avg").
    depth_per_stage : Union[int, List[int], Tuple[int, ...]], default=None
        Depth of each RSU block. If None, all blocks use the default depth.
    
    References
    ----------
    .. [1] Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R., & Jagersand, M. (2020).
        U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection.
        Pattern Recognition, 106, 107404.
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], ###TOGLIERE: non ha senso tenerlo, che ci faccio? mpm voglio mica più di un blocco rsu per ogni stage. anche perché è già enorme così.
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
        depth_per_stage: Union[int, List[int], Tuple[int, ...]] = None
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
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            blocks_nonlin,  
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