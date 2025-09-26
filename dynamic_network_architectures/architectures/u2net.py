from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.RSU_blocks import RSUEncoder, RSUDecoder
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

__author__ = ["Stefano Petraccini"]
__email__ = ["stefano.petraccini@studio.unibo.it"]

class U2Net(AbstractDynamicNetworkArchitectures):
    """
    U2Net architecture: nested U-structure with RSU blocks.

    Parameters
    ----------
    input_channels : int
        Number of input channels (for example, 1 for grayscale, 3 for RGB).
    n_stages : int
        Number of encoder/decoder stages.
    features_per_stage : Union[int, List[int], Tuple[int, ...]]
        Number of output channels per stage. May be a single integer (uniform across
        stages) or a list/tuple providing one value per stage.
    conv_op : Type[_ConvNd]
        Convolution operator to use (for example, "nn.Conv2d" or "nn.Conv3d").
    kernel_sizes : Union[int, List[int], Tuple[int, ...]]
        Kernel sizes for each stage (single value or per-stage list/tuple).
    strides : Union[int, List[int], Tuple[int, ...]]
        Strides for each stage (single value or per-stage list/tuple).
    num_classes : int
        Number of output classes for the final segmentation head.
    conv_bias : bool, default=False
        If "True", add a learnable bias to convolution layers.
    norm_op : Union[None, Type[nn.Module]], default=None
        Normalization layer class to use. If "None", no normalization is applied.
    norm_op_kwargs : dict, default=None
        Keyword arguments for "norm_op".
    dropout_op : Union[None, Type[_DropoutNd]], default=None
        Dropout layer class to use. If "None", dropout is not applied.
    dropout_op_kwargs : dict, default=None
        Keyword arguments for "dropout_op".
    nonlin : Union[None, Type[torch.nn.Module]], default=None
        Nonlinearity layer class to use. If "None", no nonlinearity is applied.
    nonlin_kwargs : dict, default=None
        Keyword arguments for "nonlin".
    blocks_nonlin : Union[None, Type[torch.nn.Module]], default=None
        Specific nonlinearity for RSU blocks. If "None", falls back to "nonlin".
    blocks_nonlin_kwargs : dict, default=None
        Keyword arguments for the RSU block nonlinearity.
    deep_supervision : bool, default=False
        If "True", return intermediate outputs for deep supervision.
    return_skips : bool, default=True
        If "True", return intermediate feature maps (skips) from the encoder.
    nonlin_first : bool, default=False
        If "True", apply nonlinearity before normalization in conv blocks.
    pool : str, default="max"
        Pooling type used in pooling RSU blocks (""max"" or ""avg"").
    depth_per_stage : Union[int, List[int], Tuple[int, ...]], default=None
        Depth of each RSU block. If "None", a default depth is used per stage.

    Notes
    -----
    - This implementation composes an "RSUEncoder" and "RSUDecoder". Some stages
      may use dilated RSU blocks (RSU-4F) to preserve spatial resolution.
    - Outputs are logits; apply a nonlinearity (for example, Sigmoid/Softmax) outside if needed.

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
                """
                Forward pass of U2Net.

                Parameters
                ----------
                x : torch.Tensor
                        Input tensor of shape "(batch_size, input_channels, *spatial_dims)".

                Returns
                -------
                torch.Tensor or List[torch.Tensor]
                        - If "deep_supervision" is False: the final segmentation logits with shape
                            "(batch_size, num_classes, *spatial_dims)".
                        - If "deep_supervision" is True: a list of segmentation logits at different
                            resolutions (highest resolution first).

                Notes
                -----
                Outputs are raw logits. Apply activation (for example, Sigmoid/Softmax) depending
                on your loss or evaluation setup.
                """
                skips = self.encoder(x)
                return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        """
        Estimate the total convolutional feature map size traversed by the model.

        Parameters
        ----------
        input_size : List[int] or Tuple[int, ...]
            Spatial dimensions of the input (exclude batch and channel), for example "(H, W)"
            or "(D, H, W)".

        Returns
        -------
        int
            Proxy measure of feature map elements processed by convolutions across encoder and decoder.

        Notes
        -----
        - This is used as an approximate VRAM indicator; it is not the number of learnable parameters.
        - Pass only spatial dims. For example, use "(H, W)" not "(B, C, H, W)".
        """
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )