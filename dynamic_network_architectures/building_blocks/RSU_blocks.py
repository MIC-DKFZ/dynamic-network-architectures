import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple, Type, Optional

from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
    convert_conv_op_to_dim,
)

__author__ = ["Stefano Petraccini"]
__email__ = ["stefano.petraccini@studio.unibo.it"]

class RSUBlock(nn.Module):
    """
    Residual U-shaped (RSU) block for neural network architectures.
    
    This block implements a mini U-Net architecture within each level of a larger network.
    It consists of an encoder path that downsamples the input, a bottleneck layer,
    and a decoder path that upsamples back to the original resolution with skip connections.
    A residual connection is added between the input and output.
    
    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    mid_ch : int, optional
        Number of channels in the middle layers. If None, defaults to out_ch // 2.
    depth : int, default=4
        Depth of the RSU block, determining how many downsampling operations occur.
    conv_op : Type[nn.Module], default=nn.Conv2d
        Type of convolution operation.
    kernel_size : int or tuple or list, default=3
        Size of the convolving kernel.
    stride : int or tuple or list, default=1
        Stride of the convolution.
    bias : bool, default=True
        If True, adds a learnable bias to the convolution layers.
    nonlin : Type[nn.Module], optional, default=nn.ReLU
        Type of nonlinearity to use.
    norm_op : Type[nn.Module], optional, default=nn.BatchNorm2d
        Type of normalization to use.
    norm_op_kwargs : dict, optional
        Additional arguments for the normalization operation.
    dropout_op : Type[nn.Module], optional
        Type of dropout to use.
    dropout_op_kwargs : dict, optional
        Additional arguments for the dropout operation.
    nonlin_kwargs : dict, optional
        Additional arguments for the nonlinearity.
    pool : str, default="max"
        Type of pooling to use ("max" or "avg").
    nonlin_first : bool, default=False
        if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: Optional[int],
        depth: int = 4,
        conv_op: Type[nn.Module] = nn.Conv2d,
        kernel_size: Union[int, Tuple[int, ...], List[int]] = 3,
        stride: Union[int, Tuple[int, ...], List[int]] = 1,
        bias: bool = True,
        nonlin: Optional[Type[nn.Module]] = nn.ReLU,
        norm_op: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin_kwargs: Optional[dict] = None,
        pool: str = "max",
        nonlin_first: bool = False
    ):
        super().__init__()
        self.depth = depth
        self.nonlin_first = nonlin_first
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Here we define the dimensions based on the conv_op type
        dim = convert_conv_op_to_dim(conv_op)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        padding = [k // 2 for k in kernel_size]

        # Activation and dropout defaults
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        self.dropout = dropout_op(**(dropout_op_kwargs or {})) if dropout_op else nn.Identity()

        # Input convolution
        self.conv_in = conv_op(in_ch, out_ch, kernel_size, stride, padding=padding, bias=bias)
        self.norm_in = norm_op(out_ch, **norm_op_kwargs) if norm_op else nn.Identity()

        # Encoder path 
        pool_op = get_matching_pool_op(conv_op, pool_type=pool)
        self.enc_norms = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                conv_op(out_ch if i == 0 else mid_ch, mid_ch, kernel_size, stride, padding=padding, bias=bias)
            )
            self.enc_norms.append(
                norm_op(mid_ch, **norm_op_kwargs) if norm_op else nn.Identity()
            )
            self.pools.append(pool_op(kernel_size=2, stride=2))

        # Bottom
        self.bottom = conv_op(mid_ch, mid_ch, kernel_size, stride, padding=padding, bias=bias)
        self.norm_bottom = norm_op(mid_ch, **norm_op_kwargs) if norm_op else nn.Identity()

        # Decoder path
        self.dec_norms = nn.ModuleList()
        for i in range(depth):
            skip_ch = out_ch if i == depth - 1 else mid_ch
            in_ch_dec = mid_ch + skip_ch
            out_ch_dec = mid_ch if i < depth - 1 else out_ch
            self.decoders.append(
                conv_op(in_ch_dec, out_ch_dec, kernel_size, stride, padding=padding, bias=bias)
            )
            self.dec_norms.append(
                norm_op(out_ch_dec, **norm_op_kwargs) if norm_op else nn.Identity()
            )

    def _apply_block(self, conv, norm, nonlin, x):
        """
        Apply a convolutional block with normalization and nonlinearity.
        
        Parameters
        ----------
        conv : nn.Module
            Convolution layer.
        norm : nn.Module
            Normalization layer.
        nonlin : nn.Module
            Nonlinearity function.
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Output tensor after applying convolution, normalization, and nonlinearity.
            
        Notes
        -----
        The order of operations depends on self.nonlin_first:
        - If True: conv -> nonlin -> norm
        - If False: conv -> norm -> nonlin
        
        For InstanceNorm with small spatial dimensions (prod(spatial_dims) <= 1),
        normalization is skipped to avoid numerical issues.
        """
        x = conv(x)
        # Check if the norm is an instance of InstanceNorm and if the spatial dimensions are too small
        is_instancenorm = isinstance(norm, (nn.InstanceNorm2d, nn.InstanceNorm3d))
        # The spatial dimensions are all after the second (batch, channels, ...)
        spatial_dims = x.shape[2:]
        too_small_for_instancenorm = np.prod(spatial_dims) <= 1

        if not isinstance(norm, nn.Identity) and not (is_instancenorm and too_small_for_instancenorm):
            if self.nonlin_first:
                x = nonlin(x)
                x = norm(x)
            else:
                x = norm(x)
                x = nonlin(x)
        else:
            # if norm is Identity or InstanceNorm with too small spatial dimensions
            x = nonlin(x)
    
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RSU block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_ch, *spatial_dims).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_ch, *spatial_dims).
            The spatial dimensions remain the same as the input.
        """
        x_in = self._apply_block(self.conv_in, self.norm_in, self.nonlin, x)
        x_in = self.dropout(x_in)
        enc_feats = [x_in]
        xi = x_in

        # Encoder
        for enc, norm, pool in zip(self.encoders, self.enc_norms, self.pools):
            xi = self._apply_block(enc, norm, self.nonlin, xi)
            xi = self.dropout(xi)
            enc_feats.append(xi)
            if all(s > 1 for s in xi.shape[2:]):
                xi = pool(xi)

        # Bottleneck
        xb = self._apply_block(self.bottom, self.norm_bottom, self.nonlin, xi)
        xb = self.dropout(xb)

        # Decoder
        xu = xb
        for i, (dec, norm) in enumerate(zip(self.decoders, self.dec_norms)):
            skip = enc_feats[-(i+2)]
            mode = 'trilinear' if xu.dim() == 5 else 'bilinear'
            xu = F.interpolate(xu, size=skip.shape[2:], mode=mode, align_corners=False)
            xu = torch.cat([xu, skip], dim=1)
            xu = self._apply_block(dec, norm, self.nonlin, xu)
            xu = self.dropout(xu)
        if xu.shape[2:] != x_in.shape[2:]:
            mode = 'trilinear' if xu.dim() == 5 else 'bilinear'
            xu = F.interpolate(xu, size=x_in.shape[2:], mode=mode, align_corners=False)
        return xu + x_in


class RSUEncoder(nn.Module):
    """
    Encoder part of the network using RSU blocks.
    
    This encoder creates a series of RSU blocks that progressively reduce the spatial
    dimensions of the input while increasing the number of channels.
    
    Parameters
    ----------
    input_channels : int
        Number of input channels.
    n_stages : int
        Number of stages (RSU blocks) in the encoder.
    features_per_stage : List[int]
        Number of output channels for each stage.
    conv_op : Type[nn.Module]
        Type of convolution operation.
    kernel_sizes : List[Union[int, Tuple[int, ...], List[int]]]
        Kernel sizes for each stage.
    strides : List[Union[int, Tuple[int, ...], List[int]]]
        Strides for each stage.
    n_conv_per_stage : Union[int, List[int], Tuple[int, ...]]
        Number of convolutions per stage.
    conv_bias : bool
        If True, adds a learnable bias to the convolution layers.
    norm_op : Optional[Type[nn.Module]]
        Type of normalization to use.
    norm_op_kwargs : Optional[dict]
        Additional arguments for the normalization operation.
    dropout_op : Optional[Type[nn.Module]]
        Type of dropout to use.
    dropout_op_kwargs : Optional[dict]
        Additional arguments for the dropout operation.
    nonlin : Optional[Type[nn.Module]]
        Type of nonlinearity to use.
    nonlin_kwargs : Optional[dict]
        Additional arguments for the nonlinearity.
    return_skips : bool, default=True
        If True, returns intermediate feature maps (skips) for U-Net-like architectures.
    nonlin_first : bool, default=False
        If True, applies nonlinearity before normalization.
    pool : str, default="max"
        Type of pooling to use ("max" or "avg").
    depth_per_stage : Optional[List[int]], default=None
        Depth of each RSU block. If None, all blocks use depth=4.
    blocks_nonlin : Optional[Type[nn.Module]], default=None
        Specific nonlinearity for RSU blocks. If None, uses the same as nonlin.
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[nn.Module],
        kernel_sizes: List[Union[int, Tuple[int, ...], List[int]]],
        strides: List[Union[int, Tuple[int, ...], List[int]]],
        conv_bias: bool,
        norm_op: Optional[Type[nn.Module]],
        norm_op_kwargs: Optional[dict],
        dropout_op: Optional[Type[nn.Module]],
        dropout_op_kwargs: Optional[dict],
        nonlin: Optional[Type[nn.Module]],
        nonlin_kwargs: Optional[dict],
        return_skips: bool = True,
        nonlin_first: bool = False,
        pool: str = "max",
        depth_per_stage: Optional[List[int]] = None,
        blocks_nonlin: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.return_skips = return_skips
        self.features_per_stage = features_per_stage
        self.depth_per_stage = depth_per_stage if depth_per_stage is not None else [4] * n_stages
        self.bias = conv_bias
        self.blocks_nonlin = blocks_nonlin if blocks_nonlin is not None else nonlin
        self.nonlin_first = nonlin_first
        self.stages = nn.ModuleList()
        prev_ch = input_channels
        for i in range(n_stages):
            depth = self.depth_per_stage[i]
            mid_ch = features_per_stage[i] // 2
            self.stages.append(
                RSUBlock(
                    in_ch=prev_ch,
                    out_ch=features_per_stage[i],
                    mid_ch=mid_ch,
                    depth=depth,
                    conv_op=conv_op,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    bias=conv_bias,
                    nonlin=self.blocks_nonlin,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin_kwargs=nonlin_kwargs,
                    pool=pool,
                    nonlin_first=nonlin_first
                )
            )
            prev_ch = features_per_stage[i]

    def forward(self, x: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the RSU encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_channels, *spatial_dims).
            
        Returns
        -------
        Union[List[torch.Tensor], torch.Tensor]
            If return_skips is True:
                List of feature maps from each stage, for use in a decoder with skip connections.
            Otherwise:
                Output tensor from the final stage.
        """
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        if self.return_skips:
            return skips
        return x
    
    def compute_conv_feature_map_size(self, input_size: List[int]) -> int:
        """
        Compute the number of parameters in the convolutional feature maps.
        
        Parameters
        ----------
        input_size : List[int]
            Spatial dimensions of the input tensor (excluding batch and channel dimensions).
            
        Returns
        -------
        int
            Number of parameters in the convolutional feature maps.
        """
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class RSUDecoder(nn.Module):
    """
    Decoder part of a network using RSU blocks.
    
    This decoder creates a series of RSU blocks that progressively increase the spatial
    dimensions of the input while decreasing the number of channels. It uses skip
    connections from a corresponding encoder.
    
    Parameters
    ----------
    encoder : RSUEncoder
        The encoder to get skip connections from and share parameters with.
    num_classes : int
        Number of output classes for the final segmentation layer.
    deep_supervision : bool, default=False
        If True, returns intermediate outputs for deep supervision.
    blocks_nonlin : Optional[Type[nn.Module]], default=None
        Specific nonlinearity for RSU blocks. If None, uses the same as in the encoder.
    nonlin_first : bool, default=False
        If True, applies nonlinearity before normalization.
    """
    def __init__(
        self,
        encoder: RSUEncoder,
        num_classes: int,
        deep_supervision: bool = False,
        blocks_nonlin: Optional[Type[nn.Module]] = None,
        nonlin_first: bool = False
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.encoder = encoder
        self.blocks_nonlin = blocks_nonlin if blocks_nonlin is not None else encoder.blocks_nonlin
        self.nonlin_first = nonlin_first

        features_per_stage = encoder.features_per_stage
        n_stages = len(features_per_stage)
        for i in range(n_stages-1, 0, -1):
            self.stages.append(
                RSUBlock(
                    in_ch = features_per_stage[i]+features_per_stage[i-1],
                    out_ch = features_per_stage[i-1],
                    mid_ch = features_per_stage[i-1]//2,
                    depth = encoder.depth_per_stage[i-1],
                    conv_op = encoder.conv_op,
                    kernel_size = encoder.kernel_sizes[i-1],
                    stride = 1,
                    bias = encoder.bias,
                    nonlin = self.blocks_nonlin,
                    norm_op = encoder.norm_op,
                    norm_op_kwargs = encoder.norm_op_kwargs,
                    dropout_op = encoder.dropout_op,
                    dropout_op_kwargs = encoder.dropout_op_kwargs,
                    nonlin_kwargs = encoder.nonlin_kwargs,
                    nonlin_first = nonlin_first
                )
            )
        # Segmentation heads: one per output for deep supervision, or just one if not
        self.seg_layers = nn.ModuleList([
            encoder.conv_op(features_per_stage[i-1], num_classes, 1, 1, 0, bias=True)
            for i in range(n_stages-1, 0, -1)
        ])

    def forward(self, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the RSU decoder.
        
        Parameters
        ----------
        skips : List[torch.Tensor]
            List of feature maps from the encoder, in order from input to bottleneck.
            
        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            If deep_supervision is False:
                Final output tensor with shape (batch_size, num_classes, *spatial_dims).
            Otherwise:
                List of output tensors at different resolutions for deep supervision.
        """
        x = skips[-1]
        outputs = []
        for i, stage in enumerate(self.stages):
            mode = 'trilinear' if x.dim() == 5 else 'bilinear'
            x = F.interpolate(x, size=skips[-(i+2)].shape[2:], mode=mode, align_corners=False)
            x = torch.cat([x, skips[-(i+2)]], dim=1)
            x = stage(x)
            # Apply segmentation head to get num_classes channels
            seg_x = self.seg_layers[i](x)
            outputs.append(seg_x)
        # Order the outputs from the last to the first 
        outputs = outputs[::-1]
        if not self.deep_supervision:
            r = outputs[0]
        else:
            r = outputs
        return r

    def compute_conv_feature_map_size(self, input_size: List[int]) -> int:
        """
        Compute the number of parameters in the convolutional feature maps.
        
        Parameters
        ----------
        input_size : List[int]
            Spatial dimensions of the input tensor (excluding batch and channel dimensions).
            
        Returns
        -------
        int
            Number of parameters in the convolutional feature maps.
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        assert len(skip_sizes) == len(self.stages)
        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output