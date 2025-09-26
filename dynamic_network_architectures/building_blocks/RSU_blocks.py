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

    Notes
    -----
    - Internally downsamples/upsamples features via pooling and interpolation, but the
        block output has the same spatial size as the input (residual connection).
    - Pooling is skipped when spatial dimensions are too small ("> 1" check per axis).
    - Normalization may be effectively skipped on extremely small feature maps by the
        block internals to avoid numerical issues.

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
        #dim = convert_conv_op_to_dim(conv_op)
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        self.strides = maybe_convert_scalar_to_list(conv_op, stride)
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
        Output tensor of shape "(batch_size, out_ch, *spatial_dims)" with the same
        spatial size as the input.

        Notes
        -----
        - Uses interpolation to align skip connections and maintain spatial size.
        - Output adds a residual connection "(x + F(x))".
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

    Notes
    -----
    This is a proxy used for memory/VRAM estimation and does not include parameters,
    only the feature map element counts traversed by convolutions.
        """
        output = np.int64(0)
        
        # Input convolution feature map
        out_ch = self.conv_in.out_channels
        output += np.prod([out_ch, *input_size], dtype=np.int64)
        
        # Encoder path feature maps
        enc_sizes = [input_size]
        curr_size = input_size
        
        # Always ensure we have enough encoder sizes even when input is small
        for i in range(self.depth):
            # Account for pooling if dimensions are large enough
            if all(s > 1 for s in curr_size):
                curr_size = [s // 2 for s in curr_size]  # Pool by factor of 2
            else:
                # If dimensions are too small, don't reduce further
                curr_size = curr_size
                
            enc_sizes.append(curr_size)
            # Add encoder feature map size
            out_ch = self.encoders[i].out_channels
            output += np.prod([out_ch, *curr_size], dtype=np.int64)
        
        # Bottleneck feature map
        out_ch = self.bottom.out_channels
        output += np.prod([out_ch, *curr_size], dtype=np.int64)
        
        # Decoder path feature maps
        for i in range(self.depth):
            # Skip connection level (safely access with bounds checking)
            idx = min(i+2, len(enc_sizes))
            skip_size = enc_sizes[-idx]
            # Feature map after concatenation and convolution
            out_ch = self.decoders[i].out_channels
            output += np.prod([out_ch, *skip_size], dtype=np.int64)
        
        return output

class RSUdilatedBlock(nn.Module):
    """
    Residual U-shaped (RSU) dilated block (RSU-4F).

    This block implements a mini U-Net that replaces pooling/upsampling with
    dilated convolutions so that all intermediate feature maps preserve the
    input spatial resolution. Dilation rates increase along the encoder path
    and decrease along the decoder path. Dilation is capped at runtime based on
    the current input size to avoid invalid effective kernel sizes.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    mid_ch : int, optional
        Number of channels in the intermediate layers. If "None", defaults to
        "out_ch // 2".
    depth : int, default=4
        Number of internal encoder/decoder levels inside the block.
    conv_op : Type[nn.Module], default=nn.Conv2d
        Convolution operator class to use (e.g., 2D or 3D variant).
    kernel_size : int or tuple of int, default=3
        Convolution kernel size for all internal convolutions.
    stride : int or tuple of int, default=1
        Stride applied by the input convolution only; internal layers use stride "1".
    bias : bool, default=True
        If "True", adds a learnable bias to the convolution layers.
    nonlin : Type[nn.Module], optional, default=nn.ReLU
        Nonlinearity module class to use.
    norm_op : Type[nn.Module], optional, default=nn.BatchNorm2d
        Normalization module class to use.
    norm_op_kwargs : dict, optional
        Keyword arguments forwarded to "norm_op".
    dropout_op : Type[nn.Module], optional
        Dropout module class to use.
    dropout_op_kwargs : dict, optional
        Keyword arguments forwarded to "dropout_op".
    nonlin_kwargs : dict, optional
        Keyword arguments forwarded to "nonlin".
    nonlin_first : bool, default=False
        If "True", apply nonlinearity before normalization in conv blocks. If "False",
        apply normalization before nonlinearity.

    Notes
    -----
    - All intermediate feature maps keep the same spatial size as the input.
    - Encoder dilations follow "[1, 2, 4, ...]" and the decoder mirrors this schedule.
    - Dilation is capped so that "(k - 1) * dilation + 1 <= min(spatial_dim)" to prevent
      "kernel larger than input" errors.
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
        nonlin_first: bool = False,
    ):
        super().__init__()
        assert depth >= 2, "RSUdilatedBlock expects depth >= 2"
        self.depth = depth
        self.nonlin_first = nonlin_first

        # Defaults
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin else nn.Identity()
        self.dropout = dropout_op(**(dropout_op_kwargs or {})) if dropout_op else nn.Identity()

        # Normalize args
        ksize_list = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride_list = maybe_convert_scalar_to_list(conv_op, stride)

        # Padding for dilation=1 (will be recomputed per dilation at runtime)
        base_padding = [k // 2 for k in ksize_list]

        # Input conv can optionally downsample by provided stride (stage-level)
        self.conv_in = conv_op(in_ch, out_ch, ksize_list, stride_list, padding=base_padding, bias=bias)
        self.norm_in = norm_op(out_ch, **norm_op_kwargs) if norm_op else nn.Identity()

        # Encoder path (increasing dilations)
        self.encoders = nn.ModuleList()
        self.enc_norms = nn.ModuleList()
        for i in range(depth):
            cin = out_ch if i == 0 else (mid_ch if mid_ch is not None else out_ch // 2)
            cout = (mid_ch if mid_ch is not None else out_ch // 2)
            self.encoders.append(
                conv_op(cin, cout, ksize_list, 1, padding=base_padding, bias=bias)
            )
            self.enc_norms.append(norm_op(cout, **norm_op_kwargs) if norm_op else nn.Identity())

        # Bottleneck (largest dilation)
        self.bottom = conv_op((mid_ch if mid_ch is not None else out_ch // 2),
                              (mid_ch if mid_ch is not None else out_ch // 2),
                              ksize_list, 1, padding=base_padding, bias=bias)
        self.norm_bottom = norm_op((mid_ch if mid_ch is not None else out_ch // 2), **norm_op_kwargs) if norm_op else nn.Identity()

        # Decoder path (decreasing dilations)
        self.decoders = nn.ModuleList()
        self.dec_norms = nn.ModuleList()
        for i in range(depth):
            # Skip from encoder level depth-1-i
            skip_ch = out_ch if i == depth - 1 else (mid_ch if mid_ch is not None else out_ch // 2)
            in_dec_ch = (mid_ch if mid_ch is not None else out_ch // 2) + skip_ch
            out_dec_ch = (mid_ch if i < depth - 1 else out_ch)
            if mid_ch is None and i < depth - 1:
                out_dec_ch = out_ch // 2
            self.decoders.append(
                conv_op(in_dec_ch, out_dec_ch, ksize_list, 1, padding=base_padding, bias=bias)
            )
            self.dec_norms.append(norm_op(out_dec_ch, **norm_op_kwargs) if norm_op else nn.Identity())

        # Precompute base dilation schedule (will be capped at runtime)
        # Encoder: 1, 2, 4, 8, ... ; Bottom: last; Decoder: reverse without the last
        self.base_enc_dils = [1] + [2 ** i for i in range(1, depth)]
        self.base_dec_dils = list(reversed(self.base_enc_dils[:-1]))  # len = depth-1

        # Store kernel size for runtime padding/dilation adjustments
        self._ksize_tuple = tuple(ksize_list)

    @staticmethod
    def _set_conv_dilation(conv: nn.Module, ksize: Tuple[int, ...], dil: int):
        """
        Adjust dilation and padding on a convolution to keep spatial size constant.

        Parameters
        ----------
        conv : nn.Module
            Convolution layer whose dilation and padding will be adjusted.
        ksize : tuple of int
            Kernel size of the convolution per spatial dimension.
        dil : int
            Dilation factor to apply uniformly across spatial dimensions.
        """
        if hasattr(conv, 'dilation'):
            # Set dilation per spatial dim
            dim = len(ksize)
            conv.dilation = (dil,) * dim
            # Padding to keep "same" spatial size
            conv.padding = tuple((k // 2) * dil for k in ksize)

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
        The order of operations depends on "self.nonlin_first":
        - If True: conv -> nonlin -> norm
        - If False: conv -> norm -> nonlin

        For InstanceNorm with small spatial dimensions ("prod(spatial_dims) <= 1"),
        normalization is skipped to avoid numerical issues.
        """
        x = conv(x)
        # InstanceNorm safety on tiny maps
        is_instancenorm = isinstance(norm, (nn.InstanceNorm2d, nn.InstanceNorm3d))
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
            x = nonlin(x)
        return x

    def _max_safe_dilation(self, spatial: Tuple[int, ...]) -> int:
        """
        Compute maximum safe dilation so that the effective kernel fits the input.

        For a k-sized kernel and dilation "d", the effective size is
        "(k - 1) * d + 1" which must be "<= min(spatial)".

        Parameters
        ----------
        spatial : tuple of int
            Current spatial dimensions of the feature map.

        Returns
        -------
        int
            Maximum safe dilation value (at least 1).
        """
        min_dim = int(min(spatial)) if len(spatial) > 0 else 1
        # Use smallest kernel dim (they are usually equal)
        k_min = min(self._ksize_tuple) if len(self._ksize_tuple) > 0 else 3
        if k_min <= 1 or min_dim <= 1:
            return 1
        return max(1, (min_dim - 1) // (k_min - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RSU-4F dilated block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape "(batch_size, in_ch, *spatial_dims)".

        Returns
        -------
        torch.Tensor
            Output tensor of shape "(batch_size, out_ch, *spatial_dims)".
            The spatial dimensions remain the same as the input.

                Notes
                -----
                - Residual connection ensures the same spatial size as input.
                - Dilation is capped per-batch to ensure the effective kernel fits the current
                    feature map size.
        """
        # Entrance conv (may downsample per provided stride)
        x_in = self._apply_block(self.conv_in, self.norm_in, self.nonlin, x)
        x_in = self.dropout(x_in)

        # Determine safe dilations for current input size
        max_d = self._max_safe_dilation(tuple(x_in.shape[2:]))
        enc_dils = [min(d, max_d) for d in self.base_enc_dils]
        dec_dils = [min(d, max_d) for d in self.base_dec_dils]
        bottom_dil = enc_dils[-1]

        # Encoder with dilations, collect skips (all same spatial size)
        skips = [x_in]
        xi = x_in
        for i, (enc, norm) in enumerate(zip(self.encoders, self.enc_norms)):
            self._set_conv_dilation(enc, self._ksize_tuple, enc_dils[i])
            xi = self._apply_block(enc, norm, self.nonlin, xi)
            xi = self.dropout(xi)
            skips.append(xi)

        # Bottom with max dilation
        self._set_conv_dilation(self.bottom, self._ksize_tuple, bottom_dil)
        xb = self._apply_block(self.bottom, self.norm_bottom, self.nonlin, xi)
        xb = self.dropout(xb)

        # Decoder with decreasing dilations, concat with skips (no upsampling needed)
        xu = xb
        for i, (dec, norm) in enumerate(zip(self.decoders, self.dec_norms)):
            # Match skip from encoder: reverse order, skip does not include x_in at index 0 when i=0? we added x_in as first
            skip = skips[-(i + 2)]  # enc_feats[-(i+2)] style
            xu = torch.cat([xu, skip], dim=1)
            # Choose dilation for this decoder level
            dil = dec_dils[i] if i < len(dec_dils) else 1
            self._set_conv_dilation(dec, self._ksize_tuple, dil)
            xu = self._apply_block(dec, norm, self.nonlin, xu)
            xu = self.dropout(xu)

        # Residual connection
        # Ensure same spatial dims (should be by design); interpolate if minor mismatch
        if xu.shape[2:] != x_in.shape[2:]:
            mode = 'trilinear' if xu.dim() == 5 else 'bilinear'
            xu = F.interpolate(xu, size=x_in.shape[2:], mode=mode, align_corners=False)
        return xu + x_in

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

    Notes
    -----
    Since this block preserves spatial dimensions internally (no pooling), the
    feature map sizes are constant across layers and this proxy reflects that.
        """
        output = np.int64(0)
        # after conv_in
        output += np.prod([self.conv_in.out_channels, *input_size], dtype=np.int64)
        # encoders
        for enc in self.encoders:
            output += np.prod([enc.out_channels, *input_size], dtype=np.int64)
        # bottom
        output += np.prod([self.bottom.out_channels, *input_size], dtype=np.int64)
        # decoders (spatial size constant)
        for dec in self.decoders:
            output += np.prod([dec.out_channels, *input_size], dtype=np.int64)
        return output




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
    
    Notes
    -----
    - By default, early stages use pooling RSU blocks while the last stages can be
        configured to use dilated RSU blocks to preserve spatial resolution.
    - "strides" per stage control downsampling at the stage input.
    - When "return_skips=True", the forward method returns all stage outputs for
        use in the decoder.
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
        self.strides = strides
        self.return_skips = return_skips
        self.features_per_stage = features_per_stage
        self.depth_per_stage = depth_per_stage if depth_per_stage is not None else [4] * n_stages
        self.bias = conv_bias
        self.blocks_nonlin = blocks_nonlin if blocks_nonlin is not None else nonlin
        self.nonlin_first = nonlin_first
        self.stages = nn.ModuleList()
        prev_ch = input_channels
        for i in range(n_stages-2):
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
        # Last 2 stages are RSUdilatedBlock
        
        for i in range(n_stages-2, n_stages):
            depth = self.depth_per_stage[i]
            mid_ch = features_per_stage[i] // 2
            self.stages.append(
                RSUdilatedBlock(
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

        Notes
        -----
        - The list of feature maps is ordered from shallow (early stage) to deep (bottleneck).
        - Each stage may apply a stride at the entrance; additional downsampling may happen
            inside RSU blocks that use pooling.
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

    Notes
    -----
    This estimate aggregates the feature map sizes of all RSU stages and accounts for
    per-stage strides. It is used as a proxy for VRAM estimation.
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

        Notes
        -----
        - The first decoder stage can use a dilated RSU block to better preserve spatial
            detail at higher resolutions.
        - One segmentation head per decoder stage enables deep supervision.
    
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
        #first decoder stage is rsu dilated block
        self.stages.append(
            RSUdilatedBlock(
                in_ch=features_per_stage[-1] + features_per_stage[-2],
                out_ch=features_per_stage[-2],
                mid_ch=features_per_stage[-2] // 2,
                depth=encoder.depth_per_stage[-2],
                conv_op=encoder.conv_op,
                kernel_size=encoder.kernel_sizes[-2],
                stride=1,
                bias=encoder.bias,
                nonlin=self.blocks_nonlin,
                norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs,
                dropout_op=encoder.dropout_op,
                dropout_op_kwargs=encoder.dropout_op_kwargs,
                nonlin_kwargs=encoder.nonlin_kwargs,
                nonlin_first=nonlin_first
            )
        )
        for i in range(n_stages-2, 0, -1):
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

                Notes
                -----
                - Each decoder stage upsamples to match its corresponding encoder skip size,
                    concatenates, and applies an RSU block.
                - The outputs list is reversed so that index 0 corresponds to the highest
                    resolution output.
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

    Notes
    -----
    This aggregates decoder RSU block contributions, skip concatenations, and
    segmentation heads (for deep supervision or the final output).
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        assert len(skip_sizes) == len(self.stages)
        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            output += np.prod([self.encoder.features_per_stage[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output