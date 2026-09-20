import math
from typing import List, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_convtransp,
    get_matching_pool_op,
    maybe_convert_scalar_to_list,
)
from dynamic_network_architectures.building_blocks.regularization import DropPath


def _to_n_tuple(value: Union[int, List[int], Tuple[int, ...]], n: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return tuple([value] * n)
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Expected length {n}, got {len(value)}")
        return tuple(int(v) for v in value)
    raise TypeError(f"Unsupported parameter type: {type(value)}")


def _linear_macs(n_tokens: int, in_features: int, out_features: int) -> int:
    return int(n_tokens * in_features * out_features)


def _conv_nd_macs_from_module(conv_module: nn.Module, output_size: Union[List[int], Tuple[int, ...]]) -> int:
    kernel_size = conv_module.kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    groups = getattr(conv_module, "groups", 1)
    return int(
        np.prod(output_size, dtype=np.int64)
        * conv_module.out_channels
        * (conv_module.in_channels // groups)
        * np.prod(kernel_size, dtype=np.int64)
    )


def _compute_sequential_flops(module: nn.Module, input_size: Union[List[int], Tuple[int, ...]]) -> int:
    if module is None or isinstance(module, nn.Identity):
        return 0
    if isinstance(module, nn.Sequential):
        output = 0
        for submodule in module:
            if hasattr(submodule, "compute_approx_flops"):
                output += int(submodule.compute_approx_flops(input_size))
        return int(output)
    if hasattr(module, "compute_approx_flops"):
        return int(module.compute_approx_flops(input_size))
    return 0


def _compute_sequential_max_size(module: nn.Module, input_size: Union[List[int], Tuple[int, ...]]) -> int:
    if module is None or isinstance(module, nn.Identity):
        return 0
    if isinstance(module, nn.Sequential):
        output = 0
        for submodule in module:
            if hasattr(submodule, "compute_conv_max_size"):
                output = max(output, int(submodule.compute_conv_max_size(input_size)))
            elif hasattr(submodule, "compute_conv_feature_map_size"):
                output = max(output, int(submodule.compute_conv_feature_map_size(input_size)))
        return int(output)
    if hasattr(module, "compute_conv_max_size"):
        return int(module.compute_conv_max_size(input_size))
    if hasattr(module, "compute_conv_feature_map_size"):
        return int(module.compute_conv_feature_map_size(input_size))
    return 0


def estimate_cost_from_flops(flops_per_sample: int, batch_size: int = 1) -> dict:
    """
    Convert per-sample FLOPs into a small cost summary.

    Parameters
    ----------
    flops_per_sample : int
        Approximate FLOPs for one sample.
    batch_size : int, default=1
        Batch size used to scale per-sample estimates.

    Returns
    -------
    dict
        Dictionary with FLOPs and MACs per sample and per batch.
    """
    flops_per_sample = int(flops_per_sample)
    batch_size = int(batch_size)
    flops_per_batch = int(flops_per_sample * batch_size)

    return {
        "approx_flops_per_sample": flops_per_sample,
        "approx_macs_per_sample": flops_per_sample // 2,
        "approx_flops_per_batch": flops_per_batch,
        "approx_macs_per_batch": flops_per_batch // 2,
    }


class LayerNormNd(nn.Module):
    """
    Channel-wise LayerNorm for channels-first N-D tensors.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    eps : float, default=1e-6
        Numerical stability term.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)

        shape = [1, -1] + [1] * (x.ndim - 2)
        return self.weight.view(*shape) * x + self.bias.view(*shape)


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention for flattened local windows.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool, default=True
        If True, add bias to QKV projections.
    attn_drop : float, default=0.0
        Dropout applied to attention weights.
    proj_drop : float, default=0.0
        Dropout applied to the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply window attention to a batch of flattened windows.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (num_windows, window_volume, channels).

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as the input.
        """
        b_windows, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_windows, n_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]], window_size: Tuple[int, ...]) -> int:
        """
        Estimate the approximate FLOPs for attention over a spatial input.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.
        window_size : tuple
            Local attention window size.

        Returns
        -------
        int
            Approximate FLOPs for the attention module.
        """
        padded_shape = [int(math.ceil(i / w) * w) for i, w in zip(input_size, window_size)]
        windows_per_sample = int(np.prod([p // w for p, w in zip(padded_shape, window_size)], dtype=np.int64))
        window_volume = int(np.prod(window_size, dtype=np.int64))

        qkv_macs = windows_per_sample * _linear_macs(window_volume, self.dim, 3 * self.dim)
        attn_macs = windows_per_sample * self.num_heads * window_volume * window_volume * self.head_dim
        proj_macs = windows_per_sample * _linear_macs(window_volume, self.dim, self.dim)

        return int(2 * (qkv_macs + attn_macs + attn_macs + proj_macs))


def _window_partition_nd(
    x: torch.Tensor,
    window_size: Tuple[int, ...],
) -> Tuple[torch.Tensor, Tuple[int, ...], Tuple[int, ...]]:
    # x shape: [B, *spatial, C]
    spatial_shape = x.shape[1:-1]
    spatial_dims = len(spatial_shape)

    pad_needed = tuple((ws - (s % ws)) % ws for s, ws in zip(spatial_shape, window_size))
    if any(pad_needed):
        x_cf = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1)).contiguous()
        pad = []
        for p in reversed(pad_needed):
            pad.extend([0, p])
        x_cf = F.pad(x_cf, pad)
        x = x_cf.permute(0, *range(2, x_cf.ndim), 1).contiguous()

    padded_shape = tuple(x.shape[1:-1])
    grid_shape = [p // ws for p, ws in zip(padded_shape, window_size)]

    view_shape = [x.shape[0]]
    for g, ws in zip(grid_shape, window_size):
        view_shape.extend([g, ws])
    view_shape.append(x.shape[-1])

    x = x.view(*view_shape)
    grid_indices = [1 + 2 * i for i in range(spatial_dims)]
    window_indices = [2 + 2 * i for i in range(spatial_dims)]
    x = x.permute(0, *grid_indices, *window_indices, 1 + 2 * spatial_dims).contiguous()

    window_volume = math.prod(window_size)
    windows = x.view(-1, window_volume, x.shape[-1])
    return windows, padded_shape, pad_needed


def _window_reverse_nd(
    windows: torch.Tensor,
    window_size: Tuple[int, ...],
    padded_shape: Tuple[int, ...],
    original_shape: Tuple[int, ...],
    batch_size: int,
) -> torch.Tensor:
    spatial_dims = len(window_size)
    grid_shape = [p // ws for p, ws in zip(padded_shape, window_size)]

    x = windows.view(batch_size, *grid_shape, *window_size, windows.shape[-1])
    permute_order = [0]
    for i in range(spatial_dims):
        permute_order.extend([1 + i, 1 + spatial_dims + i])
    permute_order.append(1 + 2 * spatial_dims)

    x = x.permute(*permute_order).contiguous()
    x = x.view(batch_size, *padded_shape, windows.shape[-1])

    crop = [slice(None)]
    for s in original_shape:
        crop.append(slice(0, s))
    crop.append(slice(None))
    x = x[tuple(crop)].contiguous()
    return x


class SwinLiteBlock(nn.Module):
    """
    Swin transformer block with optional shifted windows and MLP.

    Parameters
    ----------
    dim : int
        Feature dimension.
    num_heads : int
        Number of attention heads.
    window_size : tuple of int
        Spatial window size used for attention.
    shift_size : tuple of int
        Cyclic shift applied before partitioning into windows.
    mlp_ratio : float, default=4.0
        Expansion ratio for the MLP hidden dimension.
    qkv_bias : bool, default=True
        If True, add bias to QKV projections.
    proj_drop : float, default=0.0
        Dropout applied to MLP and attention projections.
    attn_drop : float, default=0.0
        Dropout applied to attention weights.
    drop_path : float, default=0.0
        Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, ...],
        shift_size: Tuple[int, ...],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.spatial_dims = len(window_size)
        self.hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(self.hidden_dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Swin block to a channels-first tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, *spatial_dims).

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as the input.
        """
        # x shape: [B, C, *spatial]
        spatial_dims = x.ndim - 2
        if spatial_dims != self.spatial_dims:
            raise RuntimeError(f"Expected {self.spatial_dims}D input, got {spatial_dims}D")

        x = x.permute(0, *range(2, x.ndim), 1).contiguous()
        shortcut = x

        x = self.norm1(x)
        spatial_shape = tuple(x.shape[1:-1])
        if any(s > 0 for s in self.shift_size):
            dims = tuple(range(1, 1 + self.spatial_dims))
            x = torch.roll(x, shifts=tuple(-s for s in self.shift_size), dims=dims)

        windows, padded_shape, _ = _window_partition_nd(x, self.window_size)
        windows = self.attn(windows)
        x = _window_reverse_nd(windows, self.window_size, padded_shape, spatial_shape, shortcut.shape[0])

        if any(s > 0 for s in self.shift_size):
            dims = tuple(range(1, 1 + self.spatial_dims))
            x = torch.roll(x, shifts=self.shift_size, dims=dims)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1)).contiguous()
        return x

    def compute_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the feature map size produced by this block.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate feature map size.
        """
        return int(3 * np.prod([self.dim, *input_size], dtype=np.int64))

    def compute_conv_max_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the largest activation tensor size for this block.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate maximum activation size.
        """
        # Transformer block proxy: largest activation tensor has shape [C, *spatial].
        return int(np.prod([self.dim, *input_size], dtype=np.int64))

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the approximate FLOPs for one Swin block.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate FLOPs for the block.
        """
        attn_flops = self.attn.compute_approx_flops(input_size, self.window_size)
        n_tokens = int(np.prod(input_size, dtype=np.int64))
        mlp_macs = _linear_macs(n_tokens, self.dim, self.hidden_dim) + _linear_macs(
            n_tokens, self.hidden_dim, self.dim
        )
        return int(attn_flops + 2 * mlp_macs)


class SwinLiteStage(nn.Module):
    """
    A stack of SwinLiteBlock modules.

    Parameters
    ----------
    dim : int
        Feature dimension for all blocks in the stage.
    depth : int
        Number of Swin blocks.
    num_heads : int
        Number of attention heads per block.
    window_size : tuple of int
        Spatial window size used by the blocks.
    mlp_ratio : float, default=4.0
        Expansion ratio for the block MLPs.
    qkv_bias : bool, default=True
        If True, add bias to QKV projections.
    proj_drop : float, default=0.0
        Dropout applied to projections.
    attn_drop : float, default=0.0
        Dropout applied to attention weights.
    drop_path_rates : list of float, optional
        Per-block stochastic depth rates.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, ...],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rates: List[float] = None,
    ):
        super().__init__()
        if depth == 0:
            self.blocks = nn.ModuleList([])
            return

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        blocks = []
        for i in range(depth):
            shift_size = tuple([0] * len(window_size)) if (i % 2 == 0) else tuple([w // 2 for w in window_size])
            blocks.append(
                SwinLiteBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rates[i],
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all blocks in the stage sequentially.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, *spatial_dims).

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as the input.
        """
        for block in self.blocks:
            x = block(x)
        return x

    def compute_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the accumulated feature map size for the stage.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate feature map size.
        """
        output = 0
        for block in self.blocks:
            output += block.compute_feature_map_size(input_size)
        return int(output)

    def compute_conv_max_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the largest activation tensor size for the stage.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate maximum activation size.
        """
        output = 0
        for block in self.blocks:
            output = max(output, int(block.compute_conv_max_size(input_size)))
        return int(output)

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the approximate FLOPs for the full stage.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate FLOPs for the stage.
        """
        output = 0
        for block in self.blocks:
            output += block.compute_approx_flops(input_size)
        return int(output)


class SqueezeExcitationNd(nn.Module):
    """
    Squeeze-and-excitation block for N-D convolutional tensors.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    conv_op : Type[_ConvNd]
        Convolution operator used for the excitation layers.
    reduction : int, default=4
        Channel reduction factor.
    """

    def __init__(self, channels: int, conv_op: Type[_ConvNd], reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = get_matching_pool_op(conv_op=conv_op, adaptive=True, pool_type="avg")(1)
        self.fc1 = conv_op(channels, reduced, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = conv_op(reduced, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation gating.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reweighted tensor.
        """
        y = self.pool(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.gate(y)
        return x * y


class LiteModule(nn.Module):
    """
    Lightweight convolutional refinement module with expansion, depthwise mixing and SE.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    conv_op : Type[_ConvNd]
        Convolution operator used throughout the module.
    expansion_ratio : float, default=2.0
        Channel expansion ratio in the hidden branch.
    se_reduction : int, default=4
        Reduction factor for squeeze-excitation.
    kernel_size_large : int, default=7
        Kernel size of the larger depthwise branch.
    kernel_size_small : int, default=3
        Kernel size of the smaller depthwise branch.
    """

    def __init__(
        self,
        channels: int,
        conv_op: Type[_ConvNd],
        expansion_ratio: float = 2.0,
        se_reduction: int = 4,
        kernel_size_large: int = 7,
        kernel_size_small: int = 3,
    ):
        super().__init__()
        spatial_dims = convert_conv_op_to_dim(conv_op)
        hidden_channels = max(channels, int(round(channels * expansion_ratio)))

        k_large = _to_n_tuple(kernel_size_large, spatial_dims)
        k_small = _to_n_tuple(kernel_size_small, spatial_dims)
        p_large = tuple(k // 2 for k in k_large)
        p_small = tuple(k // 2 for k in k_small)

        self.hidden_channels = hidden_channels

        self.expand = conv_op(channels, hidden_channels, 1, bias=True)
        self.expand_norm = LayerNormNd(hidden_channels)
        self.expand_act = nn.GELU()

        self.dw_large = conv_op(
            hidden_channels,
            hidden_channels,
            kernel_size=k_large,
            stride=1,
            padding=p_large,
            groups=hidden_channels,
            bias=True,
        )
        self.dw_small = conv_op(
            hidden_channels,
            hidden_channels,
            kernel_size=k_small,
            stride=1,
            padding=p_small,
            groups=hidden_channels,
            bias=True,
        )

        self.dw_norm = LayerNormNd(hidden_channels)
        self.dw_act = nn.GELU()
        self.se = SqueezeExcitationNd(hidden_channels, conv_op=conv_op, reduction=se_reduction)

        self.project = conv_op(hidden_channels, channels, 1, bias=True)
        self.project_norm = LayerNormNd(channels)
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Lite module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with the same channel count as the input.
        """
        residual = x

        x = self.expand(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        x = self.dw_large(x) + self.dw_small(x)
        x = self.dw_norm(x)
        x = self.dw_act(x)
        x = self.se(x)

        x = self.project(x)
        x = self.project_norm(x)

        x = x + residual
        x = self.out_act(x)
        return x

    def compute_conv_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the feature map size generated by the module.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate feature map size.
        """
        hidden = np.prod([self.hidden_channels, *input_size], dtype=np.int64)
        output = np.prod([self.project.out_channels, *input_size], dtype=np.int64)
        # pointwise expand + two depthwise branches + pointwise project
        return int(3 * hidden + output)

    def compute_conv_max_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the largest activation tensor inside the module.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate maximum activation size.
        """
        hidden = np.prod([self.hidden_channels, *input_size], dtype=np.int64)
        output = np.prod([self.project.out_channels, *input_size], dtype=np.int64)
        return int(max(hidden, output))

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the approximate FLOPs for the module.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate FLOPs for the module.
        """
        volume = int(np.prod(input_size, dtype=np.int64))

        macs_expand = _conv_nd_macs_from_module(self.expand, input_size)
        macs_dw_large = volume * self.hidden_channels * int(np.prod(self.dw_large.kernel_size, dtype=np.int64))
        macs_dw_small = volume * self.hidden_channels * int(np.prod(self.dw_small.kernel_size, dtype=np.int64))
        macs_project = _conv_nd_macs_from_module(self.project, input_size)

        se_hidden = self.se.fc1.out_channels
        macs_se = self.hidden_channels * se_hidden + se_hidden * self.hidden_channels

        return int(2 * (macs_expand + macs_dw_large + macs_dw_small + macs_project + macs_se))


class LiteSwinEncoder(nn.Module):
    """
    Hierarchical encoder composed of strided convolutions, optional Lite modules and Swin stages.

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    n_stages : int
        Number of encoder stages.
    features_per_stage : int or list or tuple
        Number of channels per stage.
    conv_op : Type[_ConvNd]
        Convolution operator used throughout the encoder.
    kernel_sizes : int or list or tuple
        Kernel sizes for the encoder stages.
    strides : int or list or tuple
        Strides for the encoder stages.
    stage_depths : int or list or tuple, optional
        Number of Swin blocks per stage.
    num_heads : int or list or tuple, optional
        Attention heads per stage.
    window_size : int or list or tuple, default=7
        Attention window size.
    return_skips : bool, default=True
        If True, return all stage outputs for skip connections.
    mlp_ratio : float, default=4.0
        Expansion ratio used in transformer blocks.
    qkv_bias : bool, default=True
        If True, add bias to QKV projections.
    drop_path_rate : float, default=0.0
        Maximum stochastic depth rate.
    proj_drop_rate : float, default=0.0
        Dropout applied to attention and MLP projections.
    attn_drop_rate : float, default=0.0
        Dropout applied to attention weights.
    encoder_lite_modules_per_stage : int or list or tuple, optional
        Number of Lite modules per stage.
    lite_expansion_ratio : float, default=2.0
        Channel expansion ratio used by Lite modules.
    lite_se_reduction : int, default=4
        Reduction factor used by squeeze-excitation blocks.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        stage_depths: Union[int, List[int], Tuple[int, ...], None] = None,
        num_heads: Union[int, List[int], Tuple[int, ...], None] = None,
        window_size: Union[int, List[int], Tuple[int, ...]] = 7,
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        encoder_lite_modules_per_stage: Union[int, List[int], Tuple[int, ...], None] = None,
        lite_expansion_ratio: float = 2.0,
        lite_se_reduction: int = 4,
    ):
        super().__init__()
        self.n_stages = n_stages
        self.conv_op = conv_op
        self.return_skips = return_skips
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.spatial_dims = convert_conv_op_to_dim(conv_op)
        self.window_size = _to_n_tuple(window_size, self.spatial_dims)
        self.lite_expansion_ratio = lite_expansion_ratio

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if len(features_per_stage) != n_stages:
            raise ValueError(f"features_per_stage must have {n_stages} entries")
        self.output_channels = [int(i) for i in features_per_stage]

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if len(kernel_sizes) != n_stages:
            raise ValueError(f"kernel_sizes must have {n_stages} entries")
        self.kernel_sizes = [tuple(maybe_convert_scalar_to_list(conv_op, i)) for i in kernel_sizes]

        if isinstance(strides, int):
            strides = [strides] * n_stages
        if len(strides) != n_stages:
            raise ValueError(f"strides must have {n_stages} entries")
        self.strides = [tuple(maybe_convert_scalar_to_list(conv_op, i)) for i in strides]

        if stage_depths is None:
            stage_depths = [0] + [2] * (n_stages - 1)
        elif isinstance(stage_depths, int):
            stage_depths = [stage_depths] * n_stages
        if len(stage_depths) != n_stages:
            raise ValueError(f"stage_depths must have {n_stages} entries")
        self.stage_depths = [int(i) for i in stage_depths]
        self.stage_depths[0] = 0

        if num_heads is None:
            num_heads = [1] + [max(1, c // 12) for c in self.output_channels[1:]]
        elif isinstance(num_heads, int):
            num_heads = [num_heads] * n_stages
        if len(num_heads) != n_stages:
            raise ValueError(f"num_heads must have {n_stages} entries")
        self.num_heads = [int(i) for i in num_heads]

        if encoder_lite_modules_per_stage is None:
            encoder_lite_modules_per_stage = [1] + ([1] if n_stages > 1 else []) + ([3] * max(0, n_stages - 2))
        elif isinstance(encoder_lite_modules_per_stage, int):
            encoder_lite_modules_per_stage = [encoder_lite_modules_per_stage] * n_stages
        if len(encoder_lite_modules_per_stage) != n_stages:
            raise ValueError(f"encoder_lite_modules_per_stage must have {n_stages} entries")
        self.encoder_lite_modules_per_stage = [int(i) for i in encoder_lite_modules_per_stage]

        self.stem = conv_op(
            input_channels,
            self.output_channels[0],
            kernel_size=1,
            stride=self.strides[0],
            padding=0,
            bias=conv_bias,
        )

        if self.encoder_lite_modules_per_stage[0] > 0:
            self.stem_lite = nn.Sequential(
                *[
                    LiteModule(
                        self.output_channels[0],
                        conv_op=conv_op,
                        expansion_ratio=lite_expansion_ratio,
                        se_reduction=lite_se_reduction,
                    )
                    for _ in range(self.encoder_lite_modules_per_stage[0])
                ]
            )
        else:
            self.stem_lite = nn.Identity()

        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList([nn.Identity()])
        self.stage_lite_modules = nn.ModuleList([self.stem_lite])

        total_transformer_blocks = sum(self.stage_depths[1:])
        dpr = torch.linspace(0, drop_path_rate, total_transformer_blocks).tolist() if total_transformer_blocks > 0 else []
        dpr_pointer = 0

        for s in range(1, n_stages):
            self.downsamples.append(
                conv_op(
                    self.output_channels[s - 1],
                    self.output_channels[s],
                    kernel_size=self.strides[s],
                    stride=self.strides[s],
                    padding=0,
                    bias=conv_bias,
                )
            )

            depth = self.stage_depths[s]
            if depth > 0:
                heads = self.num_heads[s]
                if self.output_channels[s] % heads != 0:
                    heads = max(1, math.gcd(self.output_channels[s], heads))

                stage_dpr = dpr[dpr_pointer : dpr_pointer + depth]
                dpr_pointer += depth
                stage = SwinLiteStage(
                    dim=self.output_channels[s],
                    depth=depth,
                    num_heads=heads,
                    window_size=self.window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rates=stage_dpr,
                )
            else:
                stage = nn.Identity()
            self.stages.append(stage)

            if self.encoder_lite_modules_per_stage[s] > 0:
                stage_lite = nn.Sequential(
                    *[
                        LiteModule(
                            self.output_channels[s],
                            conv_op=conv_op,
                            expansion_ratio=lite_expansion_ratio,
                            se_reduction=lite_se_reduction,
                        )
                        for _ in range(self.encoder_lite_modules_per_stage[s])
                    ]
                )
            else:
                stage_lite = nn.Identity()
            self.stage_lite_modules.append(stage_lite)

    @staticmethod
    def _estimate_lite_module_size(
        channels: int,
        input_size: Union[List[int], Tuple[int, ...]],
        count: int,
        expansion_ratio: float,
    ) -> int:
        if count <= 0:
            return 0
        hidden = max(channels, int(round(channels * expansion_ratio)))
        hidden_size = np.prod([hidden, *input_size], dtype=np.int64)
        out_size = np.prod([channels, *input_size], dtype=np.int64)
        return int((3 * hidden_size + out_size) * count)

    def forward(self, x: torch.Tensor):
        """
        Run the encoder and collect skip features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, *spatial_dims).

        Returns
        -------
        list[torch.Tensor] or torch.Tensor
            Skip tensors if return_skips is True, otherwise the final stage output.
        """
        skips = []

        x = self.stem(x)
        x = self.stage_lite_modules[0](x)
        skips.append(x)

        for s in range(1, self.n_stages):
            x = self.downsamples[s - 1](x)
            x = self.stages[s](x)
            x = self.stage_lite_modules[s](x)
            skips.append(x)

        if self.return_skips:
            return skips
        return skips[-1]

    def compute_conv_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the total feature map size across the encoder.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate feature map size.
        """
        if len(input_size) != self.spatial_dims:
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )

        output = np.int64(0)
        current_size = list(input_size)

        current_size = [i // j for i, j in zip(current_size, self.strides[0])]
        output += np.prod([self.output_channels[0], *current_size], dtype=np.int64)
        output += self._estimate_lite_module_size(
            self.output_channels[0],
            current_size,
            self.encoder_lite_modules_per_stage[0],
            self.lite_expansion_ratio,
        )

        for s in range(1, self.n_stages):
            current_size = [i // j for i, j in zip(current_size, self.strides[s])]
            output += np.prod([self.output_channels[s], *current_size], dtype=np.int64)
            output += self.stages[s].compute_feature_map_size(current_size)
            output += self._estimate_lite_module_size(
                self.output_channels[s],
                current_size,
                self.encoder_lite_modules_per_stage[s],
                self.lite_expansion_ratio,
            )

        return int(output)

    def compute_conv_max_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the largest activation tensor across the encoder.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate maximum activation size.
        """
        if len(input_size) != self.spatial_dims:
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )

        max_size = 0
        current_size = list(input_size)

        current_size = [i // j for i, j in zip(current_size, self.strides[0])]
        max_size = max(max_size, int(np.prod([self.output_channels[0], *current_size], dtype=np.int64)))
        max_size = max(max_size, _compute_sequential_max_size(self.stage_lite_modules[0], current_size))

        for s in range(1, self.n_stages):
            current_size = [i // j for i, j in zip(current_size, self.strides[s])]
            max_size = max(max_size, int(np.prod([self.output_channels[s], *current_size], dtype=np.int64)))

            if hasattr(self.stages[s], "compute_conv_max_size"):
                max_size = max(max_size, int(self.stages[s].compute_conv_max_size(current_size)))
            max_size = max(max_size, _compute_sequential_max_size(self.stage_lite_modules[s], current_size))

        return int(max_size)

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the approximate FLOPs for the encoder.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate FLOPs for the encoder.
        """
        if len(input_size) != self.spatial_dims:
            raise AssertionError(
                "just give the image size without color/feature channels or batch channel. "
                "Give input_size=(x, y(, z))"
            )

        flops = 0
        current_size = list(input_size)

        current_size = [i // j for i, j in zip(current_size, self.strides[0])]
        flops += 2 * _conv_nd_macs_from_module(self.stem, current_size)
        flops += _compute_sequential_flops(self.stage_lite_modules[0], current_size)

        for s in range(1, self.n_stages):
            current_size = [i // j for i, j in zip(current_size, self.strides[s])]
            flops += 2 * _conv_nd_macs_from_module(self.downsamples[s - 1], current_size)

            if hasattr(self.stages[s], "compute_approx_flops"):
                flops += int(self.stages[s].compute_approx_flops(current_size))
            flops += _compute_sequential_flops(self.stage_lite_modules[s], current_size)

        return int(flops)


class LiteSwinUNETRDecoder(nn.Module):
    """
    Decoder that upsamples encoder features with transpose convolutions and Lite modules.

    Parameters
    ----------
    encoder : LiteSwinEncoder
        Encoder that provides the skip features.
    num_classes : int
        Number of output classes.
    deep_supervision : bool
        If True, produce segmentation outputs at all decoder stages.
    lite_modules_per_stage : int or list or tuple, default=1
        Number of Lite modules per decoder stage.
    lite_expansion_ratio : float, default=2.0
        Channel expansion ratio used by Lite modules.
    lite_se_reduction : int, default=4
        Reduction factor used by squeeze-excitation blocks.
    """

    def __init__(
        self,
        encoder: LiteSwinEncoder,
        num_classes: int,
        deep_supervision: bool,
        lite_modules_per_stage: Union[int, List[int], Tuple[int, ...]] = 1,
        lite_expansion_ratio: float = 2.0,
        lite_se_reduction: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.deep_supervision = deep_supervision
        self.lite_expansion_ratio = lite_expansion_ratio

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(lite_modules_per_stage, int):
            lite_modules_per_stage = [lite_modules_per_stage] * (n_stages_encoder - 1)
        if len(lite_modules_per_stage) != (n_stages_encoder - 1):
            raise ValueError(
                "lite_modules_per_stage must have n_stages_encoder - 1 entries "
                f"({n_stages_encoder - 1}), got {len(lite_modules_per_stage)}"
            )
        self.decoder_lite_modules_per_stage = [int(i) for i in lite_modules_per_stage]

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        stages = []
        transpconvs = []
        seg_layers = []
        fuse_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    kernel_size=stride_for_transpconv,
                    stride=stride_for_transpconv,
                    bias=True,
                )
            )

            fuse_layers.append(
                encoder.conv_op(
                    2 * input_features_skip,
                    input_features_skip,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )

            n_lite = self.decoder_lite_modules_per_stage[s - 1]
            if n_lite > 0:
                stages.append(
                    nn.Sequential(
                        *[
                            LiteModule(
                                input_features_skip,
                                conv_op=encoder.conv_op,
                                expansion_ratio=lite_expansion_ratio,
                                se_reduction=lite_se_reduction,
                            )
                            for _ in range(n_lite)
                        ]
                    )
                )
            else:
                stages.append(nn.Identity())

            seg_layers.append(encoder.conv_op(input_features_skip, self.num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    @staticmethod
    def _estimate_lite_module_size(
        channels: int,
        input_size: Union[List[int], Tuple[int, ...]],
        count: int,
        expansion_ratio: float,
    ) -> int:
        if count <= 0:
            return 0
        hidden = max(channels, int(round(channels * expansion_ratio)))
        hidden_size = np.prod([hidden, *input_size], dtype=np.int64)
        out_size = np.prod([channels, *input_size], dtype=np.int64)
        return int((3 * hidden_size + out_size) * count)

    def forward(self, skips: List[torch.Tensor]):
        """
        Decode a list of skip tensors into segmentation outputs.

        Parameters
        ----------
        skips : list[torch.Tensor]
            Skip tensors returned by the encoder, ordered from high to low resolution.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            Segmentation output, optionally with deep supervision outputs.
        """
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.fuse_layers[s](x)
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[s](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def compute_conv_feature_map_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the total feature map size across the decoder.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate feature map size.
        """
        current_size = list(input_size)
        stage_sizes = []
        for s in range(len(self.encoder.strides)):
            current_size = [i // j for i, j in zip(current_size, self.encoder.strides[s])]
            stage_sizes.append(list(current_size))

        output = np.int64(0)
        for s in range(len(self.stages)):
            target_size = stage_sizes[-(s + 2)]
            target_channels = self.encoder.output_channels[-(s + 2)]

            # transpose conv and 1x1 fusion
            output += np.prod([target_channels, *target_size], dtype=np.int64)
            output += np.prod([target_channels, *target_size], dtype=np.int64)

            # decoder lite modules
            output += self._estimate_lite_module_size(
                target_channels,
                target_size,
                self.decoder_lite_modules_per_stage[s],
                self.lite_expansion_ratio,
            )

            # segmentation heads
            if self.deep_supervision or (s == len(self.stages) - 1):
                output += np.prod([self.num_classes, *target_size], dtype=np.int64)

        return int(output)

    def compute_conv_max_size(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the largest activation tensor across the decoder.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate maximum activation size.
        """
        current_size = list(input_size)
        stage_sizes = []
        for s in range(len(self.encoder.strides)):
            current_size = [i // j for i, j in zip(current_size, self.encoder.strides[s])]
            stage_sizes.append(list(current_size))

        max_size = 0
        for s in range(len(self.stages)):
            target_size = stage_sizes[-(s + 2)]
            target_channels = self.encoder.output_channels[-(s + 2)]

            # transpose conv and 1x1 fusion outputs
            max_size = max(max_size, int(np.prod([target_channels, *target_size], dtype=np.int64)))
            max_size = max(max_size, int(np.prod([target_channels, *target_size], dtype=np.int64)))

            # decoder lite modules
            max_size = max(max_size, _compute_sequential_max_size(self.stages[s], target_size))

            # segmentation heads
            if self.deep_supervision or (s == len(self.stages) - 1):
                max_size = max(max_size, int(np.prod([self.num_classes, *target_size], dtype=np.int64)))

        return int(max_size)

    def compute_approx_flops(self, input_size: Union[List[int], Tuple[int, ...]]) -> int:
        """
        Estimate the approximate FLOPs for the decoder.

        Parameters
        ----------
        input_size : list or tuple
            Spatial input size without batch or channel dimensions.

        Returns
        -------
        int
            Approximate FLOPs for the decoder.
        """
        current_size = list(input_size)
        stage_sizes = []
        for s in range(len(self.encoder.strides)):
            current_size = [i // j for i, j in zip(current_size, self.encoder.strides[s])]
            stage_sizes.append(list(current_size))

        flops = 0
        for s in range(len(self.stages)):
            target_size = stage_sizes[-(s + 2)]

            flops += 2 * _conv_nd_macs_from_module(self.transpconvs[s], target_size)
            flops += 2 * _conv_nd_macs_from_module(self.fuse_layers[s], target_size)
            flops += _compute_sequential_flops(self.stages[s], target_size)

            if self.deep_supervision or (s == len(self.stages) - 1):
                flops += 2 * _conv_nd_macs_from_module(self.seg_layers[s], target_size)

        return int(flops)
