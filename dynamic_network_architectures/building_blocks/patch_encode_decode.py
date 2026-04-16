from typing import Literal, Tuple
import torch
from torch import nn
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op
import numpy as np

from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD, StackedResidualBlocks
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


block_type = Literal["basic", "bottleneck"]
block_style = Literal["residual", "conv"]


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        idx = (None, slice(None), *([None] * (x.ndim - 2)))
        x = self.weight[idx] * x + self.bias[idx]
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (16, 16, 16),
        input_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns shape (B, embed_dim, px, py, pz) where (px, py, pz) is patch_size.
        This output will need to be rearranged to whatever your transformer expects!
        """
        x = self.proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)

        # don't question me
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                    norm(channels[s + 1]),
                    activation(),
                )
            )
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)


class PatchEmbed_deeper(nn.Module):
    """ResNet-style patch embedding with progressive downsampling"""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        base_features: int = 32,
        depth_per_level: tuple[int, ...] = (1, 1, 1),
        embed_proj_3x3x3: bool = False,
        embed_block_type: block_type = "basic",
        embed_block_style: block_style = "residual",  # "basic" or "bottleneck" (if "residual" style)
    ) -> None:
        """
        Iterative, convolutional patch embedding layer.

        Args:
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            base_features (int): Number of base features for the first stage.
            depth_per_level (tuple[int, ...]): Number of blocks per stage/level.
            embed_proj_3x3x3 (bool): Whether to use a 3x3x3 convolution for the final projection to embed_dim.
            embed_block_type (residual_block_style): Type of residual block to use if embed_block_style is 'residual'. Either 'basic' or 'bottleneck'.
            embed_block_style (block_style): Style of blocks to use in the embedding stages. Either 'residual' or 'conv'.

        """

        super().__init__()

        norm_op = nn.InstanceNorm3d
        block = BottleneckD if embed_block_type == "bottleneck" else BasicBlockD
        nonlin = nn.LeakyReLU if embed_block_type == "bottleneck" else nn.ReLU
        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        nonlin_kwargs = {"inplace": True}

        # Stem convolution (initial feature extraction)
        if embed_block_type == "bottleneck":
            bottleneck_channels = base_features // 4
        else:
            bottleneck_channels = None

        if embed_block_style == "residual":
            self.stem = StackedResidualBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
                block=block,
            )
        elif embed_block_style == "conv":
            self.stem = StackedConvBlocks(
                1,
                nn.Conv3d,
                input_channels,
                base_features,
                [3, 3, 3],
                1,
                True,
                norm_op,
                norm_op_kwargs,
                None,
                None,
                nonlin,
                nonlin_kwargs,
            )
        else:
            raise ValueError(f"Unknown embed_block_style: {embed_block_style}. Must be 'residual' or 'conv'.")

        # Calculate total downsampling needed
        levels_needed = len(depth_per_level)

        # Build encoder stages
        self.stages = nn.ModuleList()
        input_channels = base_features

        for i in range(levels_needed):
            # First block in each stage handles downsampling and channel increase
            stride = 2
            output_channels = base_features * (2**i)
            if embed_block_style == "residual":
                if embed_block_type == "bottleneck":
                    bottleneck_channels = output_channels // 4
                else:
                    bottleneck_channels = None
                stage = StackedResidualBlocks(
                    n_blocks=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    block=block,
                    bottleneck_channels=bottleneck_channels,
                )
            elif embed_block_style == "conv":
                stage = StackedConvBlocks(
                    num_convs=depth_per_level[i],
                    conv_op=nn.Conv3d,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=3,
                    initial_stride=stride,
                    conv_bias=False,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=None,
                    dropout_op_kwargs=None,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            self.stages.append(stage)
            input_channels = output_channels

        # Global average pooling or final conv to get to embed_dim
        final_proj_kernel = [3, 3, 3] if embed_proj_3x3x3 else [1, 1, 1]
        final_pad = [1, 1, 1] if embed_proj_3x3x3 else [0, 0, 0]
        self.final_proj = nn.Conv3d(
            input_channels, embed_dim, kernel_size=final_proj_kernel, stride=[1, 1, 1], padding=final_pad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        # Progressive encoding through residual stages
        for stage in self.stages:
            x = stage(x)

        # Final projection to embedding tokens
        x = self.final_proj(x)

        return x


class PatchEmbedDeeperControlled(nn.Module):

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 864,
        depth_per_level: tuple[int, ...] = (1, 1, 1),  # number of residual blocks per level
        ch_per_level: tuple[int, ...] = (32, 64, 256, 1024),  # Channels of current v4
        add_skips: bool = True,
    ) -> None:
        """
        ResNet-style Patch Embedding with controllable channels for each depth level.

        :param input_channels: Amount of input channels
        :type input_channels: int
        :param embed_dim: Embedding dimension of the Patch Embedding
        :type embed_dim: int
        :param depth_per_level: Amount of Stacked Residual Blocks for each dowsampling level. Tuple length defines downsampling, currently 2**3 = 8 global stride.
        :type depth_per_level: tuple[int, ...]
        :param ch_per_level: Channel dimension projected to at each level. Length must be len(depth_per_level) + 1. Index 0 defines stem as well.
        :type ch_per_level: tuple[int, ...]
        :param add_skips: Flag, adding a Conv skip connection from the feature maps at each level to the final token grid. (default: True)
        :type add_skips: bool
        """
        super().__init__()
        self.add_skips = add_skips
        norm_op = nn.InstanceNorm3d
        nonlin = nn.ReLU
        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        nonlin_kwargs = {"inplace": True}

        # Stem convolution (initial feature extraction)
        base_features = ch_per_level[0]

        self.stem = StackedResidualBlocks(
            1,
            nn.Conv3d,
            input_channels,
            base_features,
            [3, 3, 3],
            1,
            True,
            norm_op,
            norm_op_kwargs,
            None,
            None,
            nonlin,
            nonlin_kwargs,
            block=BasicBlockD,
        )
        # Calculate total downsampling needed
        levels_needed = len(depth_per_level)

        # Build encoder stages
        self.stages = nn.ModuleList()
        input_channels = base_features
        for i in range(levels_needed):
            # First block in each stage handles downsampling and channel increase
            stride = 2
            output_channels = ch_per_level[i + 1]
            stage = StackedResidualBlocks(
                n_blocks=depth_per_level[i],
                conv_op=nn.Conv3d,
                input_channels=input_channels,
                output_channels=output_channels,
                kernel_size=3,
                initial_stride=stride,
                conv_bias=False,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=BasicBlockD,
                bottleneck_channels=None,
            )
            self.stages.append(stage)
            input_channels = output_channels

        self.final_proj = nn.Conv3d(input_channels, embed_dim, kernel_size=1, stride=1, padding=0)

        # ------------------------ start CONTRIBUTED BY IKIM STUDENT ----------------------- #
        # Luc.bouteille@uk-essen.de
        if self.add_skips:
            self.proj_to_tokens = nn.ModuleList()

            def _proj(in_ch: int, down_by: int) -> nn.Conv3d:
                return nn.Conv3d(
                    in_ch,
                    embed_dim,
                    kernel_size=(down_by, down_by, down_by),
                    stride=(down_by, down_by, down_by),
                    padding=0,
                    bias=True,
                )

            # stem is at full resolution -> down_by = 2**levels_needed
            self.proj_to_tokens.append(_proj(base_features, 2**levels_needed))
            # stages[0] output is /2, stages[1] output is /4, ... -> project all but last to /2**levels_needed
            for i in range(levels_needed - 1):
                in_ch = ch_per_level[i + 1]
                down_by = 2 ** (levels_needed - (i + 1))
                self.proj_to_tokens.append(_proj(in_ch, down_by))

            # Learnable scales for token-grid skip projections, initialized near zero.
            # Indexing matches proj_to_tokens: 0=stem, 1=stage0 (/2), 2=stage1 (/4), ...
            self.scale_proj_to_tokens = nn.ParameterList(
                [nn.Parameter(torch.tensor(float(1e-5))) for _ in range(len(self.proj_to_tokens))]
            )
        # ------------------------ END CONTRIBUTION BY STUDENT ----------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        # Add_skips idea of Luc.bouteille@uk-essen.de
        if self.add_skips:
            p = self.scale_proj_to_tokens[0] * self.proj_to_tokens[0](x)

        # Progressive encoding through residual stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # Do skip projection to token grid
            if self.add_skips and i < len(self.stages) - 1:
                j = i + 1
                p = p + self.scale_proj_to_tokens[j] * self.proj_to_tokens[j](x)

        # Final projection to embedding tokens
        x = self.final_proj(x)
        # Add skip projections
        if self.add_skips:
            x = x + p

        return x
