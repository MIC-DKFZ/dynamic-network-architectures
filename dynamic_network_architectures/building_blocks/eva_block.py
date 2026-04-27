from typing import Callable, Optional

from timm.layers import LayerNorm 
import torch
from torch import nn
from timm.models.eva import EvaBlock as TimmEvaBlock


# Helper for handling tuples
def to_2tuple(x):
    return (x, x)


class GRN(nn.Module):
    """Global Response Normalization layer
    Based on ConvNeXt V2: http://arxiv.org/abs/2301.00808
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Channels)

        # 1. Compute the L2 norm of each channel across the spatial/sequence dimension.
        #    We assume dim 1 is the sequence length (N) or spatial volume.
        gx = torch.norm(x, p=2, dim=1, keepdim=True)

        # 2. Normalize the global norms relative to the channel average
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)

        # 3. Apply the calibration (scale + offset) + Residual connection
        return self.gamma * (x * nx) + self.beta + x


class SwiGLUwithGRN(nn.Module):
    """SwiGLU with Global Response Normalization (GRN)"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        # --- ADDED: GRN Layer ---
        self.grn = GRN(hidden_features)

        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

        # GRN weights are initialized to 0 by default in the class (gamma/beta),
        # which effectively makes it an identity mapping at init.
        # No extra init needed here.

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)

        # SwiGLU Activation
        x = self.act(x_gate) * x

        # GRN: Enhance channel contrast
        x = self.grn(x)

        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EvaBlockGRN(TimmEvaBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        attn_head_dim: Optional[int] = None,
        use_grn: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            mlp_ratio=mlp_ratio,
            swiglu_mlp=swiglu_mlp,
            scale_mlp=scale_mlp,
            scale_attn_inner=scale_attn_inner,
            num_prefix_tokens=num_prefix_tokens,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            init_values=init_values,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_head_dim=attn_head_dim,
        )
        # Replace the MLP if the parent built a SwiGLU
        if use_grn and swiglu_mlp and scale_mlp:
            hidden_features = int(dim * mlp_ratio)
            self.mlp = SwiGLUwithGRN(
                in_features=dim,
                hidden_features=hidden_features,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
