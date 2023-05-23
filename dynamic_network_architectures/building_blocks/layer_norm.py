import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    r"""
    Lightly adapted from https://github.com/facebookresearch/ConvNeXt! Not mine!
    (adaptation is only in the last part in order to ensure 3D compatibility)

    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, (depth,) channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width (, depth)).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.shape) == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise ValueError('Only 2d and 3d inputs (NCHW/NHWC or NCHWD/NHWDC) are supported')
            return x


if __name__ == '__main__':
    input_tensor_2d = torch.rand((32, 3, 12, 15))
    input_tensor_3d = torch.rand((32, 3, 12, 15, 18))

    ln = LayerNorm(3, data_format="channels_first")
    res_2d = ln(input_tensor_2d)
    res_3d = ln(input_tensor_3d)

    input_tensor_2d_cl = torch.rand((32, 12, 15, 3))
    input_tensor_3d_cl = torch.rand((32, 12, 15, 18, 3))

    ln = LayerNorm(3, data_format="channels_last")
    res_2d_cl = ln(input_tensor_2d_cl)
    res_3d_cl = ln(input_tensor_3d_cl)
