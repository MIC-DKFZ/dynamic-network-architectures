import torch
from torch import nn
import torch.nn.functional as F
#TODO: utilizzare le varie funzioni modulari di helper.py tipo _get_matching_pool_op, _from_conv_get_dim (o come si chiamano)
class RSUBlock(nn.Module):
    def __init__(self,
                in_ch,
                out_ch,
                mid_ch,
                depth = 4,
                conv_op = nn.Conv2d,
                kernel_size = 3,
                stride = 1,
                bias = True,
                nonlin = nn.ReLU,
                norm_op = nn.BatchNorm2d):
        
        super().__init__()
        # Parameters
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.conv_in = conv_op(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2, bias=bias)
        self.nonlin = nonlin()
        self.norm = norm_op(out_ch)

        # Encoder path 
        for i in range(depth):
            self.encoders.append(conv_op(out_ch if i == 0 else mid_ch, mid_ch, kernel_size, stride, padding=kernel_size//2, bias=bias))
            self.pools.append(nn.MaxPool2d(2, 2))

        # Bottom
        self.bottom = conv_op(mid_ch, mid_ch, kernel_size, stride, padding=kernel_size//2, bias=bias)

        # Decoder path  TODO: capire bene come si comporta, lo ha modificato copilot per risolvere errore di dimensione
        for i in range(depth):
            # L'ultimo skip è x_in (out_ch), gli altri sono mid_ch
            skip_ch = out_ch if i == depth - 1 else mid_ch
            in_ch_dec = mid_ch + skip_ch
            out_ch_dec = mid_ch if i < depth - 1 else out_ch
            self.decoders.append(
                conv_op(in_ch_dec, out_ch_dec, kernel_size, stride, padding=kernel_size//2, bias=bias)
            )

    def forward(self, x):
        x_in = self.nonlin(self.norm(self.conv_in(x))) #This is the original REBNCONV
        enc_feats = [x_in]
        xi = x_in
        # Encoder
        for enc, pool in zip(self.encoders, self.pools):
            xi = self.nonlin(enc(xi))
            enc_feats.append(xi)
            # ONLY IF DIMENSION > 1x1 
            ##### questo risolve degli errori di dimensione quando si usa il pooling
            if xi.shape[2] > 1 and xi.shape[3] > 1:
                xi = pool(xi)
        # Bottleneck
        xb = self.nonlin(self.bottom(xi)) #
        # Decoder
        xu = xb
        for i, dec in enumerate(self.decoders):
            skip = enc_feats[-(i+2)]
            xu = F.interpolate(xu, size=skip.shape[2:], mode='bilinear', align_corners=False)
            xu = torch.cat([xu, skip], dim=1)
            xu = self.nonlin(dec(xu))
        if xu.shape[2:] != x_in.shape[2:]:
            xu = F.interpolate(xu, size=x_in.shape[2:], mode='bilinear', align_corners=False)
        return xu + x_in


class RSUEncoder(nn.Module):
    def __init__(
        self,
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
        return_skips = True,
        nonlin_first = False,
        pool = "max",
        depth_per_stage = None
    ):
        super().__init__()
        self.return_skips = return_skips
        self.stages = nn.ModuleList()
        prev_ch = input_channels
        for i in range(n_stages):
            depth = depth_per_stage[i] if depth_per_stage is not None else 4
            mid_ch = features_per_stage[i] // 2
            self.stages.append(
                RSUBlock(
                    in_ch=prev_ch,
                    out_ch=features_per_stage[i],
                    mid_ch=mid_ch,
                    depth=depth,
                    conv_op=conv_op,
                    kernel_size=kernel_sizes[i][0] if isinstance(kernel_sizes[i], (list, tuple)) else kernel_sizes[i],
                    stride=strides[i][0] if isinstance(strides[i], (list, tuple)) else strides[i],
                    bias=conv_bias,
                    nonlin=nonlin,
                    norm_op=norm_op
                )
            )
            prev_ch = features_per_stage[i]

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        if self.return_skips:
            return skips
        return x


class RSUDecoder(nn.Module):
    def __init__(self, 
                features_per_stage, 
                depth_per_stage, 
                conv_op, 
                kernel_sizes, 
                strides, 
                bias, 
                nonlin, 
                norm_op):
        
        super().__init__()
        self.stages = nn.ModuleList()
        n_stages = len(features_per_stage)
        for i in range(n_stages-1, 0, -1):
            self.stages.append(
                RSUBlock(
                    in_ch = features_per_stage[i]+features_per_stage[i-1],  # skip + upsampled
                    out_ch = features_per_stage[i-1],
                    mid_ch = features_per_stage[i-1]//2,
                    depth = depth_per_stage[i-1],
                    conv_op = conv_op,
                    kernel_size = kernel_sizes[i-1][0] if isinstance(kernel_sizes[i-1], (list, tuple)) else kernel_sizes[i-1],
                    stride = 1,
                    bias = bias,
                    nonlin = nonlin,
                    norm_op = norm_op
                )
            )

    def forward(self, skips):
        x = skips[-1]
        outputs = []
        for i, stage in enumerate(self.stages):
            x = F.interpolate(x, size=skips[-(i+2)].shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skips[-(i+2)]], dim=1)
            x = stage(x)
            outputs.append(x)
        # Order the outputs from the last to the first 
        outputs = outputs[::-1]
        return outputs