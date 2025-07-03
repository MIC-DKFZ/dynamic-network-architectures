import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, convert_dim_to_conv_op, get_matching_convtransp
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
                nonlin = nn.ReLU, #TODO: voglio forse poter distinguere tra la nonlnearity generale e quella usata nei blocchi RSU come fa l'architettura originale
                norm_op = nn.BatchNorm2d
                ): #TODO: implementare un nonlin_first che permette di mettere la non linearità prima della normalizzazione
        
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
        self.stages = nn.ModuleList()
        self.depth_per_stage = depth_per_stage if depth_per_stage is not None else [4] * n_stages
        self.bias = conv_bias
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
    
    def compute_conv_feature_map_size(self, input_size):
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
    def __init__(self, 
                encoder,
                num_classes,
                deep_supervision = False):
        
        super().__init__()
        self.stages = nn.ModuleList()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.encoder = encoder  # Store the encoder for compute_conv_feature_map_size
        
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
                    kernel_size = encoder.kernel_sizes[i-1][0] if isinstance(encoder.kernel_sizes[i-1], (list, tuple)) else encoder.kernel_sizes[i-1],
                    stride = 1,
                    bias = encoder.bias,
                    nonlin = encoder.nonlin,
                    norm_op = encoder.norm_op
                )
            )
        # Segmentation heads: one per output for deep supervision, or just one if not
        self.seg_layers = nn.ModuleList([
            encoder.conv_op(features_per_stage[i-1], num_classes, 1, 1, 0, bias=True)
            for i in range(n_stages-1, 0, -1)
        ])

    def forward(self, skips):
        x = skips[-1]
        outputs = []
        for i, stage in enumerate(self.stages):
            x = F.interpolate(x, size=skips[-(i+2)].shape[2:], mode='bilinear', align_corners=False)
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
    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output