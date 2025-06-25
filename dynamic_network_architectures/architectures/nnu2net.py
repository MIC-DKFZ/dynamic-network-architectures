from typing import Union, Type, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_pool_op, get_matching_batchnorm, convert_conv_op_to_dim
from torch.nn.modules.conv import _ConvNd



class REBNCONV(nn.Module):
    def __init__(self, 
                 dim, 
                 activation = nn.ReLU, 
                 input_channels = 1, 
                 output_channels = 1, 
                 dirate = 1):
        super(REBNCONV, self).__init__()

        conv_op = convert_dim_to_conv_op(dim)
        bn = get_matching_batchnorm(dim)
        
        self.conv_s1 = conv_op(input_channels, output_channels, 3, padding = 1*dirate, dilation = 1*dirate)
        self.bn_s1 = bn(output_channels)
        self.act_s1 = activation(inplace = True) #previously self.relu_s1

    def forward(self, x):

        hx = x
        xout = self.act_s1(self.bn_s1(self.conv_s1(hx))) 

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(dim, src, tar):
   
    src = F.upsample(src, size = tar.shape[2:], mode = 'bilinear' if dim  ==  2 else 'trilinear')

    return src

### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, dim, activation_f, input_channels, mid_ch, output_channels):
        super(RSU7, self).__init__()

        self.activ = activation_f
        self.dim = dim
        pool_op = get_matching_pool_op(dimension = dim, pool_type = "max")
        
        self.rebnconvin = REBNCONV(self.dim, self.activ, input_channels, output_channels, dirate = 1)

        self.rebnconv1 = REBNCONV(self.dim, self.activ, output_channels, mid_ch, dirate = 1)
        self.pool1 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = REBNCONV(self.dim, self.activ, self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool2 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool3 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv4 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool4 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv5 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool5 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv6 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)

        self.rebnconv7 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 2)

        self.rebnconv6d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv5d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv4d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv3d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(self.dim, self.activ, mid_ch*2, output_channels, dirate = 1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(self.dim, hx6d, hx5, self.dim)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(self.dim, hx5d, hx4, self.dim)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(self.dim, hx4d, hx3, self.dim)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(self.dim, hx3d, hx2, self.dim)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(self.dim, hx2d, hx1, self.dim)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, dim, activation_f, input_channels, mid_ch, output_channels):
        super(RSU6, self).__init__()

        self.activ = activation_f
        self.dim = dim
        pool_op = get_matching_pool_op(dimension = dim, pool_type = "max")
        
        self.rebnconvin = REBNCONV(self.dim, self.activ, input_channels, output_channels, dirate = 1)

        self.rebnconv1 = REBNCONV(self.dim, self.activ, output_channels, mid_ch, dirate = 1)
        self.pool1 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool2 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool3 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv4 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool4 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv5 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)

        self.rebnconv6 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 2)

        self.rebnconv5d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv4d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv3d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(self.dim, self.activ, mid_ch*2, output_channels, dirate = 1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(self.dim, hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(self.dim, hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(self.dim, hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(self.dim, hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, dim, activation_f, input_channels, mid_ch, output_channels):
        super(RSU5, self).__init__()

        self.activ = activation_f
        self.dim = dim
        pool_op = get_matching_pool_op(dimension = dim, pool_type = "max")
        
        self.rebnconvin = REBNCONV(self.dim, self.activ, input_channels, output_channels, dirate = 1)

        self.rebnconv1 = REBNCONV(self.dim, self.activ, output_channels, mid_ch, dirate = 1)
        self.pool1 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool2 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool3 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv4 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)

        self.rebnconv5 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 2)

        self.rebnconv4d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv3d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(self.dim, self.activ, mid_ch*2, output_channels, dirate = 1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(self.dim, hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(self.dim, hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(self.dim, hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, dim, activation_f, input_channels, mid_ch, output_channels):
        super(RSU4, self).__init__()

        self.activ = activation_f
        self.dim = dim
        pool_op = get_matching_pool_op(dimension = dim, pool_type = "max")
        
        self.rebnconvin = REBNCONV(self.dim, self.activ, input_channels, output_channels, dirate = 1)

        self.rebnconv1 = REBNCONV(self.dim, self.activ, output_channels, mid_ch, dirate = 1)
        self.pool1 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv2 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)
        self.pool2 = pool_op(2, stride = 2, ceil_mode = True)

        self.rebnconv3 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 1)

        self.rebnconv4 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 2)

        self.rebnconv3d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv2d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 1)
        self.rebnconv1d = REBNCONV(self.dim, self.activ, mid_ch*2, output_channels, dirate = 1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(self.dim, hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(self.dim, hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, dim, activation_f, input_channels, mid_ch, output_channels):
        super(RSU4F, self).__init__()
        
        self.activ = activation_f
        self.dim = dim
        
        self.rebnconvin = REBNCONV(self.dim, self.activ, input_channels, output_channels, dirate = 1)

        self.rebnconv1 = REBNCONV(self.dim, self.activ, output_channels, mid_ch, dirate = 1)
        self.rebnconv2 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 2)
        self.rebnconv3 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 4)

        self.rebnconv4 = REBNCONV(self.dim, self.activ, mid_ch, mid_ch, dirate = 8)

        self.rebnconv3d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 4)
        self.rebnconv2d = REBNCONV(self.dim, self.activ, mid_ch*2, mid_ch, dirate = 2)
        self.rebnconv1d = REBNCONV(self.dim, self.activ, mid_ch*2, output_channels, dirate = 1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, 
                 dim, 
                 activation_f, 
                 input_channels, 
                 deep_supervision = True, 
                 output_channels = 1
                 ):
        super(U2NET, self).__init__()

        self.dim = dim
        self.activ = activation_f
        self.deep_supervision = deep_supervision
        
        pool_op = get_matching_pool_op(dimension = self.dim, pool_type = "max")
        conv_op = convert_dim_to_conv_op(self.dim)
        
        self.stage1 = RSU7(self.dim, self.activ, input_channels, 32, 64)
        self.pool12 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage2 = RSU6(self.dim, self.activ, 64, 32, 128)
        self.pool23 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage3 = RSU5(self.dim, self.activ, 128, 64, 256)
        self.pool34 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage4 = RSU4(self.dim, self.activ, 256, 128, 512)
        self.pool45 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage5 = RSU4F(self.dim, self.activ, 512, 256, 512)
        self.pool56 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage6 = RSU4F(self.dim, self.activ, 512, 256, 512)

        # decoder
        self.stage5d = RSU4F(self.dim, self.activ, 1024, 256, 512)
        self.stage4d = RSU4(self.dim, self.activ, 1024, 128, 256)
        self.stage3d = RSU5(self.dim, self.activ, 512, 64, 128)
        self.stage2d = RSU6(self.dim, self.activ, 256, 32, 64)
        self.stage1d = RSU7(self.dim, self.activ, 128, 16, 64)

        self.side1 = conv_op(64, output_channels, 3, padding = 1)
        self.side2 = conv_op(64, output_channels, 3, padding = 1)
        self.side3 = conv_op(128, output_channels, 3, padding = 1)
        self.side4 = conv_op(256, output_channels, 3, padding = 1)
        self.side5 = conv_op(512, output_channels, 3, padding = 1)
        self.side6 = conv_op(512, output_channels, 3, padding = 1)

        self.outconv = conv_op(6*output_channels, output_channels, 1)

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(self.dim, hx6, hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(self.dim, hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(self.dim, hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(self.dim, hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(self.dim, hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(self.dim, d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(self.dim, d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(self.dim, d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(self.dim, d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(self.dim, d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        if self.deep_supervision:
            return (
                F.sigmoid(d0), 
                F.sigmoid(d1), 
                F.sigmoid(d2), 
                F.sigmoid(d3), 
                F.sigmoid(d4), 
                F.sigmoid(d5), 
                F.sigmoid(d6), 
            )
        else:
            return F.sigmoid(d0)


### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self, 
                conv_op: Type[_ConvNd], 
                input_channels: int = 1, 
                deep_supervision: bool = True, 
                nonlin: Union[None, Type[torch.nn.Module]] = None, 
                nonlin_kwargs: dict = None, 
                output_channels: int = 1, 
                activation_function = nn.ReLU, ##RSU blocks nonlinearity. Default is ReLU. 
                nonlin_first: bool = False #not sure if I care doing that also for RSU. I guess not NGL. We'll see.
                 ):
        super(U2NETP, self).__init__()

        self.dim = convert_conv_op_to_dim(conv_op)
        self.nonlin = nonlin #storing this even if not used
        self.activ_kwargs = nonlin_kwargs
        self.activ = activation_function
        self.deep_supervision = deep_supervision
        
        
        pool_op = get_matching_pool_op(dimension = self.dim, pool_type = "max")
        
        self.stage1 = RSU7(self.dim, self.activ, input_channels, 16, 64)
        self.pool12 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage2 = RSU6(self.dim, self.activ, 64, 16, 64)
        self.pool23 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage3 = RSU5(self.dim, self.activ, 64, 16, 64)
        self.pool34 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage4 = RSU4(self.dim, self.activ, 64, 16, 64)
        self.pool45 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage5 = RSU4F(self.dim, self.activ, 64, 16, 64)
        self.pool56 = pool_op(2, stride = 2, ceil_mode = True)

        self.stage6 = RSU4F(self.dim, self.activ, 64, 16, 64)

        # decoder
        self.stage5d = RSU4F(self.dim, self.activ, 128, 16, 64)
        self.stage4d = RSU4(self.dim, self.activ, 128, 16, 64)
        self.stage3d = RSU5(self.dim, self.activ, 128, 16, 64)
        self.stage2d = RSU6(self.dim, self.activ, 128, 16, 64)
        self.stage1d = RSU7(self.dim, self.activ, 128, 16, 64)

        self.side1 = conv_op(64, output_channels, 3, padding = 1)
        self.side2 = conv_op(64, output_channels, 3, padding = 1)
        self.side3 = conv_op(64, output_channels, 3, padding = 1)
        self.side4 = conv_op(64, output_channels, 3, padding = 1)
        self.side5 = conv_op(64, output_channels, 3, padding = 1)
        self.side6 = conv_op(64, output_channels, 3, padding = 1)

        self.outconv = conv_op(6*output_channels, output_channels, 1)

    def forward(self, x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(self.dim, hx6, hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(self.dim, hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(self.dim, hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(self.dim, hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(self.dim, hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(self.dim, d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(self.dim, d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(self.dim, d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(self.dim, d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(self.dim, d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        if self.deep_supervision:
            return (d0, d1, d2, d3, d4, d5, d6)
        else:
            return d0