from torch import nn
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.RSU2 import RSUEncoder, RSUDecoder

class U2Net(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: list,
        conv_op,
        kernel_sizes: list,
        strides: list,
        n_conv_per_stage: int,
        conv_bias: bool,
        norm_op,
        norm_op_kwargs: dict,
        dropout_op,
        dropout_op_kwargs: dict,
        nonlin, ########## TODO: ideally we want to be able to define different nonlinearities for both RSUBlock and "general" nonlinearity (the one used by nnUNet)
        nonlin_kwargs: dict,
        deep_supervision: bool = True,
        return_skips: bool = True,
        nonlin_first: bool = False,
        pool: str = "max",
        depth_per_stage: list = None,
        **kwargs
    ):
        self.deep_supervision = deep_supervision
        super().__init__()
        self.encoder = RSUEncoder(
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
            return_skips,
            nonlin_first,
            pool,
            depth_per_stage = depth_per_stage
        )
        self.decoder = RSUDecoder(
            features_per_stage,
            depth_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            conv_bias,
            nonlin,
            norm_op
        )

    def forward(self, x):
        skips = self.encoder(x)
        outputs = self.decoder(skips)
        if self.training and self.deep_supervision: ###### validation wants only the last output so can't use only "if self.deep_supervision"
            return outputs  # tensor list
        else:
            return outputs[0]  # return only the last output (both in case of no deep supervision and in case of deep supervision during validation)