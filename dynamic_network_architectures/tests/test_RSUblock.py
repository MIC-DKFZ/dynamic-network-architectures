import torch
import pytest
from dynamic_network_architectures.building_blocks.RSUblock import REBNCONV, RSU

@pytest.mark.parametrize("dim, height, in_ch, mid_ch, out_ch, shape", [
    (2, 4, 1, 2, 1, (1, 1, 64, 64)),
    (3, 4, 1, 2, 1, (1, 1, 32, 32, 32)),
])
def test_rsu_forward(dim, height, in_ch, mid_ch, out_ch, shape):
    model = RSU("RSU_test", dim, height, in_ch, mid_ch, out_ch)
    x = torch.randn(shape)
    y = model(x)
    assert y.shape == x.shape, f"Output shape {y.shape} != Input shape {x.shape}"
