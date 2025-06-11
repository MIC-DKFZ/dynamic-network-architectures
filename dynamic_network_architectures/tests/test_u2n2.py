import torch
import pytest
import sys

sys.path.append('/Users/stefanopetraccini/Desktop/dynamic-network-architectures')
from dynamic_network_architectures.architectures.u2net import U2NET_full, U2NET_lite

@pytest.mark.parametrize("model_fn, in_channels, img_size, expected_maps", [
    (lambda: U2NET_full(2), 3, (1, 3, 256, 256), 7),
    (lambda: U2NET_lite(2), 3, (1, 3, 256, 256), 7),
])

def test_u2net_forward_shapes(model_fn, in_channels, img_size, expected_maps):
    model = model_fn()
    model.eval()
    x = torch.randn(*img_size)
    with torch.no_grad():
        outputs = model(x)
    assert isinstance(outputs, list)
    assert len(outputs) == expected_maps
    # All outputs should be torch tensors and have the same spatial size as input
    for out in outputs:
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == img_size[0]
        assert out.shape[2:] == img_size[2:]
        # Output channels should be 1 (out_ch)
        assert out.shape[1] == 1

def test_u2net_forward_batch_size():
    model = U2NET_lite(2)
    model.eval()
    x = torch.randn(4, 3, 128, 128)
    with torch.no_grad():
        outputs = model(x)
    for out in outputs:
        assert out.shape[0] == 4
        assert out.shape[2:] == (128, 128)

def test_u2net_forward_requires_grad():
    model = U2NET_full(2)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    outputs = model(x)
    loss = sum([o.mean() for o in outputs])
    loss.backward()
    assert x.grad is not None