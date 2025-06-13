import torch
import pytest
import sys

sys.path.append('/Users/stefanopetraccini/Desktop/dynamic-network-architectures')
from dynamic_network_architectures.architectures.u2net import U2NET_full, U2NET_lite

@pytest.mark.parametrize("model_fn, in_channels, img_size, expected_maps", [
    (lambda: U2NET_full(2, 1), 1, (1, 1, 256, 256), 7),
    (lambda: U2NET_lite(2, 1), 1, (1, 1, 256, 256), 7),
    (lambda: U2NET_full(3, 1), 1, (1, 1, 16, 256, 256), 7),
    (lambda: U2NET_lite(3, 1), 1, (1, 1, 16, 256, 256), 7),
])
def test_u2net_forward_shapes(model_fn, in_channels, img_size, expected_maps):
    model = model_fn()
    model.eval()
    x = torch.randn(*img_size)
    with torch.no_grad():
        outputs = model(x)
    assert isinstance(outputs, list)
    assert len(outputs) == expected_maps
    for out in outputs:
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == img_size[0]
        assert out.shape[2:] == img_size[2:]
        assert out.shape[1] == 1

def test_u2net_forward_batch_size():
    model = U2NET_lite(2, 1)
    model.eval()
    x = torch.randn(4, 1, 128, 128)
    with torch.no_grad():
        outputs = model(x)
    for out in outputs:
        assert out.shape[0] == 4
        assert out.shape[2:] == (128, 128)

def test_u2net_forward_requires_grad():
    model = U2NET_full(2, 1)
    x = torch.randn(2, 1, 64, 64, requires_grad=True)
    outputs = model(x)
    loss = sum([o.mean() for o in outputs])
    loss.backward()
    assert x.grad is not None

def test_u2net3d_forward_batch_size():
    model = U2NET_lite(3, 1)
    model.eval()
    x = torch.randn(2, 1, 8, 64, 64)
    with torch.no_grad():
        outputs = model(x)
    for out in outputs:
        assert out.shape[0] == 2
        assert out.shape[2:] == (8, 64, 64)

def test_u2net3d_forward_requires_grad():
    model = U2NET_full(3, 1)
    x = torch.randn(2, 1, 8, 32, 32, requires_grad=True)
    outputs = model(x)
    loss = sum([o.mean() for o in outputs])
    loss.backward()
    assert x.grad is not None