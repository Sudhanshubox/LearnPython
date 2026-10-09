import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from code_checks import LOOP_NODES, contains, names_used
from exercises import (
    SmallCNN,
    conv2d_im2col,
    conv2d_naive,
    conv_output_size,
    maxpool2d,
    random_shift,
    receptive_field,
    train_cnn,
)

FORBIDDEN = {"conv2d", "Conv2d", "max_pool2d", "MaxPool2d"}


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_conv_output_size():
    assert conv_output_size(8, 3, padding=1) == 8
    assert conv_output_size(32, 3, stride=2, padding=1) == 16
    assert conv_output_size(224, 7, stride=2, padding=3) == 112
    assert conv_output_size(5, 3) == 3
    assert conv_output_size(10, 3, dilation=2) == 6
    assert conv_output_size(7, 2, stride=2) == 3


@pytest.mark.parametrize("stride, padding", [(1, 0), (1, 1), (2, 1), (2, 0)])
def test_conv2d_naive(stride, padding):
    x = torch.randn(2, 3, 7, 6, generator=gen(0))
    w, b = torch.randn(4, 3, 3, 3, generator=gen(1)), torch.randn(4, generator=gen(2))
    out = conv2d_naive(x, w, b, stride, padding)
    assert torch.allclose(out, F.conv2d(x, w, b, stride=stride, padding=padding), atol=1e-5)
    assert not FORBIDDEN & names_used(conv2d_naive)


@pytest.mark.parametrize("stride, padding, k", [(1, 0, 3), (1, 1, 3), (2, 1, 3), (1, 2, 5), (2, 0, 2)])
def test_conv2d_im2col(stride, padding, k):
    x = torch.randn(3, 2, 9, 8, generator=gen(0))
    w, b = torch.randn(5, 2, k, k, generator=gen(1)), torch.randn(5, generator=gen(2))
    out = conv2d_im2col(x, w, b, stride, padding)
    assert torch.allclose(out, F.conv2d(x, w, b, stride=stride, padding=padding), atol=1e-5)
    assert not FORBIDDEN & names_used(conv2d_im2col)
    assert not contains(conv2d_im2col, LOOP_NODES)


def test_im2col_gradients():
    x = torch.randn(1, 2, 5, 5, generator=gen(0), dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, generator=gen(1), dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, generator=gen(2), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda *a: conv2d_im2col(*a, stride=1, padding=1), (x, w, b))


def test_maxpool2d():
    x = torch.randn(2, 3, 8, 6, generator=gen(0))
    assert torch.equal(maxpool2d(x, 2), F.max_pool2d(x, 2))
    assert torch.equal(maxpool2d(x[:, :, :6], 3), F.max_pool2d(x[:, :, :6], 3))
    assert not FORBIDDEN & names_used(maxpool2d)
    assert not contains(maxpool2d, LOOP_NODES)


def test_receptive_field():
    assert receptive_field([]) == 1
    assert receptive_field([(3, 1)]) == 3
    assert receptive_field([(3, 1), (3, 1)]) == 5
    assert receptive_field([(3, 1)] * 3) == 7
    assert receptive_field([(3, 1), (2, 2), (3, 1), (2, 2)]) == 10
    assert receptive_field([(7, 2), (3, 2), (3, 1), (3, 1)]) == 27


def test_small_cnn():
    model = SmallCNN()
    assert isinstance(model.features, nn.Sequential)
    assert [type(m) for m in model.features] == [nn.Conv2d, nn.ReLU, nn.MaxPool2d] * 2
    assert isinstance(model.classifier, nn.Linear) and model.classifier.in_features == 128
    out = model(torch.randn(5, 1, 8, 8))
    assert out.shape == (5, 10)
    assert sum(p.numel() for p in model.parameters()) == 160 + 4640 + 1290


def test_random_shift():
    x = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).view(2, 1, 4, 4)
    assert torch.equal(random_shift(x, 0, gen(0)), x)
    out = random_shift(x, 1, gen(3))
    shifts = torch.randint(-1, 2, (2, 2), generator=gen(3))
    for n in range(2):
        dy, dx = shifts[n].tolist()
        for i in range(4):
            for j in range(4):
                si, sj = i - dy, j - dx
                expected = x[n, 0, si, sj] if 0 <= si < 4 and 0 <= sj < 4 else 0.0
                assert out[n, 0, i, j] == expected
    assert out.shape == x.shape


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    a, b, c, d = train_test_split(X / 16.0, y, test_size=0.25, random_state=0, stratify=y)
    t = lambda z: torch.tensor(z, dtype=torch.float32).view(-1, 1, 8, 8)
    return t(a), torch.tensor(c), t(b), torch.tensor(d)


@pytest.mark.timeout(120)
def test_train_cnn(digits):
    model, acc = train_cnn(*digits, epochs=15)
    assert isinstance(model, SmallCNN) and isinstance(acc, float)
    assert acc > 0.965
    _, acc2 = train_cnn(*digits, epochs=15)
    assert acc2 == acc, "seed everything"
    _, acc_aug = train_cnn(*digits, epochs=15, augment=True)
    assert acc_aug > 0.95
