import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from code_checks import LOOP_NODES, contains
from exercises import (
    MLP,
    ClippedReLU,
    MyLinear,
    RoundSTE,
    attention_scores,
    count_parameters,
    fit_linear_autograd,
    freeze_except,
    get_device,
    grad_of,
    pairwise_sq_distances,
    to_tensors,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_get_device():
    assert get_device() in {"cuda", "mps", "cpu"}
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        assert get_device() == "cpu"
    torch.zeros(1).to(get_device())


def test_to_tensors():
    X_np, y_np = np.arange(6, dtype=np.float64).reshape(3, 2), np.array([0, 1, 2])
    X, y = to_tensors(X_np, y_np)
    assert X.dtype == torch.float32 and y.dtype == torch.int64
    assert torch.equal(X, torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    X_np[0, 0] = 99.0
    y_np[0] = 99
    assert X[0, 0] == 0.0 and y[0] == 0, "the tensors must not share memory with NumPy"


def test_pairwise_sq_distances():
    A, B = torch.randn(5, 3, generator=gen(0)), torch.randn(4, 3, generator=gen(1))
    D = pairwise_sq_distances(A, B)
    assert D.shape == (5, 4)
    assert torch.allclose(D, torch.cdist(A, B) ** 2, atol=1e-5)
    assert not contains(pairwise_sq_distances, LOOP_NODES)


def test_attention_scores():
    Q, K = torch.randn(2, 3, 8, generator=gen(0)), torch.randn(2, 5, 8, generator=gen(1))
    S = attention_scores(Q, K)
    assert S.shape == (2, 3, 5)
    assert torch.allclose(S[1, 2, 4], Q[1, 2] @ K[1, 4] / math.sqrt(8))
    assert not contains(attention_scores, LOOP_NODES)


def test_grad_of():
    x = torch.tensor([1.0, -2.0, 3.0])
    g = grad_of(lambda t: (t ** 3).sum(), x)
    assert torch.allclose(g, 3 * x ** 2)
    assert not x.requires_grad and x.grad is None
    w = torch.tensor([0.5, 0.5], requires_grad=True)
    g = grad_of(lambda t: torch.sigmoid(t).prod(), w)
    s = torch.sigmoid(torch.tensor(0.5))
    assert torch.allclose(g, torch.full((2,), s * (1 - s) * s))
    assert w.grad is None


def test_fit_linear_autograd():
    X = torch.randn(200, 3, generator=gen(0))
    y = X @ torch.tensor([2.0, -1.0, 0.5]) + 3.0
    w, b = fit_linear_autograd(X, y, steps=300, lr=0.1)
    assert not w.requires_grad and not b.requires_grad
    assert torch.allclose(w, torch.tensor([2.0, -1.0, 0.5]), atol=1e-3)
    assert float(b) == pytest.approx(3.0, abs=1e-3)
    w1, _ = fit_linear_autograd(X, y, steps=1, lr=0.1)
    expected = 0.1 * 2 * (X * y[:, None]).mean(dim=0)       # one step from zero: -lr * grad
    assert torch.allclose(w1, expected, atol=1e-5), "zero the gradients and use the mean loss"


def test_my_linear():
    torch.manual_seed(0)
    layer = MyLinear(16, 4)
    names = dict(layer.named_parameters())
    assert set(names) == {"weight", "bias"}
    assert names["weight"].shape == (4, 16) and names["bias"].shape == (4,)
    assert all(isinstance(p, nn.Parameter) for p in names.values())
    assert layer.weight.abs().max() <= 0.25 and layer.weight.std() > 0.05
    x = torch.randn(3, 16, generator=gen(1))
    assert torch.allclose(layer(x), x @ layer.weight.T + layer.bias)
    layer(x).sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.shape == (4, 16)


def test_mlp():
    torch.manual_seed(0)
    model = MLP([4, 8, 8, 3])
    assert isinstance(model.layers, nn.ModuleList)
    assert [type(l) for l in model.layers] == [nn.Linear] * 3
    assert len(list(model.parameters())) == 6
    x = torch.randn(5, 4, generator=gen(1))
    l1, l2, l3 = model.layers
    assert torch.allclose(model(x), l3(torch.relu(l2(torch.relu(l1(x))))))


def test_count_and_freeze():
    model = MLP([4, 8, 3])
    assert count_parameters(model) == 4 * 8 + 8 + 8 * 3 + 3
    assert freeze_except(model, ["layers.1"]) is model
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["layers.1.weight", "layers.1.bias"]
    assert count_parameters(model) == 8 * 3 + 3
    freeze_except(model, ["layers.0.bias", "layers.1.bias"])
    assert count_parameters(model) == 8 + 3


def test_round_ste():
    x = torch.tensor([0.2, 1.7, -1.4], requires_grad=True)
    y = RoundSTE.apply(x)
    assert torch.equal(y, torch.tensor([0.0, 2.0, -1.0]))
    (y * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    assert torch.equal(x.grad, torch.tensor([1.0, 2.0, 3.0]))


def test_clipped_relu():
    x = torch.tensor([-1.0, 0.5, 2.0, 7.0], requires_grad=True)
    y = ClippedReLU.apply(x, 6.0)
    assert torch.equal(y, torch.tensor([0.0, 0.5, 2.0, 6.0]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    x64 = torch.randn(10, dtype=torch.float64, generator=gen(0)) * 3
    x64 = x64[(x64.abs() > 0.1) & ((x64 - 2).abs() > 0.1)].requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: ClippedReLU.apply(t, 2.0), (x64,))
