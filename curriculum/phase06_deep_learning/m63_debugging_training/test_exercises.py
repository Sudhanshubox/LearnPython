import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    add_nan_checks,
    dead_relu_fractions,
    expected_initial_loss,
    find_nonfinite,
    grad_norms_by_layer,
    overfit_single_batch,
    train_classifier,
    update_to_data_ratios,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def mlp(seed=0, d=20, h=32, k=5, dropout=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, k))


def batch(n=32, d=20, k=5, seed=0):
    return torch.randn(n, d, generator=gen(seed)), torch.randint(0, k, (n,), generator=gen(seed + 100))


def test_expected_initial_loss():
    assert expected_initial_loss(10) == pytest.approx(2.302585, rel=1e-6)
    assert expected_initial_loss(50257) == pytest.approx(10.825, abs=1e-3)
    xb, yb = batch(512)
    model = mlp()
    with torch.no_grad():
        model[3].weight.mul_(0.01)
        loss = F.cross_entropy(model(xb), yb).item()
    assert loss == pytest.approx(expected_initial_loss(5), abs=0.05)


def test_overfit_single_batch():
    xb, yb = batch()
    model = mlp()
    model.eval()
    losses = overfit_single_batch(model, xb, yb, steps=200)
    assert len(losses) == 200 and all(isinstance(v, float) for v in losses)
    assert model.training
    assert losses[-1] < 0.01
    assert (model(xb).argmax(1) == yb).all()
    tiny = mlp(h=2)
    for p in tiny[0].parameters():
        p.requires_grad = False
    assert overfit_single_batch(tiny, xb, yb, steps=200)[-1] > 0.3, "a crippled model can't overfit: that's the signal"


def test_grad_norms_by_layer():
    xb, yb = batch()
    model = mlp()
    model[3].weight.grad = torch.full_like(model[3].weight, 100.0)    # stale gradient
    model[0].bias.requires_grad = False
    norms = grad_norms_by_layer(model, xb, yb)
    assert list(norms) == ["0.weight", "3.weight", "3.bias"]
    ref = mlp()
    F.cross_entropy(ref(xb), yb).backward()
    assert norms["0.weight"] == pytest.approx(ref[0].weight.grad.norm().item(), rel=1e-5)
    assert norms["3.weight"] == pytest.approx(ref[3].weight.grad.norm().item(), rel=1e-5), "zero old grads first"

    class Detached(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(20, 8), nn.Linear(8, 5)
        def forward(self, x):
            return self.b(self.a(x).detach())
    norms = grad_norms_by_layer(Detached(), xb, yb)
    assert norms["a.weight"] is None and norms["b.weight"] > 0


def test_update_to_data_ratios():
    xb, yb = batch()
    model = mlp()
    F.cross_entropy(model(xb), yb).backward()
    model[0].bias.grad = None
    ratios = update_to_data_ratios(model, 1e-3)
    assert set(ratios) == {"0.weight", "3.weight", "3.bias"}
    w = model[3].weight
    assert ratios["3.weight"] == pytest.approx((1e-3 * w.grad.norm() / w.norm()).item(), rel=1e-5)


def test_dead_relu_fractions():
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3), nn.ReLU())
    with torch.no_grad():
        model[0].weight.zero_(); model[0].bias.copy_(torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0]))
        model[2].weight.fill_(0.1); model[2].bias.copy_(torch.tensor([-100.0, 0.0, 0.0]))
    fractions = dead_relu_fractions(model, torch.randn(10, 4, generator=gen(0)))
    assert fractions == pytest.approx([2 / 6, 1 / 3])
    assert all(not m._forward_hooks for m in model.modules())
    conv = nn.Sequential(nn.Conv2d(1, 2, 1), nn.ReLU())
    with torch.no_grad():
        conv[0].weight.fill_(1.0); conv[0].bias.zero_()
    x = -torch.ones(3, 1, 2, 2)
    x[0, 0, 0, 0] = 1.0
    assert dead_relu_fractions(conv, x) == pytest.approx([6 / 8])


def test_find_nonfinite():
    model = mlp()
    assert find_nonfinite(model) == []
    with torch.no_grad():
        model[3].bias[0] = float("nan")
    model[0].weight.grad = torch.zeros_like(model[0].weight)
    model[0].weight.grad[1, 2] = float("inf")
    assert find_nonfinite(model) == ["0.weight", "3.bias"]


def test_add_nan_checks():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 3)
            self.act = nn.ReLU()
            self.out = nn.Linear(3, 1)
        def forward(self, x):
            return self.out(torch.log(self.act(self.fc(x)) - 1.0))   # log of a negative -> nan
    net = Net()
    handles = add_nan_checks(net)
    assert len(handles) == 3
    with pytest.raises(RuntimeError, match="non-finite output in out"):
        net(torch.zeros(2, 3))
    with torch.no_grad():
        net.fc.weight.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="non-finite output in fc"):
        net(torch.ones(2, 3))
    for h in handles:
        h.remove()
    net(torch.zeros(2, 3))


class SpySGD(torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self.seen = []

    def step(self, closure=None):
        self.seen.append([p.grad.clone() for p in self.param_groups[0]["params"]])
        return super().step(closure)


def test_bug_hunt_gradients_are_right():
    X, y = batch(n=100, seed=1)
    model = mlp(dropout=0.0)
    modes, inputs = [], []

    def record(module, inp, out):
        modes.append(module.training)
        inputs.append(inp[0].clone())
    model.register_forward_hook(record)
    opt = SpySGD(model.parameters(), lr=0.0)
    history = train_classifier(model, opt, X, y, epochs=2, batch_size=32)
    assert all(modes), "train in train mode"
    assert len(opt.seen) == 8
    label_of = {tuple(row.tolist()): int(t) for row, t in zip(X, y)}
    for xb, grads in zip(inputs, opt.seen):
        yb = torch.tensor([label_of[tuple(r.tolist())] for r in xb])
        model.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        for p, g in zip(model.parameters(), grads):
            assert torch.allclose(g, p.grad, atol=1e-6), \
                "gradients differ: check zero_grad, the loss on raw logits, and that labels stay aligned"
    assert len(history) == 2 and all(type(v) is float for v in history), "store floats, not tensors"


@pytest.mark.timeout(60)
def test_bug_hunt_trains():
    X = torch.randn(600, 20, generator=gen(2))
    y = (X[:, :5].argmax(dim=1))
    model = mlp(dropout=0.1)
    history = train_classifier(model, torch.optim.Adam(model.parameters(), lr=3e-3), X, y, epochs=15)
    assert history[-1] < 0.3 * history[0]
    model.eval()
    with torch.no_grad():
        assert (model(X).argmax(1) == y).float().mean() > 0.95
