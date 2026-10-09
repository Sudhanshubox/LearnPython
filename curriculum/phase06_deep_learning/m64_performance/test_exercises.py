import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    CheckpointedMLP,
    DynamicLossScaler,
    accumulated_step,
    adam_training_bytes,
    autocast_train_step,
    benchmark,
    mlp_forward_flops,
    mlp_training_flops,
    parameter_bytes,
    training_time_seconds,
    underflow_fraction,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def test_flops():
    assert mlp_forward_flops([784, 256, 10], 64) == 2 * 64 * (784 * 256 + 256 * 10)
    assert mlp_training_flops([784, 256, 10], 64) == 3 * mlp_forward_flops([784, 256, 10], 64)
    assert mlp_forward_flops([5], 10) == 0


def test_training_time():
    seconds = training_time_seconds(175e9, 300e9, peak_flops=312e12, utilization=0.5)
    assert seconds == pytest.approx(6 * 175e9 * 300e9 / 156e12)
    assert seconds / (86400 * 365) == pytest.approx(64.1, abs=0.1)     # GPT-3 on ONE A100: ~64 years


def test_memory():
    assert adam_training_bytes(7e9) / 1e9 == pytest.approx(112)
    model = mlp()
    n = sum(p.numel() for p in model.parameters())
    assert parameter_bytes(model) == 4 * n
    assert parameter_bytes(model.to(torch.bfloat16)) == 2 * n


def test_float_formats_underflow():
    x = 10 ** (torch.rand(100_000, generator=gen(0)) * -6 - 4)       # magnitudes 1e-10 .. 1e-4
    x[:10] = 0.0
    fp16 = underflow_fraction(x, torch.float16)
    assert isinstance(fp16, float)
    assert 0.3 < fp16 < 0.6, "float16 can't represent gradients below ~3e-8"
    assert underflow_fraction(x, torch.bfloat16) == 0.0, "bfloat16 has float32's range"
    assert underflow_fraction(x * 2 ** 16, torch.float16) == 0.0, "that's why we scale the loss"


def test_loss_scaler_update():
    s = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    assert s.scale == 1024.0 and s.good_steps == 0
    s.update(True); s.update(True)
    assert s.scale == 1024.0 and s.good_steps == 2
    s.update(True)
    assert s.scale == 2048.0 and s.good_steps == 0
    s.update(True); s.update(False)
    assert s.scale == 1024.0 and s.good_steps == 0


def test_loss_scaler_step():
    w = nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = torch.optim.SGD([w], lr=0.1)
    s = DynamicLossScaler(init_scale=8.0, growth_interval=100)
    (w.sum() * s.scale).backward()
    assert s.step(opt, [w]) is True
    assert torch.allclose(w.grad, torch.ones(2)), "unscale the gradients"
    assert torch.allclose(w.detach(), torch.tensor([0.9, 1.9]))
    w.grad = torch.tensor([float("inf"), 1.0])
    assert s.step(opt, [w]) is False
    assert torch.allclose(w.detach(), torch.tensor([0.9, 1.9])), "skip the step on overflow"
    assert s.scale == 4.0


def test_autocast_train_step():
    model = mlp()
    dtypes = []
    model[0].register_forward_hook(lambda m, i, o: dtypes.append(o.dtype) and None)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[0].weight.detach().clone()
    xb, yb = torch.randn(8, 16, generator=gen(1)), torch.randint(0, 4, (8,), generator=gen(2))
    loss = autocast_train_step(model, opt, xb, yb)
    assert isinstance(loss, float)
    assert dtypes == [torch.bfloat16], "the forward pass runs in bfloat16"
    assert model[0].weight.dtype == torch.float32, "parameters stay float32"
    assert not torch.equal(before, model[0].weight)
    with torch.no_grad():
        ref = F.cross_entropy(mlp()(xb), yb).item()
    assert loss == pytest.approx(ref, abs=0.05)


def test_accumulation_equals_big_batch():
    X, y = torch.randn(32, 16, generator=gen(3)), torch.randint(0, 4, (32,), generator=gen(4))
    big, small = mlp(0), mlp(0)
    opt_big = torch.optim.SGD(big.parameters(), lr=0.5)
    opt_big.zero_grad()
    big_loss = F.cross_entropy(big(X), y)
    big_loss.backward()
    opt_big.step()
    opt_small = torch.optim.SGD(small.parameters(), lr=0.5)
    for p in small.parameters():
        p.grad = torch.ones_like(p)                   # stale gradients must be cleared
    micro = [(X[i:i + 8], y[i:i + 8]) for i in range(0, 32, 8)]
    loss = accumulated_step(small, opt_small, micro)
    assert loss == pytest.approx(big_loss.item(), rel=1e-5)
    for a, b in zip(big.parameters(), small.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def saved_bytes(model, x):
    total = [0]
    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(x)
    return out, total[0]


def test_checkpointed_mlp():
    torch.manual_seed(0)
    plain = CheckpointedMLP(64, 8, use_checkpoint=False)
    torch.manual_seed(0)
    ckpt = CheckpointedMLP(64, 8, use_checkpoint=True)
    assert isinstance(ckpt.blocks, nn.ModuleList) and len(ckpt.blocks) == 8
    x = torch.randn(256, 64, generator=gen(5))
    out_p, bytes_p = saved_bytes(plain, x)
    out_c, bytes_c = saved_bytes(ckpt, x)
    assert torch.allclose(out_p, out_c)
    assert bytes_c < 0.5 * bytes_p, "checkpointing stores far fewer activations"
    out_p.sum().backward()
    out_c.sum().backward()
    for a, b in zip(plain.parameters(), ckpt.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)
    ckpt.eval()
    _, bytes_eval = saved_bytes(ckpt, x)
    assert bytes_eval == bytes_p, "no checkpointing in eval mode"


def test_benchmark():
    calls = []
    def fn():
        calls.append(1)
        time.sleep(0.002 if len(calls) <= 3 else 0.001)
    t = benchmark(fn, warmup=3, repeats=7)
    assert len(calls) == 10
    assert 0.0009 < t < 0.01
