import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    MyBatchNorm1d,
    activation_stds,
    batchnorm_forward,
    deep_mlp,
    dropout,
    init_weights,
    label_smoothing_cross_entropy,
    layernorm,
    weight_decay_param_groups,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_init_weights():
    torch.manual_seed(0)
    model = deep_mlp(3, 512)
    assert init_weights(model, "kaiming") is model
    w = model[0].weight
    assert w.std().item() == pytest.approx((2 / 512) ** 0.5, rel=0.03)
    assert not model[2].bias.any()
    init_weights(model, "normal1")
    assert model[4].weight.std().item() == pytest.approx(1.0, rel=0.03)
    init_weights(model, "small")
    assert model[4].weight.std().item() == pytest.approx(0.01, rel=0.03)
    init_weights(model, "xavier")
    assert model[0].weight.std().item() == pytest.approx((2 / 1024) ** 0.5, rel=0.03)
    with pytest.raises(ValueError):
        init_weights(model, "magic")


def test_activation_stds_uses_hooks_and_cleans_up():
    torch.manual_seed(0)
    model = deep_mlp(4, 32, nn.Tanh)
    x = torch.randn(64, 32, generator=gen(1))
    stds = activation_stds(model, x)
    assert len(stds) == 4 and all(isinstance(s, float) for s in stds)
    with torch.no_grad():
        expected = model[:2](x).std().item()
    assert stds[0] == pytest.approx(expected)
    assert all(not m._forward_hooks for m in model.modules()), "remove your hooks"


@pytest.mark.parametrize("activation", [nn.ReLU, nn.Tanh])
def test_init_experiment(activation):
    x = torch.randn(512, 256, generator=gen(0))
    results = {}
    for scheme in ["normal1", "small", "xavier", "kaiming"]:
        torch.manual_seed(1)
        results[scheme] = activation_stds(init_weights(deep_mlp(20, 256, activation), scheme), x)
    assert len(results["kaiming"]) == 20
    assert 0.2 < results["kaiming"][-1] < 2
    assert results["small"][-1] < 1e-10
    if activation is nn.ReLU:
        assert results["normal1"][-1] > 1e10
        assert results["xavier"][-1] < 0.01
    else:
        assert results["normal1"][-1] > 0.9, "tanh saturates at +-1 instead of exploding"


def test_batchnorm_forward_train_matches_pytorch():
    x = torch.randn(32, 5, generator=gen(0)) * 3 + 2
    gamma, beta = torch.rand(5, generator=gen(1)) + 0.5, torch.randn(5, generator=gen(2))
    rm, rv = torch.zeros(5), torch.ones(5)
    out = batchnorm_forward(x, gamma, beta, rm, rv, training=True)
    ref = nn.BatchNorm1d(5)
    with torch.no_grad():
        ref.weight.copy_(gamma); ref.bias.copy_(beta)
    expected = ref(x)
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.allclose(rm, ref.running_mean, atol=1e-6), "update running_mean in place"
    assert torch.allclose(rv, ref.running_var, atol=1e-5), "running_var uses the unbiased variance"


def test_batchnorm_forward_eval():
    x = torch.randn(8, 3, generator=gen(0))
    rm, rv = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 1.0, 0.25])
    out = batchnorm_forward(x, torch.ones(3), torch.zeros(3), rm, rv, training=False)
    assert torch.allclose(out, (x - rm) / torch.sqrt(rv + 1e-5))
    assert torch.equal(rm, torch.tensor([1.0, 2.0, 3.0])) and torch.equal(rv, torch.tensor([4.0, 1.0, 0.25]))


def test_batchnorm_gradients():
    x = torch.randn(16, 4, generator=gen(0), dtype=torch.float64, requires_grad=True)
    gamma = torch.rand(4, generator=gen(1), dtype=torch.float64, requires_grad=True)
    beta = torch.randn(4, generator=gen(2), dtype=torch.float64, requires_grad=True)
    f = lambda a, g, b: batchnorm_forward(a, g, b, torch.zeros(4, dtype=torch.float64),
                                          torch.ones(4, dtype=torch.float64), training=True)
    assert torch.autograd.gradcheck(f, (x, gamma, beta))


def test_my_batchnorm_module():
    bn, ref = MyBatchNorm1d(6), nn.BatchNorm1d(6)
    assert {n for n, _ in bn.named_parameters()} == {"weight", "bias"}
    assert {n for n, _ in bn.named_buffers()} == {"running_mean", "running_var"}
    assert set(bn.state_dict()) >= {"weight", "bias", "running_mean", "running_var"}
    for seed in range(5):
        x = torch.randn(20, 6, generator=gen(seed)) * 2 + 1
        assert torch.allclose(bn(x), ref(x), atol=1e-5)
    bn.eval(); ref.eval()
    x = torch.randn(3, 6, generator=gen(9))
    assert torch.allclose(bn(x), ref(x), atol=1e-5)
    assert torch.allclose(bn.running_var, ref.running_var, atol=1e-5)


def test_layernorm():
    x = torch.randn(2, 3, 8, generator=gen(0)) * 4 + 1
    gamma, beta = torch.rand(8, generator=gen(1)), torch.randn(8, generator=gen(2))
    assert torch.allclose(layernorm(x, gamma, beta), F.layer_norm(x, (8,), gamma, beta), atol=1e-5)
    out = layernorm(x, torch.ones(8), torch.zeros(8))
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)


def test_dropout():
    x = torch.ones(1000, 100)
    out = dropout(x, 0.3, training=True, generator=gen(0))
    assert (out == 0).float().mean().item() == pytest.approx(0.3, abs=0.01)
    assert torch.allclose(out[out != 0], torch.tensor(1 / 0.7))
    assert out.mean().item() == pytest.approx(1.0, abs=0.01)
    mask = torch.rand(x.shape, generator=gen(0)) >= 0.3
    assert torch.equal(out != 0, mask)
    assert dropout(x, 0.3, training=False) is x
    assert torch.equal(dropout(x, 0.0, training=True), x)


def test_weight_decay_param_groups():
    model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    model[2].bias.requires_grad = False
    groups = weight_decay_param_groups(model, 0.1)
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
    assert [p.shape for p in groups[0]["params"]] == [(8, 4), (2, 8)]
    assert [id(p) for p in groups[1]["params"]] == [id(model[0].bias), id(model[1].weight), id(model[1].bias)]
    torch.optim.AdamW(groups, lr=1e-3)


def test_label_smoothing():
    logits = torch.randn(10, 5, generator=gen(0))
    y = torch.randint(0, 5, (10,), generator=gen(1))
    for eps in [0.0, 0.1, 0.3]:
        expected = F.cross_entropy(logits, y, label_smoothing=eps)
        assert label_smoothing_cross_entropy(logits, y, eps).item() == pytest.approx(expected.item(), rel=1e-5)
