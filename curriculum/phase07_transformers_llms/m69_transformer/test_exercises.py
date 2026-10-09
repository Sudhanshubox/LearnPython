import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    GPT,
    MLP,
    Block,
    CausalSelfAttention,
    GPTConfig,
    RMSNorm,
    count_parameters,
    estimate_parameters,
    rope,
    sinusoidal_positions,
)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def rand(*shape, seed=0):
    return torch.randn(*shape, generator=gen(seed))


TINY = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, d_model=32)


def test_sinusoidal_positions():
    pe = sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8)
    assert torch.allclose(pe[0], torch.tensor([0.0, 1.0] * 4))
    angle = 3 / 10000 ** (4 / 8)
    assert pe[3, 4].item() == pytest.approx(math.sin(angle), abs=1e-6)
    assert pe[3, 5].item() == pytest.approx(math.cos(angle), abs=1e-6)
    assert pe.abs().max() <= 1.0


def test_rmsnorm():
    norm = RMSNorm(6)
    assert isinstance(norm.weight, nn.Parameter) and torch.equal(norm.weight, torch.ones(6))
    x = rand(2, 3, 6) * 5
    expected = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(norm(x), expected, atol=1e-5)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    assert torch.allclose(norm(x), 2 * expected, atol=1e-5)


def test_rope_properties():
    x = rand(2, 7, 8)
    r = rope(x)
    assert r.shape == x.shape
    assert torch.allclose(r[:, 0], x[:, 0]), "position 0 is not rotated"
    assert torch.allclose(r.norm(dim=-1), x.norm(dim=-1), atol=1e-5), "rotations preserve length"
    half = 4
    angle = 3 * 10000.0 ** (-1 / half)
    x1, x2 = x[0, 3, 1], x[0, 3, 1 + half]
    assert r[0, 3, 1].item() == pytest.approx((x1 * math.cos(angle) - x2 * math.sin(angle)).item(), abs=1e-5)


def test_rope_relative_positions():
    q, k = rand(16, seed=1), rand(16, seed=2)
    def score(m, n):
        T = max(m, n) + 1
        qs = torch.zeros(T, 16); ks = torch.zeros(T, 16)
        qs[m], ks[n] = q, k
        return (rope(qs)[m] @ rope(ks)[n]).item()
    assert score(5, 3) == pytest.approx(score(9, 7), abs=1e-4), "only m - n matters"
    assert score(2, 2) == pytest.approx((q @ k).item(), abs=1e-4)
    assert score(5, 3) != pytest.approx(score(5, 1), abs=1e-3)


def test_mlp():
    torch.manual_seed(0)
    mlp = MLP(8)
    assert mlp.fc.weight.shape == (32, 8) and mlp.proj.weight.shape == (8, 32)
    x = rand(2, 3, 8)
    assert torch.allclose(mlp(x), mlp.proj(F.gelu(mlp.fc(x))), atol=1e-6)


def test_block_is_pre_norm_residual():
    torch.manual_seed(0)
    block = Block(16, 2)
    assert isinstance(block.attn, CausalSelfAttention) and isinstance(block.mlp, MLP)
    assert isinstance(block.ln1, nn.LayerNorm) and isinstance(block.ln2, nn.LayerNorm)
    x = rand(2, 5, 16)
    h = x + block.attn(block.ln1(x))
    assert torch.allclose(block(x), h + block.mlp(block.ln2(h)), atol=1e-5)


def test_gpt_structure_and_tying():
    torch.manual_seed(0)
    model = GPT(TINY)
    assert model.config is TINY
    assert model.lm_head.weight is model.tok_emb.weight, "tie the weights (same Parameter)"
    assert model.lm_head.bias is None
    assert isinstance(model.blocks, nn.ModuleList) and len(model.blocks) == 2
    assert model.pos_emb.weight.shape == (16, 32) and isinstance(model.ln_f, nn.LayerNorm)


def test_gpt_init():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=500, block_size=64, n_layer=8, n_head=4, d_model=128))
    assert model.tok_emb.weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert model.blocks[0].attn.qkv.weight.std().item() == pytest.approx(0.02, rel=0.05)
    assert model.blocks[0].mlp.proj.weight.std().item() == pytest.approx(0.02 / math.sqrt(16), rel=0.05)
    assert model.blocks[3].attn.proj.weight.std().item() == pytest.approx(0.02 / math.sqrt(16), rel=0.05)
    assert not model.blocks[0].attn.qkv.bias.any()


def test_gpt_forward():
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, 50, (3, 10), generator=gen(1))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, 50) and loss is None
    targets = torch.randint(0, 50, (3, 10), generator=gen(2))
    _, loss = model(idx, targets)
    assert loss.item() == pytest.approx(math.log(50), abs=0.1), "the initial loss should be about ln(V)"
    assert torch.allclose(loss, F.cross_entropy(logits.reshape(-1, 50), targets.reshape(-1)))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_gpt_is_causal():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    idx = torch.randint(0, 50, (1, 12), generator=gen(1))
    other = idx.clone()
    other[0, 8:] = (other[0, 8:] + 1) % 50
    assert torch.allclose(model(idx)[0][0, :8], model(other)[0][0, :8], atol=1e-5)


@pytest.mark.timeout(60)
def test_gpt_overfits_one_batch():
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, 50, (4, 16), generator=gen(3))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_parameter_counts():
    torch.manual_seed(0)
    model = GPT(TINY)
    assert count_parameters(model) == sum(p.numel() for p in set(model.parameters()))
    assert estimate_parameters(TINY) == count_parameters(model)
    gpt2 = GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, d_model=768)
    with torch.device("meta"):
        big = GPT(gpt2)
    assert count_parameters(big) == 124_439_808
    assert estimate_parameters(gpt2) == 124_439_808
