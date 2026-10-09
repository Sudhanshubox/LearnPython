import copy
import math

import pytest
import torch
import torch.nn as nn

from code_checks import LOOP_NODES, contains
from exercises import (
    QuantizedLinear,
    dequantize,
    dequantize_int4_groups,
    forward_with_cache,
    generate_cached,
    kv_cache_bytes,
    quantize_int4_groups,
    quantize_int8,
    quantize_model,
    sample_next,
    top_k_filter,
    top_p_filter,
    weight_bytes,
)
from minigpt import GPT, GPTConfig

CONFIG = GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, d_model=32)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = GPT(CONFIG)
    with torch.no_grad():                       # larger weights so outputs are far from uniform
        for p in m.parameters():
            p.mul_(5)
    return m.eval()


def test_top_k_filter():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.5], [4.0, 1.0, 2.0, 3.0]])
    out = top_k_filter(logits, 2)
    assert torch.equal(out[0], torch.tensor([-math.inf, 3.0, 2.0, -math.inf]))
    assert torch.equal(out[1], torch.tensor([4.0, -math.inf, -math.inf, 3.0]))
    assert torch.equal(top_k_filter(logits, 10), logits)


def test_top_p_filter():
    probs = torch.tensor([[0.1, 0.5, 0.25, 0.15]])
    logits = probs.log()
    out = top_p_filter(logits, 0.7)
    assert torch.isinf(out[0, 0]) and torch.isinf(out[0, 3])
    assert torch.allclose(out[0, [1, 2]], logits[0, [1, 2]])
    assert torch.isfinite(top_p_filter(logits, 0.75)[0]).sum() == 2, "0.5 + 0.25 reaches 0.75 exactly"
    assert torch.isfinite(top_p_filter(logits, 0.76)[0]).sum() == 3
    only_top = top_p_filter(torch.tensor([[0.0, 10.0, 0.0]]), 0.5)
    assert torch.isfinite(only_top).tolist() == [[False, True, False]], "always keep the top token"
    batch = torch.randn(5, 50, generator=gen(0))
    assert top_p_filter(batch, 1.0).isfinite().all()
    assert not contains(top_p_filter, LOOP_NODES)


def test_sample_next():
    logits = torch.tensor([[0.0, 5.0, 1.0], [3.0, 0.0, 0.0]])
    assert torch.equal(sample_next(logits, temperature=0), torch.tensor([[1], [0]]))
    draws = torch.cat([sample_next(logits, 1.0, top_k=1, generator=gen(i)) for i in range(20)], dim=1)
    assert (draws[0] == 1).all() and (draws[1] == 0).all()
    many = sample_next(torch.zeros(1, 4).expand(2000, 4), 1.0, generator=gen(1))
    assert many.shape == (2000, 1)
    counts = torch.bincount(many[:, 0], minlength=4).float() / 2000
    assert torch.allclose(counts, torch.full((4,), 0.25), atol=0.04)
    a = sample_next(logits, 0.5, top_p=0.9, generator=gen(2))
    assert torch.equal(a, sample_next(logits, 0.5, top_p=0.9, generator=gen(2)))


@torch.no_grad()
def test_cache_prefill_matches_full(model):
    idx = torch.randint(0, 40, (2, 10), generator=gen(0))
    full = model(idx)[0]
    logits, cache = forward_with_cache(model, idx)
    assert torch.allclose(logits, full, atol=1e-5)
    assert len(cache) == 2 and cache[0][0].shape == (2, 4, 10, 8)


@torch.no_grad()
def test_cache_incremental_matches_full(model):
    idx = torch.randint(0, 40, (2, 12), generator=gen(1))
    full = model(idx)[0]
    logits, cache = forward_with_cache(model, idx[:, :5])
    pieces = [logits]
    for t in range(5, 12, 3):                      # chunks of 3, 3, then 1
        logits, cache = forward_with_cache(model, idx[:, t:t + 3], cache)
        pieces.append(logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5), "positions and masks must line up"
    assert cache[1][1].shape[2] == 12
    with pytest.raises(ValueError):
        forward_with_cache(model, torch.zeros(1, 33, dtype=torch.long))


@torch.no_grad()
def naive_greedy(model, idx, n):
    for _ in range(n):
        nxt = model(idx)[0][:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def test_generate_cached(model):
    prompt = torch.randint(0, 40, (2, 6), generator=gen(2))
    model.train()
    out = generate_cached(model, prompt, 15, temperature=0)
    assert not model.training
    assert torch.equal(out, naive_greedy(model, prompt, 15))
    seen = []
    handle = model.blocks[0].attn.qkv.register_forward_hook(lambda m, i, o: seen.append(i[0].shape[1]))
    generate_cached(model, prompt, 5, temperature=0)
    handle.remove()
    assert seen[0] == 6 and set(seen[1:]) == {1}, "after the prefill, feed only the new token"
    a = generate_cached(model, prompt, 10, 1.0, top_p=0.9, generator=gen(3))
    assert torch.equal(a, generate_cached(model, prompt, 10, 1.0, top_p=0.9, generator=gen(3)))
    long = generate_cached(model, prompt, 100, temperature=0)
    assert long.shape[1] <= 33


def test_kv_cache_bytes():
    llama7b = GPTConfig(vocab_size=32000, block_size=4096, n_layer=32, n_head=32, d_model=4096)
    assert kv_cache_bytes(llama7b, 1, 1) == 2 * 32 * 4096 * 2
    assert kv_cache_bytes(llama7b, 1, 4096) / 2 ** 30 == pytest.approx(2.0)


def test_quantize_int8():
    w = torch.randn(16, 64, generator=gen(0))
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8 and q.shape == w.shape and scale.shape == (16, 1)
    assert q.abs().max() == 127
    assert torch.allclose(scale[:, 0], w.abs().amax(dim=1) / 127)
    deq = dequantize(q, scale)
    assert deq.dtype == torch.float32
    assert (deq - w).abs().max() <= scale.max() / 2 + 1e-6
    assert ((deq - w).norm() / w.norm()) < 0.01
    zq, zs = quantize_int8(torch.zeros(2, 4))
    assert torch.isfinite(dequantize(zq, zs)).all()


def test_quantize_int4_groups_and_outliers():
    w = torch.randn(8, 128, generator=gen(1))
    w[0, 5] = 40.0                                   # an outlier
    q, scale = quantize_int4_groups(w, 32)
    assert q.dtype == torch.int8 and q.shape == (8, 4, 32) and scale.shape == (8, 4, 1)
    assert q.min() >= -8 and q.max() <= 7
    deq = dequantize_int4_groups(q, scale)
    assert deq.shape == w.shape
    err_grouped = (deq[0, 32:] - w[0, 32:]).abs().mean()
    q_row, s_row = quantize_int4_groups(w, 128)      # one group per row
    err_row = (dequantize_int4_groups(q_row, s_row)[0, 32:] - w[0, 32:]).abs().mean()
    assert err_grouped < 0.3 * err_row, "groups stop an outlier from ruining the whole row"


def test_quantized_linear():
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    ql = QuantizedLinear(lin)
    assert {n for n, _ in ql.named_buffers()} == {"q", "scale"}
    assert [n for n, _ in ql.named_parameters()] == ["bias"]
    assert ql.q.dtype == torch.int8
    x = torch.randn(5, 64, generator=gen(1))
    rel = (ql(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.01


@torch.no_grad()
def test_quantize_model(model):
    original = copy.deepcopy(model)
    before = weight_bytes(model)
    assert before == sum(p.numel() * 4 for p in model.parameters())
    quantize_model(model)
    assert isinstance(model.blocks[0].attn.qkv, QuantizedLinear)
    assert isinstance(model.blocks[1].mlp.proj, QuantizedLinear)
    assert model.lm_head.weight is model.tok_emb.weight
    assert weight_bytes(model) < 0.6 * before
    idx = torch.randint(0, 40, (2, 16), generator=gen(4))
    a, b = original(idx)[0], model(idx)[0]
    assert (a - b).norm() / a.norm() < 0.05
