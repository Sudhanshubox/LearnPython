import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from code_checks import LOOP_NODES, contains, names_used
from exercises import (
    MultiHeadAttention,
    SelfAttentionHead,
    attention,
    attention_cost,
    causal_mask,
    padding_mask,
    tiled_attention,
)

FORBIDDEN = {"scaled_dot_product_attention", "MultiheadAttention"}


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def rand(*shape, seed=0):
    return torch.randn(*shape, generator=gen(seed))


def test_attention_basic():
    q, k, v = rand(2, 3, 5, 8, seed=0), rand(2, 3, 7, 8, seed=1), rand(2, 3, 7, 4, seed=2)
    out, w = attention(q, k, v)
    assert out.shape == (2, 3, 5, 4) and w.shape == (2, 3, 5, 7)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5))
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    expected = torch.softmax(q[0, 0] @ k[0, 0].T / math.sqrt(8), dim=-1)
    assert torch.allclose(w[0, 0], expected, atol=1e-6), "scale the scores by 1/sqrt(d)"
    assert not FORBIDDEN & names_used(attention)


def test_attention_with_mask():
    q, k, v = rand(2, 6, 8, seed=0), rand(2, 6, 8, seed=1), rand(2, 6, 8, seed=2)
    mask = causal_mask(6)
    out, w = attention(q, k, v, mask)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5)
    assert (w[:, ~mask] == 0).all()
    assert torch.allclose(out[:, 0], v[:, 0], atol=1e-6), "the first position can only see itself"


def test_soft_lookup():
    keys = torch.eye(4) * 20                        # very distinct keys
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    out, _ = attention(keys[2:3] * 1.0, keys, values)
    assert out.item() == pytest.approx(3.0, abs=1e-3), "a query matching key 2 retrieves value 2"


def test_causal_mask():
    m = causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist() == [[True, False, False, False], [True, True, False, False],
                          [True, True, True, False], [True, True, True, True]]


def test_padding_mask():
    m = padding_mask([3, 1], 4)
    assert m.shape == (2, 1, 1, 4) and m.dtype == torch.bool
    assert m[:, 0, 0].tolist() == [[True, True, True, False], [True, False, False, False]]
    assert not contains(padding_mask, LOOP_NODES)
    q = k = v = rand(2, 1, 4, 8)
    out, w = attention(q, k, v, m)
    assert (w[1, 0, :, 1:] == 0).all()


def test_self_attention_head():
    torch.manual_seed(0)
    head = SelfAttentionHead(16, 8)
    for lin in (head.query, head.key, head.value):
        assert isinstance(lin, nn.Linear) and lin.bias is None and lin.weight.shape == (8, 16)
    x = rand(2, 5, 16)
    out = head(x)
    assert out.shape == (2, 5, 8)
    expected = F.scaled_dot_product_attention(head.query(x), head.key(x), head.value(x), is_causal=True)
    assert torch.allclose(out, expected, atol=1e-5)
    full = SelfAttentionHead(16, 8, causal=False)
    full.load_state_dict(head.state_dict())
    assert torch.allclose(full(x), F.scaled_dot_product_attention(head.query(x), head.key(x), head.value(x)), atol=1e-5)


def test_causality():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4)
    x = rand(1, 6, 16)
    y = x.clone()
    y[0, 4:] += 10.0                               # change the future
    assert torch.allclose(mha(x)[0, :4], mha(y)[0, :4], atol=1e-5), "outputs must not depend on later tokens"


def test_permutation_equivariance():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 4, causal=False)
    x = rand(1, 6, 16)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    assert torch.allclose(mha(x)[:, perm], mha(x[:, perm]), atol=1e-5), "attention alone ignores order"


@pytest.mark.parametrize("causal", [False, True])
def test_multihead_matches_pytorch(causal):
    torch.manual_seed(0)
    mine = MultiHeadAttention(32, 4, causal=causal)
    assert isinstance(mine.qkv, nn.Linear) and mine.qkv.weight.shape == (96, 32)
    ref = nn.MultiheadAttention(32, 4, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(mine.qkv.weight)
        ref.in_proj_bias.copy_(mine.qkv.bias)
        ref.out_proj.weight.copy_(mine.proj.weight)
        ref.out_proj.bias.copy_(mine.proj.bias)
    x = rand(3, 7, 32, seed=1)
    attn_mask = ~causal_mask(7) if causal else None              # PyTorch: True = NOT allowed
    expected, _ = ref(x, x, x, attn_mask=attn_mask, need_weights=False)
    assert torch.allclose(mine(x), expected, atol=1e-5)
    assert not FORBIDDEN & names_used(MultiHeadAttention.forward)


def test_multihead_padding_mask():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 2, causal=False)
    x = rand(2, 5, 16)
    pm = padding_mask([3, 5], 5)
    out = mha(x, pm)
    x2 = x.clone()
    x2[0, 3:] = 99.0                               # padding content must not matter
    assert torch.allclose(mha(x2, pm)[0, :3], out[0, :3], atol=1e-5)


def test_attention_cost():
    c = attention_cost(1024, 768, 12)
    assert c == {"flops": 4 * 1024 * 1024 * 768, "score_entries": 12 * 1024 * 1024}
    assert attention_cost(2048, 768, 12)["flops"] == 4 * c["flops"], "quadratic in T"


class ShapeRecorder(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.shapes.append(tuple(out.shape))
        return out


@pytest.mark.parametrize("block", [1, 3, 8, 64])
def test_tiled_attention(block):
    q, k, v = rand(2, 5, 4, seed=0) * 3, rand(2, 20, 4, seed=1) * 3, rand(2, 20, 6, seed=2)
    expected = F.scaled_dot_product_attention(q, k, v)
    recorder = ShapeRecorder()
    with recorder:
        out = tiled_attention(q, k, v, block)
    assert torch.allclose(out, expected, atol=1e-5)
    if block < 20:
        assert all(s[-2:] != (5, 20) for s in recorder.shapes), "never build the full score matrix"
    big = tiled_attention(q * 100, k, v, 4)
    assert torch.isfinite(big).all(), "subtract the running max for stability"
