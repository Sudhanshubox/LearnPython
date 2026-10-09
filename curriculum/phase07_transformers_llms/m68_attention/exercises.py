"""m68 exercises: attention from scratch.

Don't use F.scaled_dot_product_attention or nn.MultiheadAttention. Masks: True = allowed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Scaled dot-product attention for inputs with any leading dims: q (..., Tq, d),
#    k (..., Tk, d), v (..., Tk, dv). mask is None or a boolean tensor broadcastable to
#    (..., Tq, Tk) where True means "may attend"; disallowed scores become -inf before the
#    softmax. Return (output (..., Tq, dv), weights (..., Tq, Tk)).
def attention(q, k, v, mask=None):
    raise NotImplementedError


# 2. The (T, T) boolean causal mask: True where key j <= query i.
def causal_mask(T):
    raise NotImplementedError


# 3. A key-padding mask of shape (B, 1, 1, T): True for positions < lengths[b]. No loops.
def padding_mask(lengths, T):
    raise NotImplementedError


# 4. One self-attention head: self.query, self.key, self.value = nn.Linear(d_model, head_size,
#    bias=False). forward(x) with x (B, T, d_model) returns (B, T, head_size), using the
#    causal mask when self.causal.
class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, head_size, causal=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 5. Multi-head self-attention (README section 5): self.qkv = nn.Linear(d_model, 3 * d_model)
#    whose output splits into q, k, v in that order, and self.proj = nn.Linear(d_model, d_model).
#    forward(x, mask=None): mask (if given) is broadcastable to (B, h, T, T); when self.causal,
#    combine it with the causal mask (logical and). No loops over heads.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, mask=None):
        raise NotImplementedError


# 6. The cost of one multi-head attention operation (excluding the linear projections) for a
#    sequence of length T: {"flops": 4 * T * T * d_model, "score_entries": n_heads * T * T}.
def attention_cost(T, d_model, n_heads):
    raise NotImplementedError


# 7. Exact attention computed block by block over the keys with the online softmax
#    (README section 7), never building the full (Tq, Tk) score matrix. No mask.
#    q (..., Tq, d), k (..., Tk, d), v (..., Tk, dv). Return the output (..., Tq, dv).
def tiled_attention(q, k, v, block_size):
    raise NotImplementedError
