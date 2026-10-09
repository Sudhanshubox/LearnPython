"""m69 exercises: a GPT-style Transformer from its parts."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Given: multi-head causal self-attention from m68, using PyTorch's fast kernel."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


# 1. Sinusoidal positional encodings (README section 4): a (T, d) float tensor with
#    pe[p, 2i] = sin(p / 10000 ** (2i / d)) and pe[p, 2i + 1] = cos(the same angle). d is even.
def sinusoidal_positions(T, d):
    raise NotImplementedError


# 2. RMSNorm over the last dimension with a learnable gain self.weight (ones) and eps.
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 3. Rotary position embedding for x of shape (..., T, d), d even (README section 4):
#    half = d // 2, freqs[i] = base ** (-i / half) for i in range(half), angle[p, i] = p * freqs[i],
#    x1 = x[..., :half], x2 = x[..., half:], result = cat(x1*cos - x2*sin, x1*sin + x2*cos).
def rope(x, base=10000.0):
    raise NotImplementedError


# 4. The MLP: self.fc = Linear(d_model, 4 * d_model), GELU, self.proj = Linear(4 * d_model,
#    d_model), then dropout.
class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 5. A pre-norm Transformer block: self.ln1, self.attn (CausalSelfAttention), self.ln2,
#    self.mlp, with x = x + attn(ln1(x)); x = x + mlp(ln2(x)). Use nn.LayerNorm.
class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0


# 6. The GPT model (README sections 1, 5, 6).
#    Attributes: self.config, self.tok_emb (Embedding V x d), self.pos_emb (Embedding
#    block_size x d), self.drop (Dropout), self.blocks (ModuleList of Blocks), self.ln_f
#    (LayerNorm), self.lm_head (Linear d -> V, no bias) whose weight IS tok_emb.weight.
#    Init: every Linear and Embedding weight ~ N(0, 0.02^2), Linear biases zero, then every
#    parameter whose name ends with "proj.weight" ~ N(0, (0.02 / sqrt(2 * n_layer))^2).
#    forward(idx, targets=None): raise ValueError if T > block_size; return (logits (B, T, V),
#    loss) where loss is the mean cross-entropy against targets (B, T), or None.
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError


# 7. The number of parameters (shared parameters count once: model.parameters() already
#    does that).
def count_parameters(model):
    raise NotImplementedError


# 8. The same number computed from the config alone with the README section 7 formula.
def estimate_parameters(config):
    raise NotImplementedError
