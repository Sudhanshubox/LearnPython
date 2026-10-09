"""minigpt: a small GPT (the model you built in m69), given here so later modules share it.

Usage:
    from minigpt import GPT, GPTConfig
    model = GPT(GPTConfig(vocab_size=..., block_size=..., n_layer=..., n_head=..., d_model=...))
    logits, loss = model(idx, targets)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (m68), using PyTorch's fast kernel."""

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


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config.d_model, config.n_head, config.dropout) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
