"""m67 exercises: n-gram and neural character-level language models."""

import math
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. (stoi, itos) for the sorted unique characters of text: itos is a list, stoi a dict.
def build_vocab(text):
    raise NotImplementedError


# 2. Split a list of ids by position: the first int(len * (1 - val_fraction)) for training,
#    the rest for validation. Return (train_ids, val_ids).
def split_ids(ids, val_fraction=0.1):
    raise NotImplementedError


# 3. A (V, V) float tensor where [a, b] counts how often b follows a in ids. No Python loops
#    over the ids: use torch indexing (e.g. index_put_ with accumulate=True).
def bigram_counts(ids, V):
    raise NotImplementedError


# 4. Add-alpha smoothed probabilities: each row of (counts + alpha) normalized to sum to 1.
def bigram_probs(counts, alpha=1.0):
    raise NotImplementedError


# 5. Average negative log-likelihood (natural log) of every transition ids[t] -> ids[t+1]
#    under a (V, V) probability table, as a float. perplexity(nll) = exp(nll).
def bigram_nll(probs, ids):
    raise NotImplementedError


def perplexity(nll):
    raise NotImplementedError


# 6. A general n-gram model with add-alpha smoothing (README section 2).
#    fit(ids): for each position i >= n-1, count ids[i] after the context tuple(ids[i-n+1:i])
#      (the empty tuple when n == 1). Store counts in self.counts[context][token]. Return self.
#    prob(context, token): (count + alpha) / (context total + alpha * V), using only the last
#      n-1 tokens of the context; unseen contexts give 1 / V.
#    nll(ids): the average -log prob over positions i >= n-1 (float).
class NGramModel:
    def __init__(self, n, V, alpha=1.0):
        raise NotImplementedError

    def fit(self, ids):
        raise NotImplementedError

    def prob(self, context, token):
        raise NotImplementedError

    def nll(self, ids):
        raise NotImplementedError


# 7. A neural bigram: self.logits = nn.Embedding(V, V) initialized to all zeros (so it starts
#    uniform). forward(x) maps token ids (any shape) to next-token logits (shape + (V,)).
class NeuralBigram(nn.Module):
    def __init__(self, V):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 8. Sliding-window training data: X[t] = ids[t : t + block_size], Y[t] = ids[t + block_size],
#    for every t with a target. Return int64 tensors X (n, block_size) and Y (n,). No loops.
def make_context_dataset(ids, block_size):
    raise NotImplementedError


# 9. Bengio's MLP language model (README section 5): self.embed = nn.Embedding(V, embed_dim),
#    self.hidden = nn.Linear(block_size * embed_dim, hidden), self.out = nn.Linear(hidden, V).
#    forward(x) with x (B, block_size) returns logits (B, V), using tanh in the hidden layer.
class MLPLanguageModel(nn.Module):
    def __init__(self, V, block_size, embed_dim=16, hidden=128):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 10. Train any of these models: each step samples a random mini-batch of indices with
#     torch.randint(0, len(X), (batch_size,), generator=g) (g seeded with seed), computes
#     cross-entropy, and takes an AdamW(lr, weight_decay=0.0) step. Return the list of
#     per-step losses (floats).
#     evaluate_nll: average cross-entropy over ALL of (X, Y) in eval mode without gradients
#     (process it in chunks of batch_size), as a float; put the model back in train mode.
def train_lm(model, X, Y, steps=1000, batch_size=128, lr=1e-2, seed=0):
    raise NotImplementedError


def evaluate_nll(model, X, Y, batch_size=4096):
    raise NotImplementedError


# 11. The indices of the k rows of `embeddings` most cosine-similar to row `index`,
#     most similar first, excluding `index` itself.
def nearest_neighbors(embeddings, index, k=5):
    raise NotImplementedError
