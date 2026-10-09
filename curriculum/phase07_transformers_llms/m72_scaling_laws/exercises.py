"""m72 exercises: power laws, Chinchilla, and a small scaling experiment."""

import math
from pathlib import Path

import numpy as np
import torch

from minigpt import GPT, GPTConfig

CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())

CHINCHILLA_PUBLISHED = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}
CHINCHILLA_REPLICATED = {"E": 1.8172, "A": 482.01, "B": 2085.43, "alpha": 0.3478, "beta": 0.3658}


# 1. Fit y = a * x ** (-alpha) by linear regression on (log x, log y) (np.polyfit with
#    degree 1 is fine). Return (a, alpha) as floats.
def fit_power_law(x, y):
    raise NotImplementedError


# 2. Fit y = c + a * x ** (-alpha): for each candidate c in c_values that is below min(y),
#    fit a power law to y - c, and measure the mean squared error between
#    log(c + a * x ** -alpha) and log(y). Return (a, alpha, c) for the best c (floats).
def fit_power_law_with_offset(x, y, c_values):
    raise NotImplementedError


# 3. The Chinchilla parametric loss L(N, D) = E + A / N**alpha + B / D**beta for a constants
#    dict like CHINCHILLA_REPLICATED. Must work with NumPy arrays for N and D.
def chinchilla_loss(N, D, p):
    raise NotImplementedError


# 4. The compute-optimal allocation for budget C (FLOPs): over the grid of model sizes
#    N (default np.logspace(6, 13, 7001)), with D = C / (6 N), find the N minimizing
#    chinchilla_loss. Return (N_opt, D_opt, loss) as floats. No Python loops over the grid.
def compute_optimal(C, p, n_grid=None):
    raise NotImplementedError


# 5. The number of non-embedding parameters of a minigpt GPT: all parameters except
#    tok_emb.weight (shared with lm_head) and pos_emb.weight.
def non_embedding_params(model):
    raise NotImplementedError


# 6. Character-level data from the lessons matching pattern: join the files (sorted paths)
#    with "\n\n", vocabulary = sorted unique characters, split by position.
#    Return (vocab_size, train_data, val_data) with int64 tensors.
def load_char_data(pattern="phase0[1-3]*/*/README.md", val_fraction=0.1):
    raise NotImplementedError


# 7. Train one model and measure it. torch.manual_seed(seed); model = GPT(config);
#    AdamW(lr, weight_decay=0.0) with CosineAnnealingLR(T_max=steps) stepped every step;
#    random windows of length block_size (targets shifted by one) using a generator seeded
#    with seed. Then, in eval mode without gradients, the validation loss is the mean over 20
#    batches of 64 windows drawn with a generator seeded with 1234.
#    Return (non_embedding_params(model), validation loss as a float).
def train_and_evaluate(config, train_data, val_data, steps=800, batch_size=32, lr=3e-3, seed=0):
    raise NotImplementedError


# 8. For each width d, train GPTConfig(vocab_size, block_size=32, n_layer=1, n_head=2,
#    d_model=d) with train_and_evaluate(..., steps=steps). Return the list of
#    (params, val_loss) pairs in the order of widths.
def scaling_experiment(widths, vocab_size, train_data, val_data, steps=800):
    raise NotImplementedError
