import numpy as np
import pytest
import torch

from code_checks import LOOP_NODES, contains
from exercises import (
    CHINCHILLA_PUBLISHED,
    CHINCHILLA_REPLICATED,
    chinchilla_loss,
    compute_optimal,
    fit_power_law,
    fit_power_law_with_offset,
    load_char_data,
    non_embedding_params,
    scaling_experiment,
)
from minigpt import GPT, GPTConfig


def test_fit_power_law():
    x = np.array([1e3, 1e4, 1e5, 1e6])
    a, alpha = fit_power_law(x, 5.0 * x ** -0.3)
    assert a == pytest.approx(5.0, rel=1e-6) and alpha == pytest.approx(0.3, rel=1e-6)
    assert isinstance(a, float) and isinstance(alpha, float)


def test_fit_power_law_with_offset():
    x = np.logspace(2, 7, 12)
    y = 1.5 + 3.0 * x ** -0.4
    a, alpha, c = fit_power_law_with_offset(x, y, np.linspace(0, 2, 201))
    assert c == pytest.approx(1.5, abs=0.011)
    assert alpha == pytest.approx(0.4, abs=0.02)
    _, alpha0 = fit_power_law(x, y)
    assert alpha0 < 0.1, "ignoring the irreducible loss makes the exponent look tiny"


def test_chinchilla_loss():
    p = CHINCHILLA_REPLICATED
    assert chinchilla_loss(1e9, 2e10, p) == pytest.approx(1.8172 + 482.01 / 1e9 ** 0.3478 + 2085.43 / 2e10 ** 0.3658)
    arr = chinchilla_loss(np.array([1e8, 1e9]), np.array([1e10, 1e10]), p)
    assert arr.shape == (2,) and arr[0] > arr[1]


def test_compute_optimal():
    C = 5.76e23                                     # Chinchilla's own budget
    N, D, L = compute_optimal(C, CHINCHILLA_REPLICATED)
    assert 5e10 < N < 1e11, "about Chinchilla's 70B parameters"
    assert 15 < D / N < 25, "about 20 tokens per parameter"
    assert 6 * N * D == pytest.approx(C, rel=1e-6)
    Np, Dp, _ = compute_optimal(C, CHINCHILLA_PUBLISHED)
    assert Dp / Np > 60, "the published constants imply a very different ratio"
    small = compute_optimal(1e20, CHINCHILLA_REPLICATED)
    assert small[0] < N and small[2] > L
    assert not contains(compute_optimal, LOOP_NODES)


def test_non_embedding_params():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=100, block_size=32, n_layer=1, n_head=2, d_model=16)
    model = GPT(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert non_embedding_params(model) == total - 100 * 16 - 32 * 16


def test_load_char_data():
    V, train, val = load_char_data("phase01*/m01*/README.md")
    assert train.dtype == torch.int64 and V == int(torch.cat([train, val]).max()) + 1
    assert len(val) == pytest.approx(0.1 * (len(train) + len(val)), abs=1)


@pytest.mark.timeout(120)
def test_scaling_experiment():
    V, train, val = load_char_data()
    results = scaling_experiment([8, 16, 32, 64], V, train, val, steps=800)
    params = [n for n, _ in results]
    losses = [l for _, l in results]
    assert params == sorted(params) and params[0] == 888
    assert all(isinstance(l, float) for l in losses)
    assert losses[0] > losses[1] > losses[2] > losses[3], "bigger models reach lower loss"
    a, alpha = fit_power_law(params, losses)
    assert 0.02 < alpha < 0.2
    predicted = a * np.array(params, dtype=float) ** -alpha
    assert np.max(np.abs(predicted - losses)) < 0.08, "a power law describes the trend"
