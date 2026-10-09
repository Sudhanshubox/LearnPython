import math

import numpy as np
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    cross_entropy,
    cross_entropy_from_logits,
    cross_entropy_grad,
    distillation_loss,
    entropy_bits,
    kl_divergence,
    log_softmax,
    logsumexp,
    mutual_information,
    perplexity,
)

rng = np.random.default_rng(44)


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("name", [
    "entropy_bits", "cross_entropy", "kl_divergence", "logsumexp", "log_softmax",
    "cross_entropy_from_logits", "cross_entropy_grad", "perplexity", "distillation_loss", "mutual_information",
])
def test_no_loops(name):
    assert not has_loops(getattr(exercises, name))


def test_entropy_bits():
    assert entropy_bits(np.array([0.5, 0.5])) == pytest.approx(1)
    assert entropy_bits(np.array([0.9, 0.1])) == pytest.approx(0.469, abs=1e-3)
    assert entropy_bits(np.full(8, 1 / 8)) == pytest.approx(3)
    assert entropy_bits(np.array([1.0, 0.0, 0.0])) == 0
    assert not np.isnan(entropy_bits(np.array([0.0, 1.0])))


def test_cross_entropy_and_kl():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    assert cross_entropy(p, q) == pytest.approx(-(p * np.log(q)).sum())
    assert kl_divergence(p, q) == pytest.approx(cross_entropy(p, q) - cross_entropy(p, p))
    assert kl_divergence(p, p) == pytest.approx(0, abs=1e-12)
    assert kl_divergence(p, q) > 0
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p)), "KL is not symmetric"
    onehot = np.array([0.0, 1.0, 0.0])
    assert cross_entropy(onehot, q) == pytest.approx(-math.log(0.4))
    assert np.isfinite(kl_divergence(p, np.array([1.0, 0.0, 0.0])))


def test_logsumexp():
    z = rng.normal(size=(3, 5))
    np.testing.assert_allclose(logsumexp(z), np.log(np.exp(z).sum(axis=-1)))
    np.testing.assert_allclose(logsumexp(z, axis=0), np.log(np.exp(z).sum(axis=0)))
    assert logsumexp(np.array([1000.0, 1000.0])) == pytest.approx(1000 + math.log(2))
    assert logsumexp(np.array([-1000.0, -1000.0])) == pytest.approx(-1000 + math.log(2))


def test_log_softmax():
    z = rng.normal(size=(4, 6))
    np.testing.assert_allclose(log_softmax(z), np.log(softmax(z)))
    big = log_softmax(np.array([[1000.0, 0.0]]))
    assert np.isfinite(big).all()
    np.testing.assert_allclose(big, [[0.0, -1000.0]])


def test_cross_entropy_from_logits():
    logits = rng.normal(size=(8, 5))
    labels = rng.integers(0, 5, 8)
    expected = -np.log(softmax(logits)[np.arange(8), labels]).mean()
    assert cross_entropy_from_logits(logits, labels) == pytest.approx(expected)
    assert cross_entropy_from_logits(np.array([[1000.0, 0.0]]), np.array([1])) == pytest.approx(1000)
    uniform = cross_entropy_from_logits(np.zeros((10, 7)), rng.integers(0, 7, 10))
    assert uniform == pytest.approx(math.log(7)), "uniform predictions give ln(k)"


def test_cross_entropy_grad():
    logits = rng.normal(size=(4, 3))
    labels = np.array([0, 2, 1, 1])
    g = cross_entropy_grad(logits, labels)
    h = 1e-6
    num = np.zeros_like(logits)
    for i in range(4):
        for j in range(3):
            e = np.zeros_like(logits)
            e[i, j] = h
            num[i, j] = (cross_entropy_from_logits(logits + e, labels) - cross_entropy_from_logits(logits - e, labels)) / (2 * h)
    np.testing.assert_allclose(g, num, atol=1e-7)


def test_perplexity():
    assert perplexity(np.full(10, 0.25)) == pytest.approx(4)
    assert perplexity(np.array([1.0, 1.0])) == pytest.approx(1)
    assert perplexity(np.array([0.5, 0.125])) == pytest.approx(4)       # geometric mean of 1/p


def test_distillation_loss():
    t = rng.normal(size=(5, 4))
    assert distillation_loss(t, t) == pytest.approx(0, abs=1e-12)
    s = rng.normal(size=(5, 4))
    T = 3.0
    p, q = softmax(t / T), softmax(s / T)
    expected = (p * (np.log(p) - np.log(q))).sum(axis=1).mean() * T ** 2
    assert distillation_loss(t, s, T) == pytest.approx(expected)


def test_mutual_information():
    independent = np.outer([0.3, 0.7], [0.5, 0.5])
    assert mutual_information(independent) == pytest.approx(0, abs=1e-12)
    identical = np.array([[0.5, 0.0], [0.0, 0.5]])                   # Y is a copy of X
    assert mutual_information(identical) == pytest.approx(math.log(2))
    P = rng.random((3, 4))
    P /= P.sum()
    assert mutual_information(P) >= 0
