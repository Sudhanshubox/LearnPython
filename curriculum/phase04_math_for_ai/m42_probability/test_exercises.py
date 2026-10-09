import math

import numpy as np
import pytest

import exercises
from code_checks import has_loops, uses_any
from exercises import (
    birthday_exact,
    birthday_simulated,
    estimate_pi,
    mean_and_variance,
    normal_pdf,
    posterior_positive,
    prob_dice_sum,
    running_mean_of_die,
    sample_categorical,
    sample_means,
    temperature_probs,
)


@pytest.mark.parametrize("name", ["prob_dice_sum", "estimate_pi", "running_mean_of_die", "sample_means", "sample_categorical", "birthday_simulated"])
def test_vectorized_simulation(name):
    assert not has_loops(getattr(exercises, name))


def test_posterior_positive():
    assert posterior_positive(0.01, 0.99, 0.05) == pytest.approx(0.0099 / 0.0594)
    assert posterior_positive(0.5, 0.9, 0.1) == pytest.approx(0.9)
    assert posterior_positive(0.0, 0.99, 0.05) == 0


def test_mean_and_variance():
    m, v = mean_and_variance(np.arange(1, 7), np.full(6, 1 / 6))
    assert m == pytest.approx(3.5)
    assert v == pytest.approx(35 / 12)
    m, v = mean_and_variance(np.array([0, 1]), np.array([0.7, 0.3]))
    assert (m, v) == (pytest.approx(0.3), pytest.approx(0.21))


def test_prob_dice_sum():
    assert prob_dice_sum(2, 7) == pytest.approx(1 / 6, abs=0.005)
    assert prob_dice_sum(2, 12) == pytest.approx(1 / 36, abs=0.003)
    assert prob_dice_sum(3, 2) == 0
    assert prob_dice_sum(2, 7, seed=1) == prob_dice_sum(2, 7, seed=1)


def test_estimate_pi():
    assert estimate_pi(1_000_000) == pytest.approx(math.pi, abs=0.01)
    assert abs(estimate_pi(100) - math.pi) > abs(estimate_pi(1_000_000) - math.pi) - 0.01


def test_normal_pdf():
    assert normal_pdf(np.array([0.0]))[0] == pytest.approx(1 / math.sqrt(2 * math.pi))
    x = np.linspace(-20, 20, 40001)
    assert np.trapezoid(normal_pdf(x, 1.0, 2.0), x) == pytest.approx(1, abs=1e-6)
    assert normal_pdf(np.array([3.0]), 3.0, 0.5)[0] == pytest.approx(1 / (0.5 * math.sqrt(2 * math.pi)))


def test_law_of_large_numbers():
    means = running_mean_of_die(100_000, seed=3)
    assert len(means) == 100_000
    rolls = np.random.default_rng(3).integers(1, 7, 100_000)
    assert means[0] == rolls[0]
    assert means[9] == pytest.approx(rolls[:10].mean())
    assert means[-1] == pytest.approx(3.5, abs=0.02)
    assert abs(means[-1] - 3.5) < abs(means[99] - 3.5) + 0.05


def test_central_limit_theorem():
    for n in (4, 64):
        m = sample_means(n, 20_000, seed=n)
        assert m.shape == (20_000,)
        assert m.mean() == pytest.approx(1.0, abs=0.02)
        assert m.std() == pytest.approx(1 / math.sqrt(n), rel=0.05)    # sigma / sqrt(n), sigma = 1
    # skewness shrinks as n grows: the distribution of means becomes symmetric (normal)
    skew = lambda a: np.mean(((a - a.mean()) / a.std()) ** 3)
    assert skew(sample_means(64, 20_000)) < skew(sample_means(2, 20_000)) / 3


def test_sample_categorical():
    probs = np.array([0.1, 0.6, 0.3])
    draws = sample_categorical(probs, 100_000, np.random.default_rng(0))
    assert draws.shape == (100_000,)
    freq = np.bincount(draws, minlength=3) / 100_000
    np.testing.assert_allclose(freq, probs, atol=0.01)
    assert not uses_any(exercises.sample_categorical, "choice")


def test_temperature_probs():
    logits = np.array([2.0, 1.0, 0.0])
    p1 = temperature_probs(logits, 1.0)
    np.testing.assert_allclose(p1, np.exp(logits) / np.exp(logits).sum())
    cold, hot = temperature_probs(logits, 0.5), temperature_probs(logits, 5.0)
    assert cold[0] > p1[0] > hot[0]
    assert hot.max() - hot.min() < p1.max() - p1.min()
    np.testing.assert_array_equal(temperature_probs(logits, 0), [1, 0, 0])
    np.testing.assert_array_equal(temperature_probs(np.array([1.0, 3.0, 3.0]), 0), [0, 1, 0])
    assert np.isfinite(temperature_probs(np.array([1000.0, 0.0]), 0.01)).all()


def test_birthday():
    assert birthday_exact(23) == pytest.approx(0.5073, abs=1e-4)
    assert birthday_exact(1) == 0
    assert birthday_simulated(23, seed=2) == pytest.approx(birthday_exact(23), abs=0.015)
    assert birthday_simulated(50, seed=2) == pytest.approx(birthday_exact(50), abs=0.01)
