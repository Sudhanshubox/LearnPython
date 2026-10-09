import math

import numpy as np
import pytest

import exercises
from code_checks import has_loops, uses_any
from exercises import (
    accuracy_ci,
    bootstrap_ci,
    mean_ci,
    mle_bernoulli,
    mle_normal,
    nll,
    normal_log_likelihood,
    paired_bootstrap,
    pearson,
    permutation_test,
    standard_error,
)

rng = np.random.default_rng(43)


@pytest.mark.parametrize("name", ["bootstrap_ci", "permutation_test", "paired_bootstrap", "nll", "pearson"])
def test_vectorized(name):
    assert not has_loops(getattr(exercises, name))


def test_mle():
    assert mle_bernoulli(np.array([1, 0, 1, 1])) == pytest.approx(0.75)
    x = rng.normal(5.0, 2.0, 100_000)
    mu, sigma = mle_normal(x)
    assert mu == pytest.approx(5.0, abs=0.03)
    assert sigma == pytest.approx(2.0, abs=0.03)
    assert mle_normal(np.array([1.0, 3.0]))[1] == pytest.approx(1.0)      # divides by n


def test_normal_log_likelihood_is_maximized_at_mle():
    x = rng.normal(1.0, 3.0, 500)
    mu, sigma = mle_normal(x)
    best = normal_log_likelihood(x, mu, sigma)
    for dmu, dsig in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
        assert normal_log_likelihood(x, mu + dmu, sigma + dsig) < best
    assert normal_log_likelihood(np.array([0.0]), 0.0, 1.0) == pytest.approx(-0.5 * math.log(2 * math.pi))


def test_normal_log_likelihood_no_underflow():
    x = rng.normal(0, 1, 5000)
    assert np.isfinite(normal_log_likelihood(x, 0.0, 0.01))


def test_nll():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert nll(probs, np.array([0, 2])) == pytest.approx(-(math.log(0.7) + math.log(0.8)) / 2)
    assert np.isfinite(nll(np.array([[1.0, 0.0]]), np.array([1])))


def test_standard_error_and_mean_ci():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert standard_error(x) == pytest.approx(np.std(x, ddof=1) / math.sqrt(8))
    low, high = mean_ci(x)
    assert (low + high) / 2 == pytest.approx(5.0)
    assert high - low == pytest.approx(2 * 1.96 * standard_error(x))


def test_mean_ci_coverage():
    """About 95% of intervals from repeated experiments should contain the true mean."""
    hits = 0
    for i in range(1000):
        lo, hi = mean_ci(np.random.default_rng(i).normal(10, 3, 50))
        hits += lo <= 10 <= hi
    assert 0.92 < hits / 1000 < 0.97


def test_accuracy_ci():
    correct = np.array([True] * 800 + [False] * 200)
    acc, lo, hi = accuracy_ci(correct)
    assert acc == pytest.approx(0.8)
    assert hi - acc == pytest.approx(1.96 * math.sqrt(0.8 * 0.2 / 1000))


def test_bootstrap_ci():
    x = rng.normal(0, 1, 400)
    lo, hi = bootstrap_ci(x, np.mean, seed=1)
    assert lo < x.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 / math.sqrt(400), rel=0.15)
    lo_m, hi_m = bootstrap_ci(x, np.median, seed=1)
    assert lo_m < np.median(x) < hi_m
    assert bootstrap_ci(x, np.mean, seed=1) == bootstrap_ci(x, np.mean, seed=1)


def test_permutation_test():
    same_a, same_b = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
    diff_a, diff_b = rng.normal(0, 1, 50), rng.normal(1, 1, 50)
    assert permutation_test(same_a, same_b, seed=2) > 0.05
    p = permutation_test(diff_a, diff_b, seed=2)
    assert p < 0.01
    assert p >= 1 / 5001, "the smallest possible p-value is 1 / (n_perm + 1)"


def test_permutation_test_is_calibrated():
    """When there's no real difference, p < 0.05 should happen about 5% of the time."""
    false_alarms = sum(
        permutation_test(np.random.default_rng(i).normal(0, 1, 20), np.random.default_rng(1000 + i).normal(0, 1, 20), n_perm=500, seed=i) < 0.05
        for i in range(200)
    )
    assert false_alarms / 200 < 0.10


def test_paired_bootstrap():
    n = 2000
    a = rng.random(n) < 0.80
    b = a.copy()
    flip = rng.choice(n, 60, replace=False)            # B fixes some of A's mistakes...
    b[flip] = True
    diff, lo, hi = paired_bootstrap(a, b, seed=3)
    assert diff == pytest.approx(b.mean() - a.mean())
    assert lo > 0, "B is consistently better on the same examples"
    noise_b = rng.random(n) < 0.80                     # an unrelated model with the same skill
    _, lo2, hi2 = paired_bootstrap(a, noise_b, seed=3)
    assert lo2 < 0 < hi2


def test_pearson():
    x = rng.normal(size=500)
    assert pearson(x, 2 * x + 1) == pytest.approx(1)
    assert pearson(x, -x) == pytest.approx(-1)
    y = rng.normal(size=500)
    assert pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert not uses_any(exercises.pearson, "corrcoef")
