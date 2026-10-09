"""m43 exercises: statistics. Inputs are NumPy arrays.

Resampling and permutations must be vectorized (draw all indices at once) and use
np.random.default_rng(seed).
"""

import math

import numpy as np


# 1. Maximum-likelihood estimates.
#    mle_bernoulli(x): x is an array of 0/1 outcomes -> p_hat.
#    mle_normal(x): -> (mu_hat, sigma_hat) with sigma_hat dividing by n (not n - 1).
def mle_bernoulli(x):
    raise NotImplementedError


def mle_normal(x):
    raise NotImplementedError


# 2. Total log-likelihood of data x under Normal(mu, sigma):
#    sum of log p(x_i) using the normal density formula (compute the log directly:
#    -0.5*log(2*pi*sigma^2) - (x - mu)^2 / (2*sigma^2)).
def normal_log_likelihood(x, mu, sigma):
    raise NotImplementedError


# 3. Negative log-likelihood (= cross-entropy loss) of a classifier.
#    probs: (n, k) predicted probabilities (rows sum to 1), labels: (n,) true classes.
#    Return the mean of -log(probs[i, labels[i]]), clipping probabilities to >= 1e-12.
def nll(probs, labels):
    raise NotImplementedError


# 4. Standard error of the mean (sample std with ddof=1, over sqrt(n)), and a normal
#    approximation confidence interval (mean - z*SE, mean + z*SE).
def standard_error(x):
    raise NotImplementedError


def mean_ci(x, z=1.96):
    raise NotImplementedError


# 5. Accuracy with a confidence interval: `correct` is a boolean array (one per test
#    example). Return (accuracy, low, high) using SE = sqrt(p (1 - p) / n).
def accuracy_ci(correct, z=1.96):
    raise NotImplementedError


# 6. Percentile bootstrap confidence interval for any statistic.
#    stat is a function applied along axis=1 of a 2-D array, like np.mean or np.median
#    (call it as stat(resamples, axis=1)). Draw ALL indices at once with
#    rng.integers(0, n, size=(n_boot, n)). Return (low, high) at the alpha/2 and
#    1 - alpha/2 percentiles.
def bootstrap_ci(x, stat, n_boot=2000, alpha=0.05, seed=0):
    raise NotImplementedError


# 7. Two-sided permutation test for a difference in means between groups a and b.
#    For each of n_perm permutations, shuffle the pooled data (rng.permuted on a 2-D
#    array of n_perm copies, along axis=1), split it into the first len(a) and the rest,
#    and compute the difference in means. Return the p-value
#        (count of |perm_diff| >= |observed_diff| + 1) / (n_perm + 1).
def permutation_test(a, b, n_perm=5000, seed=0):
    raise NotImplementedError


# 8. Paired bootstrap comparison of two models on the SAME test examples.
#    correct_a, correct_b: boolean arrays. Resample example indices (shared by both
#    models) n_boot times; for each resample compute accuracy_b - accuracy_a.
#    Return (observed_diff, low, high) for a 95% percentile interval.
def paired_bootstrap(correct_a, correct_b, n_boot=5000, seed=0):
    raise NotImplementedError


# 9. Pearson correlation from the definition: cov(x, y) / (std(x) * std(y)), using
#    population (ddof=0) statistics consistently. Don't use np.corrcoef.
def pearson(x, y):
    raise NotImplementedError
