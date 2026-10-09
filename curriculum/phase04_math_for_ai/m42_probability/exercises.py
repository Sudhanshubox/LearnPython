"""m42 exercises: probability.

Create random numbers ONLY with np.random.default_rng(seed) (or the rng passed in),
so results are reproducible.
"""

import math

import numpy as np


# 1. Bayes' theorem for a diagnostic test. Return P(condition | positive test).
#    prevalence = P(condition), sensitivity = P(positive | condition),
#    false_positive_rate = P(positive | no condition).
def posterior_positive(prevalence, sensitivity, false_positive_rate):
    raise NotImplementedError


# 2. Expected value and variance of a discrete distribution given its values and probs.
#    Return (mean, variance).
def mean_and_variance(values, probs):
    raise NotImplementedError


# 3. Monte Carlo: the probability that the sum of `n_dice` fair six-sided dice equals
#    `target`, estimated from `trials` simulated rolls (vectorized, no loops).
def prob_dice_sum(n_dice, target, trials=200_000, seed=0):
    raise NotImplementedError


# 4. Estimate pi by sampling `n` points uniformly in the unit square and counting the
#    fraction inside the quarter circle x^2 + y^2 <= 1 (that fraction is about pi/4).
def estimate_pi(n, seed=0):
    raise NotImplementedError


# 5. The normal density p(x) for an array x, from the formula (not scipy).
def normal_pdf(x, mu=0.0, sigma=1.0):
    raise NotImplementedError


# 6. Law of large numbers: roll a fair die n times and return the array of RUNNING means
#    (element i is the mean of the first i + 1 rolls). Use rng.integers(1, 7, n).
def running_mean_of_die(n, seed=0):
    raise NotImplementedError


# 7. Central limit theorem: draw n_means samples, each the mean of `sample_size` draws
#    from an exponential distribution with scale 1 (rng.exponential(1.0, size=...)).
#    Return the array of n_means sample means. (Vectorized: draw a 2-D array.)
def sample_means(sample_size, n_means, seed=0):
    raise NotImplementedError


# 8. Inverse-CDF sampling from a categorical distribution: return n indices drawn
#    according to probs, using rng.random(n), np.cumsum and np.searchsorted
#    (don't use rng.choice).
def sample_categorical(probs, n, rng):
    raise NotImplementedError


# 9. Temperature: turn logits into probabilities with softmax(logits / T).
#    T == 0 means greedy: probability 1 for the (first) largest logit, 0 elsewhere.
def temperature_probs(logits, T):
    raise NotImplementedError


# 10. The birthday problem: the probability that among k people at least two share a
#     birthday (365 equally likely days).
#     birthday_exact(k) uses the formula 1 - prod((365 - i) / 365 for i in range(k)).
#     birthday_simulated(k, trials, seed) estimates it by simulation (vectorized:
#     draw a (trials, k) array of birthdays; sort each row and look for equal neighbours).
def birthday_exact(k):
    raise NotImplementedError


def birthday_simulated(k, trials=20_000, seed=0):
    raise NotImplementedError
