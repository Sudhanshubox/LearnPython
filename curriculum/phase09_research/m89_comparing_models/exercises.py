"""m89 exercises: statistical tests for comparing models.

Implement the test statistics yourself; use scipy.stats only for distributions
(stats.t.sf), the exact binomial test (stats.binomtest) and stats.ttest_ind in exercise 7.
"""

import math

import numpy as np
from scipy import stats


# 1. Paired t-test of b against a (README section 2). Return (t, two-sided p) as floats, with
#    t positive when b is larger on average. Use the sample standard deviation (ddof=1).
def paired_t_test(a, b):
    raise NotImplementedError


# 2. Welch's t-test of b against a. Return (t, df, two-sided p) as floats.
def welch_t_test(a, b):
    raise NotImplementedError


# 3. Cohen's d for paired data: mean(b - a) / sample std(b - a).
def cohens_d_paired(a, b):
    raise NotImplementedError


# 4. McNemar's exact test. Return (only_a, only_b, p) where only_a counts the examples that A
#    predicts correctly and B doesn't, only_b the reverse, and p is the two-sided exact
#    binomial test of only_a out of only_a + only_b with probability 0.5
#    (stats.binomtest(...).pvalue). If there are no disagreements, p = 1.0.
def mcnemar(y_true, pred_a, pred_b):
    raise NotImplementedError


# 5. Paired permutation (sign-flip) test: d = b - a; draw signs with
#    np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, len(d))); count the
#    permutations whose |mean(signs * d)| >= |mean(d)|; return (count + 1) / (n_perm + 1).
#    No Python loops.
def permutation_test_paired(a, b, n_perm=10000, seed=0):
    raise NotImplementedError


# 6. Holm-Bonferroni (README section 4): return a list of booleans (rejected or not), in the
#    original order of pvalues.
def holm(pvalues, alpha=0.05):
    raise NotImplementedError


# 7. Simulate the multiple-comparisons trap. With rng = np.random.default_rng(seed), draw
#    a = rng.normal(size=(n_experiments, n_tests, n)) and then b the same way: in every
#    experiment, n_tests comparisons where NOTHING is truly different. Get the p-values with
#    stats.ttest_ind(a, b, axis=2).pvalue. Return the fraction of experiments in which at
#    least one test is rejected (p < alpha, or by holm when correct=True).
def family_false_positive_rate(n_tests, n_experiments=1000, n=20, alpha=0.05, correct=False, seed=0):
    raise NotImplementedError
