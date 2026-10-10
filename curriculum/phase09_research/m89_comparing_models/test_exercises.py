import numpy as np
import pytest
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from code_checks import LOOP_NODES, contains
from exercises import (
    cohens_d_paired,
    family_false_positive_rate,
    holm,
    mcnemar,
    paired_t_test,
    permutation_test_paired,
    welch_t_test,
)

BASE = [0.912, 0.905, 0.921, 0.899, 0.915, 0.908, 0.918, 0.902]
NEW = [0.918, 0.909, 0.930, 0.901, 0.922, 0.913, 0.925, 0.906]


def test_paired_t_test():
    t, p = paired_t_test(BASE, NEW)
    ref = stats.ttest_rel(NEW, BASE)
    assert t == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    assert t > 0 and p < 0.001
    t_un, p_un = stats.ttest_ind(NEW, BASE)
    assert p_un > 0.1, "the same data, unpaired, shows nothing: pairing matters"


def test_welch_t_test():
    a, b = [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    t, df, p = welch_t_test(a, b)
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert t == pytest.approx(ref.statistic) and p == pytest.approx(ref.pvalue)
    assert df == pytest.approx(6.972, abs=0.001)


def test_cohens_d():
    d = np.array(NEW) - np.array(BASE)
    assert cohens_d_paired(BASE, NEW) == pytest.approx(d.mean() / d.std(ddof=1))


def test_mcnemar_small():
    y = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    a = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    b = [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    only_a, only_b, p = mcnemar(y, a, b)
    assert (only_a, only_b) == (4, 2)
    assert p == pytest.approx(stats.binomtest(4, 6, 0.5).pvalue)
    assert mcnemar([1, 0], [1, 0], [1, 0]) == (0, 0, 1.0)


@pytest.mark.timeout(60)
def test_mcnemar_real_classifiers():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    logreg = LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict(X_te)
    tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr).predict(X_te)
    only_logreg, only_tree, p = mcnemar(y_te, logreg, tree)
    assert only_logreg > 3 * only_tree and p < 1e-6


def test_permutation_test():
    p = permutation_test_paired(BASE, NEW, n_perm=5000)
    assert p == pytest.approx(2 / 256, abs=0.003), "all 8 differences are positive: only all-plus or all-minus signs are as extreme"
    noise = np.random.default_rng(1).normal(size=10)
    assert permutation_test_paired(noise, noise[::-1], n_perm=2000) > 0.2
    assert permutation_test_paired(BASE, NEW, seed=3) == permutation_test_paired(BASE, NEW, seed=3)
    assert not contains(permutation_test_paired, LOOP_NODES)


def test_holm():
    assert holm([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]
    assert holm([0.001, 0.012, 0.03]) == [True, True, True]
    assert holm([0.2, 0.01]) == [False, True]
    assert holm([0.06, 0.07]) == [False, False]


def test_multiple_comparisons_trap():
    one = family_false_positive_rate(1)
    many = family_false_positive_rate(20)
    corrected = family_false_positive_rate(20, correct=True)
    assert one == pytest.approx(0.05, abs=0.02)
    assert many == pytest.approx(1 - 0.95 ** 20, abs=0.05), "20 tests on pure noise: ~64% find 'something'"
    assert corrected <= 0.07
