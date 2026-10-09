import itertools
import time

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkTree

from exercises import DecisionTreeClassifier, RandomForestClassifier, best_split, entropy, gini


def test_impurities():
    assert gini(np.array([0, 0, 0])) == 0
    assert gini(np.array([0, 1])) == pytest.approx(0.5)
    assert gini(np.array([0, 1, 2])) == pytest.approx(2 / 3)
    assert entropy(np.array([0, 1])) == pytest.approx(1)
    assert entropy(np.array([1, 1, 1, 1])) == 0
    assert entropy(np.array([0, 1, 2, 3])) == pytest.approx(2)
    assert gini(np.array([], dtype=int)) == 0 and entropy(np.array([], dtype=int)) == 0


def brute_best_split(X, y):
    best = (None, None, 0.0)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for t in (vals[:-1] + vals[1:]) / 2:
            left, right = y[X[:, j] <= t], y[X[:, j] > t]
            g = gini(y) - (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if g > best[2] + 1e-12:
                best = (j, t, g)
    return best


def test_best_split_simple():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 1.0], [4.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    j, t, g = best_split(X, y)
    assert (j, t) == (0, 2.5)
    assert g == pytest.approx(0.5)
    assert best_split(X, np.array([1, 1, 1, 1]))[0] is None
    assert best_split(np.ones((3, 2)), np.array([0, 1, 0]))[0] is None


@pytest.mark.parametrize("seed", range(5))
def test_best_split_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(40, 3)), 1)
    y = rng.integers(0, 3, 40)
    j, t, g = best_split(X, y)
    bj, bt, bg = brute_best_split(X, y)
    assert g == pytest.approx(bg)
    assert (j, t) == (bj, pytest.approx(bt))


def test_best_split_feature_subset():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 1.0], [4.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    j, t, _ = best_split(X, y, features=[1])
    assert (j, t) == (1, 3.0)


def test_best_split_is_fast():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(3000, 10)), rng.integers(0, 2, 3000)
    start = time.perf_counter()
    best_split(X, y)
    assert time.perf_counter() - start < 1.0, "use sorting + cumulative counts instead of trying each threshold separately"


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


def test_tree_xor():
    # Slightly unbalanced XOR. (With perfectly balanced XOR, no single split has positive
    # gain at the root, so a greedy tree can't even start: see the lesson's "go deeper".)
    X = np.array([[0.0, 0.0]] * 5 + [[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 5 + [[1.0, 1.0]] * 3)
    y = np.array([0] * 5 + [1] * 5 + [1] * 5 + [0] * 3)
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.score(X, y) == 1.0, "trees can learn XOR, which no linear model can"
    assert tree.depth() == 2 and tree.n_leaves() == 4


@pytest.mark.timeout(60)
def test_tree_matches_sklearn(cancer):
    Xtr, Xte, ytr, yte = cancer
    mine = DecisionTreeClassifier(max_depth=4).fit(Xtr, ytr)
    ref = SkTree(max_depth=4, random_state=0).fit(Xtr, ytr)
    assert mine.depth() <= 4
    assert mine.score(Xte, yte) > 0.88
    assert np.mean(mine.predict(Xte) == ref.predict(Xte)) > 0.95
    assert mine.feature_importances_.sum() == pytest.approx(1)
    assert set(np.argsort(-mine.feature_importances_)[:3]) == set(np.argsort(-ref.feature_importances_)[:3])


@pytest.mark.timeout(60)
def test_full_tree_memorizes(cancer):
    Xtr, Xte, ytr, yte = cancer
    full = DecisionTreeClassifier().fit(Xtr, ytr)
    assert full.score(Xtr, ytr) == 1.0
    assert DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr).n_leaves() == 2


def test_single_leaf_tree():
    t = DecisionTreeClassifier().fit(np.zeros((4, 2)), np.array([2, 2, 2, 2]))
    assert t.depth() == 0 and t.n_leaves() == 1
    np.testing.assert_array_equal(t.predict(np.ones((2, 2))), [2, 2])
    np.testing.assert_array_equal(t.feature_importances_, [0, 0])


@pytest.mark.timeout(120)
def test_forest_beats_single_tree():
    gains = []
    for s in range(5):
        X, y = make_classification(400, n_features=20, n_informative=5, flip_y=0.1, random_state=s)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=s)
        tree = DecisionTreeClassifier().fit(Xtr, ytr).score(Xte, yte)
        forest = RandomForestClassifier(n_estimators=30, seed=s).fit(Xtr, ytr)
        assert len(forest.trees_) == 30
        gains.append(forest.score(Xte, yte) - tree)
    assert np.mean(gains) > 0.02, f"average gain over a single tree was only {np.mean(gains):.3f}"


def test_forest_is_reproducible():
    X, y = make_classification(100, n_features=6, random_state=0)
    a = RandomForestClassifier(n_estimators=5, seed=3).fit(X, y).predict(X)
    b = RandomForestClassifier(n_estimators=5, seed=3).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)
