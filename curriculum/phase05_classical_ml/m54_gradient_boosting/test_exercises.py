import types

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_friedman1
from sklearn.ensemble import GradientBoostingRegressor as SkGB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from exercises import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RegressionTree,
    best_iteration,
    best_regression_split,
)


@pytest.fixture(scope="module")
def friedman():
    X, y = make_friedman1(1000, noise=1.0, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


def sse(v):
    return ((v - v.mean()) ** 2).sum() if len(v) else 0.0


def brute(X, y, min_leaf=1):
    best = (None, None, 0.0)
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for t in (vals[:-1] + vals[1:]) / 2:
            m = X[:, j] <= t
            if m.sum() < min_leaf or (~m).sum() < min_leaf:
                continue
            g = sse(y) - sse(y[m]) - sse(y[~m])
            if g > best[2] + 1e-9:
                best = (j, t, g)
    return best


@pytest.mark.parametrize("seed", range(4))
def test_best_regression_split_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    X = np.round(rng.normal(size=(50, 3)), 1)
    y = X[:, seed % 3] * 2 + rng.normal(0, 0.3, 50)
    j, t, g = best_regression_split(X, y, min_samples_leaf=3)
    bj, bt, bg = brute(X, y, 3)
    assert (j, g) == (bj, pytest.approx(bg))
    assert t == pytest.approx(bt)


def test_best_regression_split_edge_cases():
    assert best_regression_split(np.ones((5, 2)), np.arange(5.0))[0] is None
    assert best_regression_split(np.arange(4.0).reshape(-1, 1), np.ones(4))[0] is None


def test_regression_tree_matches_sklearn(friedman):
    Xtr, Xte, ytr, yte = friedman
    mine = RegressionTree(max_depth=3).fit(Xtr, ytr)
    ref = DecisionTreeRegressor(max_depth=3, random_state=0).fit(Xtr, ytr)
    np.testing.assert_allclose(mine.predict(Xte), ref.predict(Xte), rtol=1e-6)


def test_regression_tree_leaf_means():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([1.0, 3.0, 10.0, 12.0])
    t = RegressionTree(max_depth=1).fit(X, y)
    np.testing.assert_allclose(t.predict(np.array([[0.5], [10.5]])), [2.0, 11.0])


@pytest.mark.timeout(120)
def test_boosting_regressor(friedman):
    Xtr, Xte, ytr, yte = friedman
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
    assert gb.fit(Xtr, ytr) is gb
    assert gb.init_ == pytest.approx(ytr.mean())
    assert len(gb.trees_) == 200 and len(gb.train_losses_) == 200
    assert all(a >= b - 1e-9 for a, b in zip(gb.train_losses_, gb.train_losses_[1:]))
    r2 = gb.score(Xte, yte)
    ref = SkGB(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert r2 == pytest.approx(ref, abs=0.01)
    assert r2 > LinearRegression().fit(Xtr, ytr).score(Xte, yte) + 0.1


def test_staged_predict(friedman):
    Xtr, Xte, ytr, yte = friedman
    gb = GradientBoostingRegressor(n_estimators=20).fit(Xtr[:200], ytr[:200])
    stages = gb.staged_predict(Xte)
    assert isinstance(stages, types.GeneratorType)
    stages = list(stages)
    assert len(stages) == 20
    np.testing.assert_allclose(stages[-1], gb.predict(Xte))
    np.testing.assert_allclose(stages[0], gb.init_ + 0.1 * gb.trees_[0].predict(Xte))


@pytest.mark.timeout(120)
def test_boosting_classifier():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2).fit(Xtr, ytr)
    p = ytr.mean()
    assert clf.init_ == pytest.approx(np.log(p / (1 - p)))
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(Xte),) and ((proba > 0) & (proba < 1)).all()
    assert clf.score(Xte, yte) > 0.92


def test_best_iteration(friedman):
    Xtr, Xte, ytr, yte = friedman
    gb = GradientBoostingRegressor(n_estimators=60, learning_rate=0.5, max_depth=5).fit(Xtr[:150], ytr[:150])
    losses = [np.mean((yte - p) ** 2) for p in gb.staged_predict(Xte)]
    best = best_iteration(gb, Xte, yte)
    assert best == int(np.argmin(losses)) + 1
    assert best < 60, "with a large learning rate and deep trees, validation loss bottoms out early"
