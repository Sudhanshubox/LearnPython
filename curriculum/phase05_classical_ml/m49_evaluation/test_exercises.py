import ast

import numpy as np
import pytest
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import exercises
from code_checks import uses_any
from exercises import (
    best_f1_threshold,
    binary_metrics,
    count_leaked_rows,
    cross_validate,
    kfold_indices,
    majority_baseline,
    mean_baseline_r2,
    regression_metrics,
    roc_auc,
    stratified_split,
    train_val_test_split,
)

rng = np.random.default_rng(49)


def test_no_sklearn_in_exercises():
    tree = ast.parse(open(exercises.__file__, encoding="utf-8").read())
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imported += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m.startswith("sklearn") for m in imported), "implement these with NumPy, not sklearn"


def test_train_val_test_split():
    tr, va, te = train_val_test_split(100, 0.15, 0.2, seed=1)
    assert (len(tr), len(va), len(te)) == (65, 15, 20)
    assert sorted(np.concatenate([tr, va, te])) == list(range(100))
    perm = np.random.default_rng(1).permutation(100)
    np.testing.assert_array_equal(te, perm[:20])


def test_stratified_split():
    y = np.array([0] * 90 + [1] * 10)
    tr, te = stratified_split(y, 0.2, seed=3)
    assert len(te) == 20 and len(tr) == 80
    assert (y[te] == 1).sum() == 2, "keep the class proportions"
    assert not set(tr) & set(te)
    assert list(tr) == sorted(tr) and list(te) == sorted(te)


def test_kfold_indices():
    folds = kfold_indices(23, 5, seed=0)
    assert len(folds) == 5
    vals = np.concatenate([v for _, v in folds])
    assert sorted(vals) == list(range(23))
    for tr, va in folds:
        assert not set(tr) & set(va) and len(tr) + len(va) == 23
    assert sorted(len(v) for _, v in folds) == [4, 4, 5, 5, 5]


def test_binary_metrics():
    y_true = rng.integers(0, 2, 200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, 1 - y_true)
    m = binary_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(metrics.accuracy_score(y_true, y_pred))
    assert m["precision"] == pytest.approx(metrics.precision_score(y_true, y_pred))
    assert m["recall"] == pytest.approx(metrics.recall_score(y_true, y_pred))
    assert m["f1"] == pytest.approx(metrics.f1_score(y_true, y_pred))


def test_binary_metrics_zero_division():
    m = binary_metrics(np.array([0, 0, 1]), np.array([0, 0, 0]))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["accuracy"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("trial", range(5))
def test_roc_auc(trial):
    r = np.random.default_rng(trial)
    y = r.integers(0, 2, 300)
    scores = y * 0.7 + r.normal(0, 0.6, 300)
    if trial % 2:
        scores = np.round(scores, 1)        # lots of ties
    assert roc_auc(y, scores) == pytest.approx(metrics.roc_auc_score(y, scores))


def test_roc_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_best_f1_threshold():
    y = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.2])
    t, f1 = best_f1_threshold(y, scores)
    assert t == 0.35
    assert f1 == pytest.approx(metrics.f1_score(y, scores >= 0.35))
    for cand in np.unique(scores):
        assert metrics.f1_score(y, scores >= cand) <= f1 + 1e-12


def test_regression_metrics():
    y = rng.normal(10, 3, 100)
    pred = y + rng.normal(0, 1, 100)
    m = regression_metrics(y, pred)
    assert m["mae"] == pytest.approx(metrics.mean_absolute_error(y, pred))
    assert m["rmse"] == pytest.approx(np.sqrt(metrics.mean_squared_error(y, pred)))
    assert m["r2"] == pytest.approx(metrics.r2_score(y, pred))


def test_baselines():
    assert majority_baseline(np.array([1, 1, 0, 2]), np.array([1, 0, 1, 1])) == 0.75
    assert majority_baseline(np.array([3, 1, 3, 1]), np.array([1, 3])) == 0.5     # tie -> smallest label (1)
    y_train, y_test = rng.normal(5, 1, 50), rng.normal(5, 1, 50)
    expected = metrics.r2_score(y_test, np.full(50, y_train.mean()))
    assert mean_baseline_r2(y_train, y_test) == pytest.approx(expected)
    assert mean_baseline_r2(y_train, y_test) <= 0.05


def test_cross_validate():
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    scores = cross_validate(lambda: LogisticRegression(), X, y, k=5, seed=0)
    assert len(scores) == 5
    assert scores.mean() > 0.9
    assert np.ptp(scores) < 0.2


def test_count_leaked_rows():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[3.0, 4.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0]])
    assert count_leaked_rows(X_train, X_test) == 3
    assert count_leaked_rows(X_train, np.array([[9.0, 9.0]])) == 0
