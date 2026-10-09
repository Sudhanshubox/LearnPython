import inspect
import time

import numpy as np
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    confusion_matrix,
    masked_mean,
    masked_softmax,
    mlp_forward,
    moving_average,
    nearest_neighbor_predict,
    pairwise_sq_distances,
    polynomial_features,
    softmax,
    standardize_columns,
)

rng = np.random.default_rng(37)
FUNCS = [name for name, f in inspect.getmembers(exercises, inspect.isfunction) if f.__module__ == "exercises"]


@pytest.mark.parametrize("name", FUNCS)
def test_no_loops(name):
    assert not has_loops(getattr(exercises, name)), f"{name}: vectorize it, no loops or comprehensions"


def test_standardize_columns():
    X = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 9.0]])
    Z = standardize_columns(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(Z[:, [0, 2]].std(axis=0), 1)
    np.testing.assert_array_equal(Z[:, 1], 0)


def loop_distances(A, B):
    return np.array([[((a - b) ** 2).sum() for b in B] for a in A])


def test_pairwise_sq_distances():
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    D = pairwise_sq_distances(A, B)
    assert D.shape == (5, 4)
    np.testing.assert_allclose(D, loop_distances(A, B), atol=1e-9)
    assert (pairwise_sq_distances(A, A) >= 0).all()
    np.testing.assert_allclose(np.diag(pairwise_sq_distances(A, A)), 0, atol=1e-9)


def test_pairwise_sq_distances_fast():
    A, B = rng.normal(size=(2000, 50)), rng.normal(size=(2000, 50))
    start = time.perf_counter()
    pairwise_sq_distances(A, B)
    assert time.perf_counter() - start < 1.5


def test_nearest_neighbor_predict():
    X_train = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    y_train = np.array([0, 1, 2])
    X_test = np.array([[1.0, 1.0], [9.0, 8.0], [1.0, 9.0], [0.2, 0.0]])
    np.testing.assert_array_equal(nearest_neighbor_predict(X_train, y_train, X_test), [0, 1, 2, 0])


def test_nearest_neighbor_on_clusters():
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    y = rng.integers(0, 3, 600)
    X = centers[y] + rng.normal(0, 0.8, (600, 2))
    pred = nearest_neighbor_predict(X[:500], y[:500], X[500:])
    assert (pred == y[500:]).mean() > 0.95


def test_softmax():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(p, np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum())
    big = softmax(np.array([[1000.0, 1001.0], [-1000.0, -1000.0]]))
    assert not np.isnan(big).any()
    np.testing.assert_allclose(big.sum(axis=1), 1)
    np.testing.assert_allclose(big[1], [0.5, 0.5])
    cols = softmax(np.array([[1.0, 5.0], [3.0, 5.0]]), axis=0)
    np.testing.assert_allclose(cols.sum(axis=0), 1)
    assert softmax(rng.normal(size=(2, 3, 4))).shape == (2, 3, 4)


def test_polynomial_features():
    F = polynomial_features(np.array([2.0, 3.0]), 3)
    np.testing.assert_array_equal(F, [[1, 2, 4, 8], [1, 3, 9, 27]])


@pytest.mark.parametrize("x, w, expected", [
    ([1, 2, 3, 4, 5], 3, [2, 3, 4]), ([5, 5], 1, [5, 5]), ([1, 2, 3, 4], 4, [2.5]),
])
def test_moving_average(x, w, expected):
    np.testing.assert_allclose(moving_average(np.array(x, dtype=float), w), expected)


def test_mlp_forward():
    X = np.array([[1.0, -1.0]])
    W1 = np.array([[1.0, 2.0, -1.0], [0.5, 1.0, 1.0]])
    b1 = np.array([0.0, -2.0, 0.0])
    W2 = np.array([[1.0], [1.0], [1.0]])
    b2 = np.array([0.5])
    # hidden pre-activation: [0.5, -1.0, -2.0] -> relu -> [0.5, 0, 0]; output 0.5 + 0.5
    np.testing.assert_allclose(mlp_forward(X, W1, b1, W2, b2), [[1.0]])
    Xb = rng.normal(size=(64, 10))
    out = mlp_forward(Xb, rng.normal(size=(10, 32)), np.zeros(32), rng.normal(size=(32, 4)), np.zeros(4))
    assert out.shape == (64, 4)


def test_confusion_matrix():
    y_true = np.array([0, 0, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [1, 0, 2]])
    assert cm.sum() == 6 and np.issubdtype(cm.dtype, np.integer)


def test_masked_mean():
    values = np.array([[1.0, 2.0, 99.0], [4.0, 99.0, 99.0], [7.0, 8.0, 9.0], [5.0, 5.0, 5.0]])
    mask = np.array([[True, True, False], [True, False, False], [True, True, True], [False, False, False]])
    np.testing.assert_allclose(masked_mean(values, mask), [1.5, 4.0, 8.0, 0.0])


def test_masked_softmax():
    scores = np.array([[1.0, 2.0, 50.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
    mask = np.array([[True, True, False], [True, True, True], [False, False, False]])
    p = masked_softmax(scores, mask)
    assert not np.isnan(p).any()
    np.testing.assert_allclose(p[0], [np.exp(1) / (np.exp(1) + np.exp(2)), np.exp(2) / (np.exp(1) + np.exp(2)), 0])
    np.testing.assert_allclose(p[1], [1 / 3] * 3)
    np.testing.assert_array_equal(p[2], [0, 0, 0])
