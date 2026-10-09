import numpy as np
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    baseline_predict,
    factorize,
    make_ratings,
    mf_loss_and_grads,
    recommend,
    rmse,
    similar_items,
    split_mask,
    svd_predict,
)


@pytest.fixture(scope="module")
def data():
    R, mask = make_ratings()
    train, test = split_mask(mask)
    return R, mask, train, test


@pytest.fixture(scope="module")
def trained(data):
    R, _, train, _ = data
    return factorize(R, train, k=3)


@pytest.mark.parametrize("name", ["split_mask", "rmse", "baseline_predict", "svd_predict", "mf_loss_and_grads", "recommend", "similar_items"])
def test_vectorized(name):
    assert not has_loops(getattr(exercises, name))


def test_split_mask(data):
    _, mask, train, test = data
    assert not (train & test).any()
    np.testing.assert_array_equal(train | test, mask)
    assert 0.15 < test.sum() / mask.sum() < 0.25
    t2, s2 = split_mask(mask)
    np.testing.assert_array_equal(train, t2)


def test_rmse():
    R = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.array([[1.0, 0.0], [3.0, 1.0]])
    mask = np.array([[True, True], [False, True]])
    assert rmse(R, pred, mask) == pytest.approx(np.sqrt((0 + 4 + 9) / 3))


def test_baseline_predict():
    R = np.array([[5.0, 3.0, 0.0], [4.0, 0.0, 1.0]])
    train = np.array([[True, True, False], [True, False, True]])
    pred = baseline_predict(R, train)
    mu = 13 / 4
    b_u = np.array([4 - mu, 2.5 - mu])
    b_i = np.array([((5 - mu - b_u[0]) + (4 - mu - b_u[1])) / 2, 3 - mu - b_u[0], 1 - mu - b_u[1]])
    np.testing.assert_allclose(pred, mu + b_u[:, None] + b_i[None, :])
    assert pred.shape == R.shape


def test_baseline_handles_users_without_ratings():
    R = np.ones((3, 2))
    train = np.array([[True, True], [False, False], [True, False]])
    assert np.isfinite(baseline_predict(R, train)).all()


def test_svd_predict(data):
    R, _, train, test = data
    pred = svd_predict(R, train, 5)
    assert pred.shape == R.shape
    assert np.linalg.matrix_rank(pred) == 5
    assert rmse(R, pred, test) < rmse(R, baseline_predict(R, train), test)


def test_mf_gradients():
    rng = np.random.default_rng(0)
    T = rng.normal(size=(6, 5))
    mask = rng.random((6, 5)) < 0.6
    U, V = rng.normal(size=(6, 2)), rng.normal(size=(5, 2))
    loss, dU, dV = mf_loss_and_grads(T, mask, U, V, reg=0.01)
    E = np.where(mask, U @ V.T - T, 0)
    assert loss == pytest.approx((E ** 2).sum() / mask.sum() + 0.01 * ((U ** 2).sum() + (V ** 2).sum()))
    h = 1e-6
    for M, dM, which in ((U, dU, "U"), (V, dV, "V")):
        num = np.zeros_like(M)
        for idx in np.ndindex(M.shape):
            P, Q = M.copy(), M.copy()
            P[idx] += h
            Q[idx] -= h
            args_p = (T, mask, P, V, 0.01) if which == "U" else (T, mask, U, P, 0.01)
            args_q = (T, mask, Q, V, 0.01) if which == "U" else (T, mask, U, Q, 0.01)
            num[idx] = (mf_loss_and_grads(*args_p)[0] - mf_loss_and_grads(*args_q)[0]) / (2 * h)
        np.testing.assert_allclose(dM, num, atol=1e-6)


def test_factorize_trains(data, trained):
    R, _, train, _ = data
    U, V, base, losses = trained
    assert U.shape == (R.shape[0], 3) and V.shape == (R.shape[1], 3)
    np.testing.assert_allclose(base, baseline_predict(R, train))
    assert len(losses) == 500
    assert losses[-1] < losses[0] / 5


def test_factorization_beats_baselines(data, trained):
    R, _, train, test = data
    U, V, base, _ = trained
    mf = rmse(R, base + U @ V.T, test)
    assert mf < 0.6
    assert mf < 0.5 * rmse(R, baseline_predict(R, train), test)
    assert mf < 0.5 * rmse(R, svd_predict(R, train, 5), test)


def test_recommend(data, trained):
    R, _, train, _ = data
    U, V, base, _ = trained
    pred = base + U @ V.T
    recs = recommend(pred, train, user=0, n=5)
    assert len(recs) == 5
    assert not train[0, recs].any(), "don't recommend items the user already rated"
    unrated = np.flatnonzero(~train[0])
    best = unrated[np.argsort(-pred[0, unrated], kind="stable")[:5]]
    assert list(recs) == list(best)


def test_similar_items():
    V = np.array([[1.0, 0.0], [2.0, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.7, 0.7]])
    assert similar_items(V, 0, n=2) == [1, 4]
    assert 0 not in similar_items(V, 0, n=4)
    assert similar_items(V, 0, n=4)[-1] == 3
