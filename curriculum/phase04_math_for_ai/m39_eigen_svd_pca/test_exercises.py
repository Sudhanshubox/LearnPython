import numpy as np
import pytest

import exercises
from code_checks import uses_any
from exercises import (
    covariance,
    is_eigenpair,
    lora_params,
    low_rank_approx,
    lowrank_params,
    pca_fit,
    pca_inverse,
    pca_transform,
    power_iteration,
    rank_for_energy,
    stationary_distribution,
    symmetric_matrix_power,
)

rng = np.random.default_rng(39)


def random_symmetric(n):
    M = rng.normal(size=(n, n))
    return M + M.T


def test_is_eigenpair():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert is_eigenpair(A, np.array([1.0, 1.0]), 3)
    assert is_eigenpair(A, np.array([1.0, -1.0]), 1)
    assert not is_eigenpair(A, np.array([1.0, 0.0]), 2)
    assert not is_eigenpair(A, np.zeros(2), 0)


def test_power_iteration():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    lam, v = power_iteration(A)
    assert lam == pytest.approx(3)
    assert np.linalg.norm(v) == pytest.approx(1)
    assert abs(v @ np.array([1, 1]) / np.sqrt(2)) == pytest.approx(1)
    assert not uses_any(exercises.power_iteration, "eig", "eigh", "eigvals", "eigvalsh")


def test_power_iteration_random_psd():
    M = rng.normal(size=(6, 6))
    A = M @ M.T                        # positive semi-definite: dominant eigenvalue is the largest
    lam, v = power_iteration(A, iters=2000)
    assert lam == pytest.approx(np.linalg.eigvalsh(A)[-1], rel=1e-6)
    assert is_eigenpair(A, v, lam, tol=1e-5)


def test_symmetric_matrix_power():
    A = random_symmetric(4)
    np.testing.assert_allclose(symmetric_matrix_power(A, 3), A @ A @ A, atol=1e-8)
    np.testing.assert_allclose(symmetric_matrix_power(A, 0), np.eye(4), atol=1e-10)


def test_covariance():
    X = rng.normal(size=(50, 4))
    np.testing.assert_allclose(covariance(X), np.cov(X, rowvar=False))
    assert not uses_any(exercises.covariance, "cov")


@pytest.fixture
def data():
    # 3-D data that mostly varies along one direction, a bit along a second, barely a third
    n = 500
    latent = rng.normal(size=(n, 3)) * np.array([5.0, 1.0, 0.1])
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return latent @ R.T + np.array([10.0, -3.0, 2.0]), R


def test_pca_fit(data):
    X, R = data
    mean, comps, ratio = pca_fit(X, 2)
    np.testing.assert_allclose(mean, X.mean(axis=0))
    assert comps.shape == (2, 3) and ratio.shape == (2,)
    np.testing.assert_allclose(comps @ comps.T, np.eye(2), atol=1e-10)
    assert abs(comps[0] @ R[:, 0]) > 0.99      # first component ≈ the main direction (up to sign)
    assert ratio[0] > ratio[1] > 0
    assert ratio.sum() > 0.99


def test_pca_matches_svd(data):
    X, _ = data
    _, comps, ratio = pca_fit(X, 3)
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    np.testing.assert_allclose(np.abs(np.sum(comps * Vt, axis=1)), 1, atol=1e-8)
    np.testing.assert_allclose(ratio, s ** 2 / (s ** 2).sum(), rtol=1e-8)


def test_pca_transform_and_inverse(data):
    X, _ = data
    mean, comps, _ = pca_fit(X, 2)
    Z = pca_transform(X, mean, comps)
    assert Z.shape == (len(X), 2)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)
    recon = pca_inverse(Z, mean, comps)
    rel_err = np.linalg.norm(X - recon) / np.linalg.norm(X - mean)
    assert rel_err < 0.05
    mean3, comps3, _ = pca_fit(X, 3)
    np.testing.assert_allclose(pca_inverse(pca_transform(X, mean3, comps3), mean3, comps3), X, atol=1e-8)


def test_low_rank_approx():
    A = rng.normal(size=(30, 20))
    s = np.linalg.svd(A, compute_uv=False)
    for k in (1, 5, 20):
        Ak = low_rank_approx(A, k)
        assert np.linalg.matrix_rank(Ak) == k
        assert np.linalg.norm(A - Ak) == pytest.approx(np.sqrt((s[k:] ** 2).sum()), abs=1e-8)


def test_rank_for_energy():
    U, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    V, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    s = np.array([10.0, 5.0, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0])
    A = U[:, :8] @ np.diag(s) @ V.T
    assert rank_for_energy(A, 0.79) == 1        # 100 / 126.26 ≈ 0.792
    assert rank_for_energy(A, 0.9) == 2
    assert rank_for_energy(A, 0.999) == 4


def test_param_counts():
    assert lowrank_params(1000, 1000, 50) == 50 * (1000 + 1000 + 1)
    assert lora_params(4096, 4096, 8) == 2 * 4096 * 8
    assert 4096 * 4096 // lora_params(4096, 4096, 8) == 256


def test_stationary_distribution():
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    pi = stationary_distribution(P)
    np.testing.assert_allclose(pi, [5 / 6, 1 / 6])
    M = rng.random((5, 5))
    P = M / M.sum(axis=1, keepdims=True)
    pi = stationary_distribution(P)
    assert pi.sum() == pytest.approx(1)
    assert (pi >= 0).all()
    np.testing.assert_allclose(pi @ P, pi, atol=1e-10)
    assert np.isrealobj(pi)
