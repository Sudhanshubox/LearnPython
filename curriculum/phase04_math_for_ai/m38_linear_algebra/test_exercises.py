import ast

import numpy as np
import pytest

import exercises
from code_checks import contains, has_loops, uses_any
from exercises import (
    analogy,
    angle_degrees,
    cosine_similarity,
    cosine_similarity_matrix,
    dot,
    fit_linear,
    gram_schmidt,
    norm,
    parallelogram_area,
    project,
    reject,
    rotate_points,
    rotation_matrix,
)

rng = np.random.default_rng(38)


def test_dot():
    assert dot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])) == 32
    u, v = rng.normal(size=10), rng.normal(size=10)
    assert dot(u, v) == pytest.approx(u @ v)
    assert not uses_any(exercises.dot, "dot", "inner", "matmul", "vdot")
    assert not contains(exercises.dot, ast.MatMult), "don't use the @ operator here"


@pytest.mark.parametrize("p, expected", [(1, 7.0), (2, 5.0), (np.inf, 4.0)])
def test_norm(p, expected):
    assert norm(np.array([3.0, -4.0]), p) == pytest.approx(expected)


def test_norm_from_definition():
    assert not uses_any(exercises.norm, "linalg")


def test_cosine_and_angle():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(0)
    assert cosine_similarity(np.array([1.0, 1.0]), np.array([3.0, 3.0])) == pytest.approx(1)
    assert angle_degrees(np.array([1.0, 0.0]), np.array([1.0, 1.0])) == pytest.approx(45)
    # arccos is very sensitive near 1, so tiny rounding errors become ~1e-6 degrees
    assert angle_degrees(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(0, abs=1e-3)
    assert angle_degrees(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(180)


def test_cosine_similarity_matrix():
    E = rng.normal(size=(6, 4))
    S = cosine_similarity_matrix(E)
    assert S.shape == (6, 6)
    np.testing.assert_allclose(np.diag(S), 1)
    np.testing.assert_allclose(S, S.T)
    assert S[1, 3] == pytest.approx(cosine_similarity(E[1], E[3]))
    assert not has_loops(exercises.cosine_similarity_matrix)


def test_project_and_reject():
    u, v = np.array([2.0, 3.0]), np.array([4.0, 0.0])
    np.testing.assert_allclose(project(u, v), [2, 0])
    np.testing.assert_allclose(reject(u, v), [0, 3])
    a, b = rng.normal(size=5), rng.normal(size=5)
    np.testing.assert_allclose(project(a, b) + reject(a, b), a)
    assert reject(a, b) @ b == pytest.approx(0, abs=1e-10)


def test_rotation():
    R = rotation_matrix(90)
    np.testing.assert_allclose(R @ np.array([1.0, 0.0]), [0, 1], atol=1e-12)
    np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1)
    P = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    out = rotate_points(P, 180)
    np.testing.assert_allclose(out, -P, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(rotate_points(P, 37), axis=1), np.linalg.norm(P, axis=1))


def test_gram_schmidt():
    V = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    Q = gram_schmidt(V)
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(Q[0], V[0] / np.linalg.norm(V[0]))
    # same span: each original vector is reproduced by projecting onto the new basis
    np.testing.assert_allclose((V @ Q.T) @ Q, V, atol=1e-10)
    assert not uses_any(exercises.gram_schmidt, "qr")


def test_gram_schmidt_tall():
    V = rng.normal(size=(3, 10))
    Q = gram_schmidt(V)
    assert Q.shape == (3, 10)
    np.testing.assert_allclose(Q @ Q.T, np.eye(3), atol=1e-10)


def test_parallelogram_area():
    assert parallelogram_area(np.array([2.0, 0.0]), np.array([0.0, 3.0])) == pytest.approx(6)
    assert parallelogram_area(np.array([0.0, 3.0]), np.array([2.0, 0.0])) == pytest.approx(6)
    assert parallelogram_area(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(0, abs=1e-12)


def test_fit_linear():
    X = rng.normal(size=(200, 3))
    true_w, true_b = np.array([2.0, -1.0, 0.5]), 4.0
    y = X @ true_w + true_b + rng.normal(0, 0.01, 200)
    w, b = fit_linear(X, y)
    assert w.shape == (3,)
    np.testing.assert_allclose(w, true_w, atol=0.01)
    assert b == pytest.approx(true_b, abs=0.01)


def test_analogy():
    emb = {
        "man":   np.array([1.0, 0.0, 0.2]),
        "woman": np.array([1.0, 1.0, 0.2]),
        "king":  np.array([1.0, 0.0, 1.0]),
        "queen": np.array([1.0, 1.0, 1.0]),
        "apple": np.array([-1.0, 0.3, 0.0]),
        "prince": np.array([0.8, 0.0, 0.9]),
    }
    assert analogy(emb, "man", "king", "woman") == "queen"
    assert analogy(emb, "king", "man", "queen") == "woman"
