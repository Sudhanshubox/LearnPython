import numpy as np
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    bce_loss_and_grads,
    chain_rule,
    gradient_check,
    layer_backward,
    mse_loss_and_grads,
    numerical_gradient,
    numerical_jacobian,
    quadratic,
    relative_error,
    relu_grad,
    sigmoid,
    sigmoid_grad,
    softmax_jacobian,
    tanh_grad,
)

rng = np.random.default_rng(40)


@pytest.mark.parametrize("name", ["sigmoid", "sigmoid_grad", "tanh_grad", "relu_grad", "mse_loss_and_grads", "bce_loss_and_grads", "softmax_jacobian"])
def test_vectorized(name):
    assert not has_loops(getattr(exercises, name))


def test_numerical_gradient():
    f = lambda v: (v ** 2).sum() + 3 * v[0]
    x = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(numerical_gradient(f, x), 2 * x + np.array([3, 0, 0]), atol=1e-6)
    np.testing.assert_array_equal(x, [1.0, -2.0, 0.5]), "don't modify x"
    M = rng.normal(size=(2, 3))
    g = numerical_gradient(lambda m: (m ** 3).sum(), M)
    assert g.shape == (2, 3)
    np.testing.assert_allclose(g, 3 * M ** 2, rtol=1e-6)


def test_chain_rule():
    d = chain_rule(np.cos, lambda x: x ** 2, lambda x: 2 * x)
    for x in [0.0, 0.7, -1.3]:
        assert d(x) == pytest.approx(np.cos(x ** 2) * 2 * x)
    d2 = chain_rule(np.exp, np.sin, np.cos)          # d/dx e^(sin x)
    assert d2(0.5) == pytest.approx(np.exp(np.sin(0.5)) * np.cos(0.5))


@pytest.mark.parametrize("func, grad", [
    (lambda x: 1 / (1 + np.exp(-x)), "sigmoid_grad"),
    (np.tanh, "tanh_grad"),
])
def test_activation_grads(func, grad):
    x = np.linspace(-3, 3, 13)
    numeric = (func(x + 1e-6) - func(x - 1e-6)) / 2e-6
    np.testing.assert_allclose(getattr(exercises, grad)(x), numeric, atol=1e-8)


def test_sigmoid_and_relu():
    np.testing.assert_allclose(sigmoid(np.array([0.0, 100.0, -100.0])), [0.5, 1.0, 0.0], atol=1e-12)
    np.testing.assert_array_equal(relu_grad(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 1.0])


def test_quadratic():
    M = rng.normal(size=(4, 4))
    A, b, x = M + M.T, rng.normal(size=4), rng.normal(size=4)
    value, grad = quadratic(A, b, x)
    assert value == pytest.approx(0.5 * x @ A @ x + b @ x)
    np.testing.assert_allclose(grad, numerical_gradient(lambda v: 0.5 * v @ A @ v + b @ v, x), atol=1e-6)


def test_mse_grads():
    X, y, w, b = rng.normal(size=(20, 3)), rng.normal(size=20), rng.normal(size=3), 0.3
    loss, dw, db = mse_loss_and_grads(w, b, X, y)
    assert loss == pytest.approx(np.mean((X @ w + b - y) ** 2))
    np.testing.assert_allclose(dw, numerical_gradient(lambda v: mse_loss_and_grads(v, b, X, y)[0], w), atol=1e-6)
    assert db == pytest.approx(numerical_gradient(lambda v: mse_loss_and_grads(w, v[0], X, y)[0], np.array([b]))[0], abs=1e-6)


def test_bce_grads():
    X, w, b = rng.normal(size=(30, 4)), rng.normal(size=4), -0.2
    y = (rng.random(30) > 0.5).astype(float)
    loss, dw, db = bce_loss_and_grads(w, b, X, y)
    p = 1 / (1 + np.exp(-(X @ w + b)))
    assert loss == pytest.approx(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    np.testing.assert_allclose(dw, numerical_gradient(lambda v: bce_loss_and_grads(v, b, X, y)[0], w), atol=1e-6)
    assert db == pytest.approx(numerical_gradient(lambda v: bce_loss_and_grads(w, v[0], X, y)[0], np.array([b]))[0], abs=1e-6)


def test_bce_is_finite_for_confident_predictions():
    X = np.array([[100.0], [-100.0]])
    loss, dw, db = bce_loss_and_grads(np.array([1.0]), 0.0, X, np.array([0.0, 1.0]))
    assert np.isfinite(loss) and np.isfinite(dw).all()


def test_numerical_jacobian():
    F = lambda v: np.array([v[0] * v[1], np.sin(v[0]), v[1] ** 2 + v[2]])
    x = np.array([0.5, -1.0, 2.0])
    J = numerical_jacobian(F, x)
    expected = np.array([[x[1], x[0], 0], [np.cos(x[0]), 0, 0], [0, 2 * x[1], 1]])
    np.testing.assert_allclose(J, expected, atol=1e-6)


def test_softmax_jacobian():
    z = rng.normal(size=5)
    softmax = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
    np.testing.assert_allclose(softmax_jacobian(z), numerical_jacobian(softmax, z), atol=1e-8)
    np.testing.assert_allclose(softmax_jacobian(z).sum(axis=0), 0, atol=1e-12)


def test_gradient_check_catches_bugs():
    f = lambda v: (v ** 3).sum()
    good = lambda v: 3 * v ** 2
    buggy = lambda v: 3 * v                      # forgot the square
    x = rng.normal(size=6)
    assert gradient_check(f, good, x) < 1e-7
    assert gradient_check(f, buggy, x) > 1e-2
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0


def test_layer_backward():
    W, b, x, t = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=3), rng.normal(size=4)

    def loss_of(W_, b_, x_):
        y = np.maximum(W_ @ x_ + b_, 0)
        return 0.5 * ((y - t) ** 2).sum()

    loss, dW, db, dx = layer_backward(W, b, x, t)
    assert loss == pytest.approx(loss_of(W, b, x))
    assert dW.shape == (4, 3) and db.shape == (4,) and dx.shape == (3,)
    np.testing.assert_allclose(dW, numerical_gradient(lambda m: loss_of(m, b, x), W), atol=1e-6)
    np.testing.assert_allclose(db, numerical_gradient(lambda v: loss_of(W, v, x), b), atol=1e-6)
    np.testing.assert_allclose(dx, numerical_gradient(lambda v: loss_of(W, b, v), x), atol=1e-6)
