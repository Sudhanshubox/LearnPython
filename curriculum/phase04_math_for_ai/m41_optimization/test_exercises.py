import math

import numpy as np
import pytest

from exercises import (
    adam,
    closed_form,
    fit_gd,
    fit_sgd,
    gradient_descent,
    looks_convex,
    lr_schedule,
    max_stable_lr,
    momentum,
    rosenbrock,
    rosenbrock_grad,
)

rng = np.random.default_rng(41)
bowl_grad = lambda x: 2 * (x - np.array([3.0, -1.0]))         # minimum at (3, -1)


def test_gradient_descent_bowl():
    x0 = np.zeros(2)
    x, path = gradient_descent(bowl_grad, x0, lr=0.1, steps=100)
    np.testing.assert_allclose(x, [3, -1], atol=1e-6)
    assert path.shape == (101, 2)
    np.testing.assert_array_equal(path[0], [0, 0])
    np.testing.assert_array_equal(x0, [0, 0]), "don't modify x0"


def test_gradient_descent_diverges_with_large_lr():
    x, _ = gradient_descent(lambda x: 2 * x, np.array([1.0]), lr=1.1, steps=50)
    assert abs(x[0]) > 1000


def test_momentum_and_adam_on_bowl():
    np.testing.assert_allclose(momentum(bowl_grad, np.zeros(2), lr=0.05, steps=300), [3, -1], atol=1e-4)
    np.testing.assert_allclose(adam(bowl_grad, np.zeros(2), lr=0.1, steps=2000), [3, -1], atol=1e-3)


def test_adam_first_step_is_lr_sized():
    # With bias correction, Adam's first step moves each coordinate by about lr, whatever the gradient scale.
    x = adam(lambda x: np.array([1000.0, 0.001]), np.zeros(2), lr=0.1, steps=1)
    np.testing.assert_allclose(x, [-0.1, -0.1], rtol=1e-4)


def test_rosenbrock():
    assert rosenbrock(np.array([1.0, 1.0])) == 0
    assert rosenbrock(np.array([0.0, 0.0])) == 1
    p = np.array([-1.2, 1.0])
    h = 1e-6
    num = np.array([(rosenbrock(p + e) - rosenbrock(p - e)) / (2 * h) for e in np.eye(2) * h])
    np.testing.assert_allclose(rosenbrock_grad(p), num, rtol=1e-6)


def test_optimizers_race_on_rosenbrock():
    start = np.array([-1.2, 1.0])
    gd_x, _ = gradient_descent(rosenbrock_grad, start, lr=1e-3, steps=3000)
    mom_x = momentum(rosenbrock_grad, start, lr=1e-3, steps=3000, beta=0.9)
    adam_x = adam(rosenbrock_grad, start, lr=0.02, steps=3000)
    target = np.array([1.0, 1.0])
    dist = lambda p: np.linalg.norm(p - target)
    assert dist(mom_x) < dist(gd_x), "momentum should get further along the valley than plain GD"
    assert dist(adam_x) < 0.05


def test_max_stable_lr():
    A = np.diag([1.0, 10.0])
    assert max_stable_lr(A) == pytest.approx(0.2)
    grad = lambda x: A @ x
    stable, _ = gradient_descent(grad, np.ones(2), lr=0.19, steps=500)
    unstable, _ = gradient_descent(grad, np.ones(2), lr=0.21, steps=500)
    assert np.linalg.norm(stable) < 1e-6
    assert np.linalg.norm(unstable) > 1e3


@pytest.fixture
def regression():
    X = rng.normal(size=(500, 3))
    w_true, b_true = np.array([1.5, -2.0, 0.5]), 3.0
    y = X @ w_true + b_true + rng.normal(0, 0.1, 500)
    return X, y, w_true, b_true


def test_closed_form(regression):
    X, y, w_true, b_true = regression
    w, b = closed_form(X, y)
    np.testing.assert_allclose(w, w_true, atol=0.02)
    assert b == pytest.approx(b_true, abs=0.02)


def test_fit_gd_matches_closed_form(regression):
    X, y, _, _ = regression
    w, b, losses = fit_gd(X, y, lr=0.1, epochs=500)
    w_cf, b_cf = closed_form(X, y)
    np.testing.assert_allclose(w, w_cf, atol=1e-4)
    assert b == pytest.approx(b_cf, abs=1e-4)
    assert len(losses) == 500
    assert losses[0] == pytest.approx(np.mean(y ** 2))
    assert all(a >= b - 1e-12 for a, b in zip(losses, losses[1:])), "full-batch GD with a small lr never increases the loss"


def test_fit_sgd(regression):
    X, y, w_true, b_true = regression
    w, b = fit_sgd(X, y, lr=0.05, epochs=20, batch_size=32, seed=1)
    np.testing.assert_allclose(w, w_true, atol=0.05)
    assert b == pytest.approx(b_true, abs=0.05)
    w2, b2 = fit_sgd(X, y, lr=0.05, epochs=20, batch_size=32, seed=1)
    np.testing.assert_array_equal(w, w2)


def test_lr_schedule():
    assert lr_schedule(0, 1000, 1e-3, warmup=100) == 0
    assert lr_schedule(50, 1000, 1e-3, warmup=100) == pytest.approx(5e-4)
    assert lr_schedule(100, 1000, 1e-3, 1e-5, warmup=100) == pytest.approx(1e-3)
    assert lr_schedule(550, 1000, 1e-3, 1e-5, warmup=100) == pytest.approx((1e-3 + 1e-5) / 2)
    assert lr_schedule(1000, 1000, 1e-3, 1e-5, warmup=100) == pytest.approx(1e-5)
    assert lr_schedule(0, 10, 1.0) == pytest.approx(1.0)
    values = [lr_schedule(s, 1000, 1e-3, warmup=100) for s in range(100, 1001)]
    assert all(a >= b for a, b in zip(values, values[1:])), "decays monotonically after warmup"


def test_looks_convex():
    xs = np.linspace(-3, 3, 50)
    assert looks_convex(lambda x: x ** 2, xs)
    assert looks_convex(np.exp, xs)
    assert looks_convex(lambda x: abs(x), xs)
    assert not looks_convex(np.sin, xs)
    assert not looks_convex(lambda x: -x ** 2, xs)
