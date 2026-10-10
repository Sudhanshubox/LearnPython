import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from exercises import (
    MLP,
    he_init,
    iterate_minibatches,
    linear_backward,
    linear_forward,
    numerical_grads,
    relu_backward,
    relu_forward,
    sgd_momentum_step,
    softmax_cross_entropy,
    train,
)


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + h; up = f()
        x[idx] = old - h; down = f()
        x[idx] = old
        g[idx] = (up - down) / (2 * h)
    return g


def rel_error(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_he_init(rng):
    W = he_init(500, 300, np.random.default_rng(1))
    assert W.shape == (500, 300)
    assert W.std() == pytest.approx(np.sqrt(2 / 500), rel=0.02)
    assert np.array_equal(W, np.random.default_rng(1).normal(0.0, np.sqrt(2 / 500), (500, 300)))


def test_linear(rng):
    X, W, b = rng.normal(size=(5, 4)), rng.normal(size=(4, 3)), rng.normal(size=3)
    out, cache = linear_forward(X, W, b)
    assert np.allclose(out, X @ W + b)
    G = rng.normal(size=(5, 3))
    dX, dW, db = linear_backward(G, cache)
    assert dX.shape == X.shape and dW.shape == W.shape and db.shape == b.shape
    f = lambda: float(np.sum(linear_forward(X, W, b)[0] * G))
    assert rel_error(dX, num_grad(f, X)) < 1e-7
    assert rel_error(dW, num_grad(f, W)) < 1e-7
    assert rel_error(db, num_grad(f, b)) < 1e-7


def test_relu(rng):
    z = rng.normal(size=(4, 6))
    out, cache = relu_forward(z)
    assert np.array_equal(out, np.maximum(0, z))
    G = rng.normal(size=z.shape)
    assert np.array_equal(relu_backward(G, cache), G * (z > 0))


def test_softmax_cross_entropy(rng):
    logits, y = rng.normal(size=(6, 4)), np.array([0, 3, 1, 1, 2, 0])
    loss, d = softmax_cross_entropy(logits, y)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert isinstance(loss, float)
    assert loss == pytest.approx(-np.mean(np.log(p[np.arange(6), y])))
    assert d.shape == logits.shape
    assert rel_error(d, num_grad(lambda: softmax_cross_entropy(logits, y)[0], logits)) < 1e-7
    big, dbig = softmax_cross_entropy(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([1, 1]))
    assert np.isfinite(big) and np.all(np.isfinite(dbig))
    assert big == pytest.approx(500.0)
    uniform, _ = softmax_cross_entropy(np.zeros((3, 10)), np.array([1, 2, 3]))
    assert uniform == pytest.approx(np.log(10)), "at init, the loss should be about ln(k)"


def test_mlp_init():
    m = MLP([4, 5, 3], seed=7)
    assert sorted(m.params) == ["W1", "W2", "b1", "b2"]
    assert m.params["W1"].shape == (4, 5) and m.params["b2"].shape == (3,)
    rng = np.random.default_rng(7)
    assert np.array_equal(m.params["W1"], he_init(4, 5, rng))
    assert np.array_equal(m.params["W2"], he_init(5, 3, rng))
    assert not m.params["b1"].any()


def test_mlp_forward():
    m = MLP([4, 5, 3], seed=0)
    X = np.random.default_rng(1).normal(size=(6, 4))
    p = m.params
    expected = np.maximum(0, X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    assert np.allclose(m.forward(X), expected)
    assert np.array_equal(m.predict(X), expected.argmax(axis=1))


@pytest.mark.parametrize("sizes, wd", [([4, 5, 3], 0.0), ([3, 6, 5, 4], 0.1)])
def test_mlp_gradients(sizes, wd):
    m = MLP(sizes, seed=0)
    for k in m.params:
        if k.startswith("b"):
            m.params[k] = np.random.default_rng(3).normal(size=m.params[k].shape) * 0.1
    X = np.random.default_rng(1).normal(size=(7, sizes[0]))
    y = np.random.default_rng(2).integers(0, sizes[-1], size=7)
    loss, grads = m.loss_and_grads(X, y, weight_decay=wd)
    assert sorted(grads) == sorted(m.params)
    for k, p in m.params.items():
        expected = num_grad(lambda: m.loss_and_grads(X, y, weight_decay=wd)[0], p)
        assert grads[k].shape == p.shape
        assert rel_error(grads[k], expected) < 1e-6, f"gradient for {k} is wrong"
    if wd:
        plain, _ = m.loss_and_grads(X, y)
        penalty = 0.5 * wd * sum(np.sum(m.params[k] ** 2) for k in m.params if k.startswith("W"))
        assert loss == pytest.approx(plain + penalty)


@pytest.mark.timeout(60)   # the first `import torch` can take several seconds
def test_matches_pytorch():
    torch = pytest.importorskip("torch")
    m = MLP([4, 6, 3], seed=0)
    X = np.random.default_rng(1).normal(size=(5, 4))
    y = np.array([0, 2, 1, 2, 0])
    loss, grads = m.loss_and_grads(X, y)
    t = {k: torch.tensor(v, requires_grad=True) for k, v in m.params.items()}
    logits = torch.relu(torch.tensor(X) @ t["W1"] + t["b1"]) @ t["W2"] + t["b2"]
    tl = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
    tl.backward()
    assert loss == pytest.approx(tl.item())
    for k in t:
        assert np.allclose(grads[k], t[k].grad.numpy())


def test_numerical_grads():
    m = MLP([3, 4, 2], seed=0)
    X = np.random.default_rng(1).normal(size=(5, 3))
    y = np.array([0, 1, 1, 0, 1])
    before = {k: v.copy() for k, v in m.params.items()}
    num = numerical_grads(m, X, y)
    for k in m.params:
        assert np.array_equal(m.params[k], before[k]), "restore the parameters"
        assert num[k].shape == before[k].shape
        expected = num_grad(lambda: m.loss_and_grads(X, y)[0], m.params[k])
        assert np.allclose(num[k], expected, atol=1e-6)


def test_iterate_minibatches():
    batches = list(iterate_minibatches(10, 4, np.random.default_rng(0)))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert np.array_equal(np.concatenate(batches), np.random.default_rng(0).permutation(10))
    rng = np.random.default_rng(0)
    first = np.concatenate(list(iterate_minibatches(10, 4, rng)))
    second = np.concatenate(list(iterate_minibatches(10, 4, rng)))
    assert not np.array_equal(first, second), "each epoch gets a fresh shuffle"


def test_sgd_momentum_step():
    params = {"W": np.array([1.0, 2.0])}
    W = params["W"]
    velocity = {}
    sgd_momentum_step(params, {"W": np.array([1.0, -1.0])}, velocity, lr=0.1, momentum=0.9)
    assert np.allclose(params["W"], [0.9, 2.1])
    assert params["W"] is W, "update the array in place"
    sgd_momentum_step(params, {"W": np.array([1.0, -1.0])}, velocity, lr=0.1, momentum=0.9)
    assert np.allclose(velocity["W"], [-0.19, 0.19])
    assert np.allclose(params["W"], [0.71, 2.29])


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.25, random_state=0, stratify=y)


@pytest.mark.timeout(60)
def test_train_digits(digits):
    X_train, X_val, y_train, y_val = digits
    model = MLP([64, 64, 10], seed=0)
    history = train(model, X_train, y_train, X_val, y_val, epochs=15, lr=0.05, batch_size=64)
    assert len(history["train_loss"]) == 15 and len(history["val_acc"]) == 15
    assert history["train_loss"][0] < np.log(10)
    assert history["train_loss"][-1] < 0.2 * history["train_loss"][0]
    assert history["val_acc"][-1] > 0.95
    again = train(MLP([64, 64, 10], seed=0), X_train, y_train, X_val, y_val, epochs=2)
    first = train(MLP([64, 64, 10], seed=0), X_train, y_train, X_val, y_val, epochs=2)
    assert again == first, "same seeds, same results"
