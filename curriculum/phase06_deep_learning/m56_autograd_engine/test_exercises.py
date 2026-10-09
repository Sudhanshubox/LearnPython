import math
import random

import pytest

from exercises import MLP, Layer, Neuron, Value, cross_entropy, topological_order, train, zero_grad


def grads(f, xs):
    vals = [Value(x) for x in xs]
    out = f(*vals)
    out.backward()
    return out.data, [v.grad for v in vals]


def numeric(f, xs, h=1e-6):
    result = []
    for i in range(len(xs)):
        hi = list(xs); hi[i] += h
        lo = list(xs); lo[i] -= h
        result.append((f(*map(Value, hi)).data - f(*map(Value, lo)).data) / (2 * h))
    return result


def check(f, xs, expected_value):
    value, g = grads(f, xs)
    assert value == pytest.approx(expected_value, rel=1e-9)
    assert g == pytest.approx(numeric(f, xs), rel=1e-5, abs=1e-7)


def test_add_and_mul():
    check(lambda a, b: a + b, [2.0, -3.0], -1.0)
    check(lambda a, b: a * b, [2.0, -3.0], -6.0)
    check(lambda a, b: a * b + a, [2.0, -3.0], -4.0)


def test_returns_new_values_with_graph():
    a, b = Value(2.0), Value(3.0)
    c = a * b
    assert isinstance(c, Value) and c is not a and c is not b
    assert set(map(id, c._prev)) == {id(a), id(b)}
    assert a.data == 2.0 and b.data == 3.0


def test_plain_numbers():
    check(lambda a: a * 3 + 1, [2.0], 7.0)
    check(lambda a: 3 * a, [2.0], 6.0)
    check(lambda a: 1 + a, [2.0], 3.0)
    check(lambda a: 10 - a, [2.0], 8.0)
    check(lambda a: a - 10, [2.0], -8.0)
    check(lambda a: 1 / a, [4.0], 0.25)
    check(lambda a: a / 4, [2.0], 0.5)
    check(lambda a: -a, [2.0], -2.0)


def test_pow_div_sub():
    check(lambda a: a ** 3, [1.5], 3.375)
    check(lambda a: a ** -0.5, [4.0], 0.5)
    check(lambda a, b: a / b, [3.0, -2.0], -1.5)
    check(lambda a, b: a - b, [3.0, -2.0], 5.0)


def test_nonlinearities():
    check(lambda a: a.tanh(), [0.7], math.tanh(0.7))
    check(lambda a: a.exp(), [0.7], math.exp(0.7))
    check(lambda a: a.log(), [0.7], math.log(0.7))
    check(lambda a: a.relu(), [0.7], 0.7)
    check(lambda a: a.relu(), [-0.7], 0.0)
    _, g = grads(lambda a: a.relu(), [-0.7])
    assert g == [0.0]


def test_reused_value_accumulates():
    _, g = grads(lambda a: a + a, [3.0])
    assert g == [2.0], "use += in _backward: a value used twice gets both gradients"
    check(lambda a: a * a * a, [3.0], 27.0)
    check(lambda a, b: (a * b + b) * (a + b.tanh()), [0.5, -1.2], (0.5 * -1.2 - 1.2) * (0.5 + math.tanh(-1.2)))


def test_complex_expression():
    def f(a, b, c):
        d = a * b + c ** 2
        e = (d / (1 + a.exp())).tanh()
        return e * c - (b - a).relu() + (d * d + 1).log()
    a, b, c = 0.3, -1.1, 0.8
    d = a * b + c ** 2
    expected = math.tanh(d / (1 + math.exp(a))) * c - max(0.0, b - a) + math.log(d * d + 1)
    check(f, [a, b, c], expected)


def test_matches_pytorch():
    torch = pytest.importorskip("torch")
    xs = [0.3, -1.1, 0.8]
    def f(a, b, c):
        return ((a * b + c).tanh() * a + (b ** 2) / c).exp() - (c * 3).log()
    _, g = grads(f, xs)
    t = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
    out = torch.exp(torch.tanh(t[0] * t[1] + t[2]) * t[0] + t[1] ** 2 / t[2]) - torch.log(t[2] * 3)
    out.backward()
    assert g == pytest.approx([x.grad.item() for x in t], rel=1e-9)


def test_topological_order():
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    e = d * c
    order = topological_order(e)
    assert len(order) == 5 and len(set(map(id, order))) == 5
    pos = {id(v): i for i, v in enumerate(order)}
    for node in order:
        for child in node._prev:
            assert pos[id(child)] < pos[id(node)], "children must come before their parents"
    assert order[-1] is e


def test_deep_graph_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y + 0.001 * x       # also reuses x 5,000 times
    y.backward()
    assert y.data == pytest.approx(6.0)
    assert x.grad == pytest.approx(6.0)


def test_zero_grad():
    a, b = Value(1.0), Value(2.0)
    (a * b).backward()
    zero_grad([a, b])
    assert a.grad == 0.0 and b.grad == 0.0


def test_neuron():
    rng = random.Random(1)
    n = Neuron(3, rng=random.Random(1))
    expected_w = [rng.uniform(-1, 1) for _ in range(3)]
    params = n.parameters()
    assert len(params) == 4 and all(isinstance(p, Value) for p in params)
    assert [p.data for p in params] == expected_w + [0.0]
    x = [0.5, -1.0, 2.0]
    out = n(x)
    assert out.data == pytest.approx(math.tanh(sum(w * xi for w, xi in zip(expected_w, x))))
    lin = Neuron(3, nonlin=False, rng=random.Random(1))
    assert lin(x).data == pytest.approx(sum(w * xi for w, xi in zip(expected_w, x)))


def test_layer_and_mlp_shapes():
    layer = Layer(3, 4, rng=random.Random(0))
    out = layer([1.0, 2.0, 3.0])
    assert len(out) == 4 and all(isinstance(v, Value) for v in out)
    assert len(layer.parameters()) == 16
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == 41
    assert isinstance(model([1.0, 2.0, 3.0]), Value)
    multi = MLP(2, [5, 3])
    assert len(multi([1.0, 2.0])) == 3


def test_mlp_seed_and_linear_output():
    a, b = MLP(2, [4, 1], seed=3), MLP(2, [4, 1], seed=3)
    assert [p.data for p in a.parameters()] == [p.data for p in b.parameters()]
    rng = random.Random(3)
    first = [rng.uniform(-1, 1) for _ in range(2)]
    assert [p.data for p in a.parameters()[:2]] == first, "one shared rng, used in order"
    big = MLP(1, [1], seed=0)
    big.parameters()[0].data = 10.0
    assert big([5.0]).data == pytest.approx(50.0), "the last layer must be linear"


def test_mlp_gradients():
    model = MLP(2, [3, 1], seed=0)
    x = [0.5, -0.4]
    out = model(x)
    out.backward()
    params = model.parameters()
    for i in [0, 3, 8, len(params) - 1]:
        p = params[i]
        old = p.data
        p.data = old + 1e-6; up = model(x).data
        p.data = old - 1e-6; down = model(x).data
        p.data = old
        assert p.grad == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-8)


def test_cross_entropy():
    logits = [Value(2.0), Value(1.0), Value(0.1)]
    loss = cross_entropy(logits, 0)
    lse = math.log(sum(math.exp(z) for z in [2.0, 1.0, 0.1]))
    assert loss.data == pytest.approx(lse - 2.0)
    loss.backward()
    probs = [math.exp(z - lse) for z in [2.0, 1.0, 0.1]]
    assert [z.grad for z in logits] == pytest.approx([probs[0] - 1, probs[1], probs[2]])
    big = cross_entropy([Value(1000.0), Value(0.0)], 1)
    assert big.data == pytest.approx(1000.0)


@pytest.mark.timeout(60)
def test_train_learns_xor_like_data():
    rng = random.Random(0)
    X, y = [], []
    for _ in range(24):
        a, b = rng.uniform(-1, 1), rng.uniform(-1, 1)
        X.append([a, b])
        y.append(1.0 if a * b > 0 else -1.0)
    model = MLP(2, [8, 1], seed=0)
    losses = train(model, X, y, steps=100, lr=0.2)
    assert len(losses) == 100 and all(isinstance(v, float) for v in losses)
    assert losses[-1] < 0.5 * losses[0]
    accuracy = sum((model(x).data > 0) == (t > 0) for x, t in zip(X, y)) / len(X)
    assert accuracy >= 0.9
