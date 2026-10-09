import math

import pytest

from exercises import Dataset, Matrix, Polynomial, Vector, Version

# ---------------------------------------------------------------- Vector


def test_vector_basics():
    v = Vector(1, 2, 3)
    assert repr(v) == "Vector(1, 2, 3)"
    assert repr(Vector(5)) == "Vector(5)"
    assert len(v) == 3
    assert v[0] == 1 and v[-1] == 3
    assert v[1:] == Vector(2, 3)
    assert list(v) == [1, 2, 3]
    assert 2 in v and 9 not in v


def test_vector_arithmetic():
    v, w = Vector(1, 2), Vector(3, 4)
    assert v + w == Vector(4, 6)
    assert w - v == Vector(2, 2)
    assert v * 3 == Vector(3, 6)
    assert 3 * v == Vector(3, 6)
    assert w / 2 == Vector(1.5, 2)
    assert -v == Vector(-1, -2)
    assert v @ w == 11
    assert abs(w) == 5
    assert v == Vector(1, 2), "operators must not modify their operands"


def test_vector_errors():
    with pytest.raises(ValueError):
        Vector(1, 2) + Vector(1, 2, 3)
    with pytest.raises(ValueError):
        Vector(1) @ Vector(1, 2)
    with pytest.raises(TypeError):
        Vector(1, 2) + 5
    with pytest.raises(TypeError):
        Vector(1, 2) + [1, 2]


def test_vector_bool_hash_normalized():
    assert not Vector(0, 0)
    assert Vector(0, 1)
    assert len({Vector(1, 2), Vector(1, 2), Vector(2, 1)}) == 2
    assert {Vector(1, 2): "a"}[Vector(1, 2)] == "a"
    n = Vector(3, 4).normalized()
    assert n[0] == pytest.approx(0.6) and n[1] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        Vector(0, 0).normalized()


def test_vector_is_immutable():
    v = Vector(1, 2)
    with pytest.raises(TypeError):
        v[0] = 5


# ---------------------------------------------------------------- Matrix


def test_matrix_basics():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape == (2, 3)
    assert m[1] == Vector(4, 5, 6)
    assert m[0, 2] == 3
    assert m.T == Matrix([[1, 4], [2, 5], [3, 6]])
    assert repr(Matrix([[1, 2], [3, 4]])) == "Matrix([[1, 2], [3, 4]])"


def test_matrix_validation():
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3]])
    with pytest.raises(ValueError):
        Matrix([])


def test_matrix_products():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    assert a @ b == Matrix([[19, 22], [43, 50]])
    assert a @ Vector(1, 1) == Vector(3, 7)
    assert Matrix([[1, 2, 3]]) @ Matrix([[4], [5], [6]]) == Matrix([[32]])
    with pytest.raises(ValueError):
        a @ Vector(1, 2, 3)
    with pytest.raises(ValueError):
        Matrix([[1, 2]]) @ Matrix([[1, 2]])


def test_matrix_rows_not_shared():
    rows = [[1, 2], [3, 4]]
    m = Matrix(rows)
    rows[0][0] = 99
    assert m[0, 0] == 1


def test_neural_layer_forward():
    # y = W @ x + b: exactly a neural network layer
    W = Matrix([[0.5, -1.0], [2.0, 0.0]])
    x = Vector(2, 1)
    b = Vector(1, -1)
    assert W @ x + b == Vector(1.0, 3.0)


# ---------------------------------------------------------------- Dataset


def test_dataset_protocol():
    ds = Dataset([1, 2, 3], ["a", "b", "c"])
    assert len(ds) == 3
    assert ds[0] == (1, "a")
    assert ds[-1] == (3, "c")
    with pytest.raises(IndexError):
        ds[3]
    assert [pair for pair in ds] == [(1, "a"), (2, "b"), (3, "c")]
    assert "__iter__" not in vars(Dataset), "iteration should work through __getitem__ alone"


def test_dataset_batches():
    ds = Dataset(list(range(5)), list("abcde"))
    batches = list(ds.batches(2))
    assert batches == [[(0, "a"), (1, "b")], [(2, "c"), (3, "d")], [(4, "e")]]


# ---------------------------------------------------------------- Version


def test_version_ordering():
    assert Version("1.10.0") > Version("1.9.3")
    assert Version("2") == Version("2.0.0")
    assert Version("1.2.3") <= Version("1.2.3")
    assert Version("0.9") < Version("1.0")
    versions = ["1.10.0", "1.2.0", "1.9.3", "0.1"]
    assert [str(v) for v in sorted(map(Version, versions))] == ["0.1", "1.2.0", "1.9.3", "1.10.0"]
    assert repr(Version("1.0")) == "Version('1.0')"
    assert len({Version("2"), Version("2.0")}) == 1


# ---------------------------------------------------------------- Polynomial


def test_polynomial():
    p = Polynomial(1, 0, 2)
    assert p(3) == 19
    assert p(0) == 1
    assert p + Polynomial(0, 1) == Polynomial(1, 1, 2)
    assert p.derivative() == Polynomial(0, 4)
    assert Polynomial(5).derivative() == Polynomial(0)
    assert Polynomial(1, 2, 0, 0) == Polynomial(1, 2)
    assert Polynomial(1, 2) + Polynomial(0, -2) == Polynomial(1)
