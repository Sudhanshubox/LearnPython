import numpy as np
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    border_mask,
    channels_first,
    checkerboard,
    describe,
    flatten_images,
    grid,
    normalize_rows,
    one_hot,
    random_dataset,
    relu_copy,
    relu_inplace,
    remove_outliers,
    row_argmax,
    to_uint8,
    top_left,
)

NO_LOOPS = [
    "grid", "checkerboard", "border_mask", "relu_copy", "relu_inplace", "remove_outliers",
    "normalize_rows", "flatten_images", "channels_first", "to_uint8", "one_hot", "row_argmax",
]


@pytest.mark.parametrize("name", NO_LOOPS)
def test_no_python_loops(name):
    assert not has_loops(getattr(exercises, name)), f"{name}: use array operations, not loops"


def test_grid():
    g = grid(2, 3)
    np.testing.assert_array_equal(g, [[0, 1, 2], [3, 4, 5]])
    assert np.issubdtype(g.dtype, np.integer)
    assert grid(4, 1).shape == (4, 1)


def test_describe():
    x = np.zeros((32, 3, 28, 28), dtype=np.float32)
    assert describe(x) == {"shape": (32, 3, 28, 28), "ndim": 4, "size": 75264, "dtype": "float32", "nbytes": 301056}


def test_checkerboard():
    np.testing.assert_array_equal(checkerboard(3), [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    np.testing.assert_array_equal(checkerboard(2), [[1, 0], [0, 1]])
    assert checkerboard(8).sum() == 32


def test_border_mask():
    a = np.arange(20).reshape(4, 5)
    m = border_mask(a)
    assert m.dtype == bool and m.shape == (4, 5)
    assert m.sum() == 14
    assert not m[1:-1, 1:-1].any()
    assert border_mask(np.zeros((1, 3))).all()


def test_top_left_is_a_view():
    a = np.arange(16).reshape(4, 4)
    v = top_left(a, 2)
    np.testing.assert_array_equal(v, [[0, 1], [4, 5]])
    assert np.shares_memory(a, v)
    v[0, 0] = 99
    assert a[0, 0] == 99


def test_relu_copy_and_inplace():
    a = np.array([-2.0, 0.0, 3.0])
    r = relu_copy(a)
    np.testing.assert_array_equal(r, [0, 0, 3])
    np.testing.assert_array_equal(a, [-2, 0, 3])
    assert not np.shares_memory(a, r)
    assert relu_inplace(a) is None
    np.testing.assert_array_equal(a, [0, 0, 3])


def test_remove_outliers():
    x = np.array([10, 11, 9, 10, 12, 8, 100.0])
    kept = remove_outliers(x)
    assert 100 not in kept and len(kept) == 6
    data = np.random.default_rng(0).normal(0, 1, 10_000)
    assert 0.94 < len(remove_outliers(data, z=2)) / 10_000 < 0.97   # about 95% within 2 std


def test_normalize_rows():
    out = normalize_rows(np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]]))
    np.testing.assert_allclose(out, [[0.25, 0.75], [0, 0], [0.5, 0.5]])
    assert not np.isnan(out).any()


def test_image_reshapes():
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    f = flatten_images(x)
    assert f.shape == (2, 12)
    np.testing.assert_array_equal(f[1], np.arange(12, 24))
    batch = np.random.default_rng(1).random((5, 28, 28, 3))
    cf = channels_first(batch)
    assert cf.shape == (5, 3, 28, 28)
    assert cf[2, 1, 10, 7] == batch[2, 10, 7, 1]


def test_to_uint8():
    x = np.array([[0.0, 0.5, 1.0], [-0.01, 1.02, 0.2]])
    out = to_uint8(x)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[0, 128, 255], [0, 255, 51]])


def test_one_hot():
    oh = one_hot(np.array([2, 0, 1, 2]), 3)
    np.testing.assert_array_equal(oh, [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert oh.dtype == np.float32
    assert one_hot([1], 4).shape == (1, 4)


def test_row_argmax():
    scores = np.array([[0.1, 0.7, 0.2], [2.0, -1.0, 0.5]])
    idx, vals = row_argmax(scores)
    np.testing.assert_array_equal(idx, [1, 0])
    np.testing.assert_array_equal(vals, [0.7, 2.0])


def test_random_dataset_reproducible():
    X, y = random_dataset(100, 5, 3, seed=7)
    assert X.shape == (100, 5) and y.shape == (100,)
    assert set(np.unique(y)) <= {0, 1, 2}
    X2, y2 = random_dataset(100, 5, 3, seed=7)
    np.testing.assert_array_equal(X, X2)
    rng = np.random.default_rng(7)
    np.testing.assert_array_equal(X, rng.standard_normal((100, 5)))
    np.testing.assert_array_equal(y, rng.integers(0, 3, 100))
