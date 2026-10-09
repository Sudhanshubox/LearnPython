"""m36 exercises: NumPy fundamentals.

Unless an exercise says otherwise, use NumPy operations only: no Python loops or
comprehensions (the tests check).
"""

import numpy as np


# 1. An n x m integer array containing 0, 1, 2, ... row by row.
#    grid(2, 3) -> [[0, 1, 2], [3, 4, 5]]
def grid(n, m):
    raise NotImplementedError


# 2. A dict describing an array: {"shape": ..., "ndim": ..., "size": ..., "dtype": str, "nbytes": ...}
#    The dtype as a string, e.g. "float32".
def describe(a):
    raise NotImplementedError


# 3. An n x n checkerboard of 0s and 1s, with 1 in the top-left corner.
#    Use slicing with steps (e.g. board[::2, ::2]), not loops.
#    checkerboard(3) -> [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
def checkerboard(n):
    raise NotImplementedError


# 4. A 2-D array's border: a NEW boolean array of the same shape that is True on the
#    outer rows and columns and False inside.
def border_mask(a):
    raise NotImplementedError


# 5. Views and copies.
#    top_left(a, k): the k x k top-left block of a 2-D array as a VIEW (shares memory).
#    relu_copy(a): a NEW array with negatives replaced by 0; `a` is unchanged.
#    relu_inplace(a): replace negatives in `a` itself by 0 and return None.
def top_left(a, k):
    raise NotImplementedError


def relu_copy(a):
    raise NotImplementedError


def relu_inplace(a):
    raise NotImplementedError


# 6. Remove outliers: keep only values within `z` standard deviations of the mean
#    (|x - mean| <= z * std, using the population std, x.std()). Returns a 1-D array.
def remove_outliers(x, z=2.0):
    raise NotImplementedError


# 7. Make each row of a 2-D non-negative array sum to 1 (rows of all zeros stay zeros).
#    normalize_rows([[1, 3], [0, 0]]) -> [[0.25, 0.75], [0, 0]]
def normalize_rows(a):
    raise NotImplementedError


# 8. Image batch reshaping.
#    flatten_images(x): (N, H, W) -> (N, H*W)
#    channels_first(x): (N, H, W, C) -> (N, C, H, W)
def flatten_images(x):
    raise NotImplementedError


def channels_first(x):
    raise NotImplementedError


# 9. Convert float images in [0, 1] to uint8 pixels in [0, 255]: multiply by 255,
#    round to the nearest integer, clip to 0..255 (values may be slightly out of range),
#    and return dtype uint8.
def to_uint8(x):
    raise NotImplementedError


# 10. One-hot encoding: labels (1-D ints) -> float32 array of shape (len(labels), num_classes)
#     with a 1.0 in each row at the label's column. Use fancy indexing.
#     one_hot([2, 0], 3) -> [[0, 0, 1], [1, 0, 0]]
def one_hot(labels, num_classes):
    raise NotImplementedError


# 11. For each row of a 2-D array, the column index of its largest value, and that value.
#     Return (indices, values) as two 1-D arrays. (This is how a classifier picks its
#     predicted class from its scores.)
def row_argmax(scores):
    raise NotImplementedError


# 12. Reproducible random data: using np.random.default_rng(seed), return
#     (X, y) where X has shape (n, d) of standard normal samples and y has shape (n,)
#     of random integers 0..num_classes-1. Generate X first, then y, from the same rng.
def random_dataset(n, d, num_classes, seed=0):
    raise NotImplementedError
