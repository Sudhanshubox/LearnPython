"""m26 exercises: the data model. Replace each `raise NotImplementedError`."""

import math
from functools import total_ordering


# 1. An immutable mathematical vector.
#    Vector(1, 2, 3)
#    - repr(v) -> "Vector(1, 2, 3)"   (for one component: "Vector(5)")
#    - len(v), v[i], v[1:] (returns a Vector), iteration, `x in v`
#    - v + w, v - w: elementwise; ValueError if lengths differ; NotImplemented for non-Vectors
#    - v * 3 and 3 * v: scalar multiplication;  v / 2: scalar division
#    - -v: negation
#    - v @ w: dot product (a number); ValueError if lengths differ
#    - abs(v): Euclidean length (norm)
#    - bool(v): False only if every component is 0
#    - ==, and hashable (so vectors can be dict keys / set members)
#    - normalized(): a Vector of length 1 in the same direction (ValueError for the zero vector)
class Vector:
    def __init__(self, *components):
        raise NotImplementedError


# 2. A matrix built from rows.
#    Matrix([[1, 2], [3, 4]])
#    - ValueError if rows have different lengths or there are no rows
#    - shape -> (rows, cols)  (a property)
#    - m[i] -> row i as a Vector;  m[i, j] -> a single number
#    - T -> the transpose, as a new Matrix (a property)
#    - m @ Vector -> Vector  (matrix-vector product)
#      m @ Matrix -> Matrix  (matrix-matrix product); ValueError if shapes don't match
#    - ==  compares all entries
#    - repr(m) -> "Matrix([[1, 2], [3, 4]])"
class Matrix:
    def __init__(self, rows):
        raise NotImplementedError


# 3. A dataset that works like a PyTorch Dataset: only __len__ and __getitem__.
#    Dataset(xs, ys) stores pairs; ds[i] -> (xs[i], ys[i]); len(ds).
#    Negative indices work; out of range raises IndexError. Do NOT define __iter__:
#    `for x, y in ds` must still work through __getitem__.
#    Also: ds.batches(size) yields lists of (x, y) pairs of that size (last may be smaller).
class Dataset:
    def __init__(self, xs, ys):
        raise NotImplementedError


# 4. Semantic versions that sort correctly ("1.10.0" > "1.9.3", unlike string comparison).
#    Version("1.10.0"); str(v) -> "1.10.0"; repr(v) -> "Version('1.10.0')"
#    Supports ==, <, <=, >, >= and hashing. Define __eq__ and __lt__, then put
#    @total_ordering on the line above `class Version:` to get the rest for free.
#    (It's not there yet because it fails on a class with no comparison methods.)
#    Missing parts count as 0: Version("2") == Version("2.0.0").
class Version:
    def __init__(self, text):
        raise NotImplementedError


# 5. A polynomial you can call like a function.
#    Polynomial(1, 0, 2) means 1 + 0x + 2x².  p(3) -> 19
#    p + q adds polynomials; p.derivative() returns the derivative as a Polynomial
#    (Polynomial(1, 0, 2).derivative() == Polynomial(0, 4)); equality ignores trailing zeros.
class Polynomial:
    def __init__(self, *coeffs):
        raise NotImplementedError
