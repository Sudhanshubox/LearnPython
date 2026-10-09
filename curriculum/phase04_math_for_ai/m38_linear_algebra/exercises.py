"""m38 exercises: linear algebra I. Inputs are NumPy arrays of floats."""

import numpy as np


# 1. The dot product WITHOUT np.dot, @, np.inner, np.matmul or np.vdot:
#    multiply elementwise and sum.
def dot(u, v):
    raise NotImplementedError


# 2. The p-norm of a vector for p = 1, 2 or np.inf, computed from the definition
#    (don't use np.linalg.norm).
def norm(v, p=2):
    raise NotImplementedError


# 3. Cosine similarity of two vectors, and the angle between them in degrees.
#    For angle_degrees, clip the cosine to [-1, 1] before arccos (rounding can push it
#    slightly outside).
def cosine_similarity(u, v):
    raise NotImplementedError


def angle_degrees(u, v):
    raise NotImplementedError


# 4. For a matrix E (n, d) whose rows are embeddings, the (n, n) matrix of cosine
#    similarities between all pairs of rows. No loops.
def cosine_similarity_matrix(E):
    raise NotImplementedError


# 5. The projection of u onto v, and the component of u perpendicular to v.
#    project(u, v) + reject(u, v) == u, and reject(u, v) . v == 0.
def project(u, v):
    raise NotImplementedError


def reject(u, v):
    raise NotImplementedError


# 6. The 2x2 matrix rotating points counter-clockwise by `degrees`, and a function
#    that rotates a set of points P of shape (n, 2) (each row is a point).
def rotation_matrix(degrees):
    raise NotImplementedError


def rotate_points(P, degrees):
    raise NotImplementedError


# 7. Gram-Schmidt: given linearly independent vectors as the ROWS of V (k, d), return a
#    (k, d) array whose rows are orthonormal and span the same space, processing rows in
#    order (the first output row is the first input row, normalized).
#    Don't use np.linalg.qr.
def gram_schmidt(V):
    raise NotImplementedError


# 8. Area of the parallelogram spanned by 2-D vectors p and q (always >= 0), using the
#    determinant.
def parallelogram_area(p, q):
    raise NotImplementedError


# 9. Fit y ≈ w·x + b by least squares, where X is (n, d) and y is (n,).
#    Add a column of ones for the bias and use np.linalg.lstsq.
#    Return (w, b) with w of shape (d,) and b a float.
def fit_linear(X, y):
    raise NotImplementedError


# 10. Word analogies with embeddings: "a is to b as c is to ?"
#     emb is a dict {word: 1-D vector}. Compute target = emb[b] - emb[a] + emb[c] and
#     return the word (other than a, b and c) whose embedding has the highest cosine
#     similarity with target.
#     analogy(emb, "man", "king", "woman") -> "queen"
def analogy(emb, a, b, c):
    raise NotImplementedError
