# m38 · Linear algebra I: vectors, matrices and systems

**By the end you can:** think of vectors geometrically (length, angle, projection), see a matrix as a transformation of space, orthogonalize vectors, solve linear systems, and fit a line by least squares, all in NumPy and with an intuition for what each operation means.

**Why it matters for AI:** linear algebra is the language of machine learning. Embeddings are vectors, and "similar meaning" means "small angle" (cosine similarity). Every neural network layer is a matrix transformation. Linear regression is a least-squares problem. Attention is dot products between query and key vectors. If you understand these operations geometrically, research papers become readable.

---

## 1. Vectors

A vector is a list of numbers, and also an arrow in space. In ML a vector usually represents *something*: an example's features, a word's embedding, a model's weights.

```python
u = np.array([3.0, 4.0])
v = np.array([1.0, 0.0])
u + v, 2 * u                    # add arrows tip to tail; stretch
```

**Length (norm):**

- L2 (Euclidean): ‖u‖₂ = √(Σ uᵢ²) = 5 for (3, 4)
- L1 (Manhattan): Σ |uᵢ| = 7
- L∞: max |uᵢ| = 4

`np.linalg.norm(u, ord=...)`. Norms appear in ML as **regularization** (L2 = weight decay, L1 = sparsity).

## 2. The dot product

u · v = Σ uᵢ vᵢ = ‖u‖ ‖v‖ cos θ

The dot product measures how much two vectors point the same way:

- positive: similar direction; 0: perpendicular (**orthogonal**); negative: opposite.
- **Cosine similarity** = u · v / (‖u‖ ‖v‖) ∈ [−1, 1] ignores length and compares only direction. It's how embedding search (m35's dense cousin) decides two texts mean similar things.

## 3. Projection

The projection of u onto v is the "shadow" of u along v's direction:

proj_v(u) = (u · v / v · v) v

What's left, u − proj_v(u), is perpendicular to v. Projections are behind least squares (§7), PCA (m39) and Gram–Schmidt (§5).

## 4. Matrices are transformations

Multiplying by a matrix maps vectors to vectors. Each **column** of A is where the corresponding basis vector lands:

```python
R = np.array([[0, -1],
              [1,  0]])          # rotate 90° counter-clockwise: (1,0) → (0,1), (0,1) → (-1,0)
R @ np.array([2, 1])             # [-1, 2]
```

- Rotation by θ: `[[cos θ, −sin θ], [sin θ, cos θ]]`
- Scaling: a diagonal matrix
- Composition: applying B then A is the single matrix `A @ B` (order matters!)
- **Determinant:** how much the transformation scales area (volume in higher dimensions). det = 0 means it squashes space flat, so it can't be undone (no inverse).

A neural network layer `X @ W` applies the transformation W to every example (row of X) at once.

## 5. Orthogonality and Gram–Schmidt

An **orthonormal** set of vectors are all length 1 and mutually perpendicular. A matrix Q with orthonormal columns satisfies QᵀQ = I, and rotations are like this: they preserve lengths and angles.

**Gram–Schmidt** turns any linearly independent set into an orthonormal one: take each vector, subtract its projections onto the ones already processed, then normalize. That's the QR decomposition (`np.linalg.qr`) done by hand.

## 6. Solving linear systems

```
2x +  y = 5
 x + 3y = 10        →   A = [[2, 1], [1, 3]],  b = [5, 10],  solve A x = b
```

```python
x = np.linalg.solve(A, b)        # [1, 3]
```

Use `solve`, not `np.linalg.inv(A) @ b`: it's faster and more accurate. A system has exactly one solution when A is square and invertible (det ≠ 0, full **rank**).

## 7. Least squares: when there's no exact solution

With more equations than unknowns (100 data points, 2 parameters for a line), there's usually no exact solution. Least squares finds the x minimizing ‖Ax − b‖². Geometrically, Ax is the **projection** of b onto the space spanned by A's columns. Setting the gradient to zero gives the **normal equations**:

AᵀA x = Aᵀb

In code: `np.linalg.lstsq(A, b, rcond=None)` (more stable than forming AᵀA yourself). Fitting `y ≈ w·x + b` means adding a column of 1s to X for the bias. This is linear regression, the subject of m41 and Phase 5.

---

## Problem-solving habit #35: draw it in 2-D first

Every linear-algebra idea has a 2-D picture: a projection is a shadow, a determinant is an area, a rotation is a rotation. When an equation in 768 dimensions confuses you, draw the 2-D case on paper, convince yourself there, then trust the algebra to generalize.

## Go deeper (optional, research-level)

1. Watch 3Blue1Brown's *Essence of Linear Algebra* series (chapters 1–7). It's the best geometric introduction to this material.
2. Word analogies (exercise 10) come from *Efficient Estimation of Word Representations in Vector Space* (Mikolov et al., 2013). Read section 4. Why might analogies work less cleanly than the famous example suggests? (Search for critiques of word-analogy evaluation.)
3. The **condition number** of A (`np.linalg.cond`) measures how much errors in b get amplified in x. Build a nearly singular 2×2 matrix and see how tiny changes in b change the solution. Why does this matter for the normal equations?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
