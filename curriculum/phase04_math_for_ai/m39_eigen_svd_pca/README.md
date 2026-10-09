# m39 · Linear algebra II: eigenvectors, SVD and PCA

**By the end you can:** explain what eigenvectors and eigenvalues mean, find them with power iteration, decompose any matrix with the SVD, compress data with low-rank approximations, and implement PCA from scratch.

**Why it matters for AI:** PCA is the standard way to visualize and compress high-dimensional data (embeddings, activations). The SVD underlies recommender systems (m45), latent semantic analysis, and **LoRA**, the most popular way to fine-tune large language models cheaply, which assumes weight updates are low-rank. Eigenvectors explain PageRank (m20), the stability of training dynamics, and why some networks' gradients explode.

---

## 1. Eigenvectors: directions a matrix only stretches

Most vectors change direction when multiplied by a matrix. An **eigenvector** v of A doesn't; it only gets scaled by its **eigenvalue** λ:

A v = λ v

```python
A = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)      # eigh: for symmetric matrices (real, sorted ascending)
# vals = [1, 3]; vecs[:, 1] ∝ [1, 1] is stretched ×3, vecs[:, 0] ∝ [1, -1] is kept ×1
```

**Symmetric** matrices (like covariance matrices) are especially nice: real eigenvalues, and orthogonal eigenvectors. So A = V Λ Vᵀ: rotate into the eigenvector basis, scale each axis, rotate back. This makes powers easy: Aᵏ = V Λᵏ Vᵀ.

## 2. Power iteration

Multiply any starting vector by A again and again (normalizing each time). The component along the eigenvector with the largest |λ| grows fastest, so the vector turns towards it:

```python
v = rng.normal(size=n)
for _ in range(100):
    v = A @ v
    v /= np.linalg.norm(v)
lam = v @ A @ v          # the Rayleigh quotient: the eigenvalue estimate
```

PageRank (m20) is exactly this: power iteration on the link matrix, converging to the eigenvector with eigenvalue 1, the **stationary distribution** of a random walk.

## 3. The singular value decomposition (SVD)

Every matrix, any shape, can be written as

A = U Σ Vᵀ

- U (m×m) and V (n×n) are orthogonal (rotations/reflections),
- Σ is diagonal with **singular values** σ₁ ≥ σ₂ ≥ … ≥ 0.

Any linear map is "rotate, stretch along the axes, rotate". In NumPy: `U, s, Vt = np.linalg.svd(A, full_matrices=False)`.

## 4. Low-rank approximation

Keep only the k largest singular values:

A_k = U[:, :k] diag(s[:k]) Vt[:k, :]

**Eckart–Young theorem:** A_k is the *best* rank-k approximation of A, and its error is √(σ²ₖ₊₁ + … + σ²ᵣ) (in the Frobenius norm). If the singular values drop off quickly, a small k captures almost everything.

- An m×n matrix has m·n numbers; a rank-k factorization stores k(m + n + 1). For a 1000×1000 image with k = 50, that's about 10% of the storage.
- **LoRA:** instead of updating a d×d weight matrix W (d² parameters), learn W + B A with B (d×r) and A (r×d), only 2dr parameters. For d = 4096 and r = 8, that's 256× fewer.

## 5. PCA: the directions of greatest variance

Given data X (n examples × d features):

1. **Center** it: subtract the mean of each feature.
2. Compute the **covariance matrix** C = XᶜᵀXᶜ / (n − 1), shape (d, d). Entry (i, j) is how features i and j vary together.
3. Eigen-decompose C. The eigenvectors are the **principal components** (directions); the eigenvalues are the **variance** along each.
4. Sort by eigenvalue (largest first), keep the top k, and project: Z = Xᶜ Wₖ.

The **explained variance ratio** λᵢ / Σλ tells you how much of the data's spread each component captures. PCA can also be computed directly with the SVD of Xᶜ (more numerically stable): the components are the rows of Vt, and the variances are s² / (n − 1).

Eigenvectors are only defined up to sign (v and −v are both valid), so different libraries may return components with flipped signs. Compare results up to sign.

---

## Problem-solving habit #36: look at the spectrum

When a matrix appears in a problem (a covariance, a weight matrix, a graph), plot or print its eigenvalues or singular values. A sharp drop-off means low-rank structure you can exploit; a huge ratio between the largest and smallest means numerical trouble (m38's condition number). The spectrum tells you a matrix's personality.

## Go deeper (optional, research-level)

1. Read the abstract and Figure 1 of *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021). What rank r did they find sufficient, and why might fine-tuning updates be low-rank?
2. *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (Aghajanyan et al., 2020) goes further. How few dimensions were enough to fine-tune RoBERTa?
3. PCA finds linear structure only. Read about t-SNE and UMAP, which people use to visualize embeddings. What can they show that PCA can't, and what can they mislead you about?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
