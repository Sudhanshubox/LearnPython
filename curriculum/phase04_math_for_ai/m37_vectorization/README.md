# m37 · Vectorization and broadcasting

**By the end you can:** replace Python loops with whole-array operations, predict the result shape of any broadcast, compute pairwise distances and softmax without loops, and build a working classifier in a few vectorized lines.

**Why it matters for AI:** this is *the* core skill of numerical ML code. A neural-network layer is one vectorized expression (`X @ W + b`) applied to a whole batch at once. GPUs only pay off when you give them big array operations instead of millions of tiny ones. Code that loops over examples in Python is often 100× too slow to be usable.

---

## 1. Elementwise operations

Arithmetic, comparisons and math functions apply to every element:

```python
x = np.array([1.0, 2.0, 3.0])
x * 10 + 1          # [11. 21. 31.]
np.exp(x), np.sqrt(x), np.maximum(x, 2)
x > 1.5             # [False  True  True]
```

**Universal functions** (ufuncs) like `np.exp` run a C loop over the data. Reductions (`sum`, `mean`, `max`, `argmax`, `cumsum`) take an `axis` (m36).

## 2. Broadcasting

What happens when shapes differ?

```python
X = np.ones((4, 3))        # 4 examples, 3 features
mean = X.mean(axis=0)      # shape (3,)
X - mean                   # (4, 3) - (3,) → (4, 3): mean is subtracted from every row
```

NumPy **stretches** dimensions of size 1 (without copying data) so the shapes match. The rules, comparing shapes **from the right**:

1. If one array has fewer dimensions, pad its shape with 1s on the left.
2. Two dimensions are compatible if they're equal, or one of them is 1.
3. The result takes the larger size in each dimension.

| A | B | Result |
|---|---|---|
| (4, 3) | (3,) | (4, 3) |
| (4, 3) | (4, 1) | (4, 3) |
| (4, 1) | (1, 5) | (4, 5): an "outer" operation |
| (4, 3) | (4,) | **error**: 3 vs 4 from the right |

The last row is the classic bug. Fix it with `keepdims=True` or by adding an axis: `X - row_means[:, None]`.

## 3. The outer-difference trick

Adding axes turns pairwise computations into one expression:

```python
a = np.array([1, 2, 3])           # (3,)
b = np.array([10, 20])            # (2,)
a[:, None] + b[None, :]           # (3, 1) + (1, 2) → (3, 2)
# [[11 21]
#  [12 22]
#  [13 23]]
```

**Pairwise squared distances** between n points `A` (n, d) and m points `B` (m, d):

```python
diff = A[:, None, :] - B[None, :, :]       # (n, m, d)
d2 = (diff ** 2).sum(axis=-1)              # (n, m)
```

That uses n·m·d memory. A faster, leaner identity: ‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b:

```python
d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
```

The `A @ B.T` part is a matrix multiplication, the most heavily optimized operation in computing.

## 4. Vectorizing common patterns

| Loop pattern | Vectorized |
|---|---|
| `if` inside a loop | `np.where(cond, a, b)` or a boolean mask |
| running total | `np.cumsum` |
| counting categories | `np.bincount`, `np.unique(..., return_counts=True)` |
| "for each row, pick column i" | `X[np.arange(n), idx]` |
| scattered additions | `np.add.at(target, indices, values)` |
| sliding windows | cumulative sums, or `np.lib.stride_tricks.sliding_window_view` |

## 5. A stable softmax

Softmax turns scores into probabilities: `exp(z) / sum(exp(z))`. But `exp(1000)` overflows to `inf`. Since softmax doesn't change if you subtract the same number from every score, subtract the max first:

```python
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

You'll meet this "subtract the max" trick again in m44 (log-sum-exp) and in every attention implementation.

## 6. Masking

Sequences in a batch have different lengths, so they're **padded** to the same length, and a boolean **mask** marks the real positions. Every computation must ignore the padding: averages divide by the number of real tokens, and attention sets padded scores to −∞ before the softmax. Exercise 9 practises this.

## 7. When loops are fine

Vectorize the hot inner work. A Python loop over *epochs* or over a handful of layers is fine; a Python loop over *examples* or *pixels* is not.

---

## Problem-solving habit #34: check shapes with a tiny example

Before vectorizing a big computation, write the slow loop version and test both on a tiny input (2 examples, 3 features) with `np.testing.assert_allclose`. Then delete the loop. You'll catch broadcasting mistakes immediately, while the numbers are small enough to check by hand.

## Go deeper (optional, research-level)

1. `np.einsum` expresses almost any combination of products and sums with index notation: `np.einsum("nd,md->nm", A, B)` is `A @ B.T`. Rewrite exercises 2 and 7 with einsum. Transformers' attention is often written this way.
2. Read about **memory bandwidth vs compute**: why is an elementwise operation (like ReLU) "memory-bound" while matrix multiplication is "compute-bound"? This is why GPU kernels "fuse" operations (as FlashAttention does).
3. Look up "k-nearest neighbours" and the **curse of dimensionality**: in 1,000 dimensions, how different are the nearest and farthest distances between random points? Measure it with your `pairwise_sq_distances`.

## Your turn

Open the **Exercises** tab. Every function must be loop-free (the tests check).
