# m36 · NumPy fundamentals

**By the end you can:** create and inspect arrays, understand shape and dtype, index and slice in any number of dimensions, filter with boolean masks, reshape and transpose, and know when an operation gives you a **view** versus a **copy**.

**Why it matters for AI:** every number in machine learning lives in an array: a batch of images is a 4-D array of shape `(batch, channels, height, width)`, a sentence is an array of token IDs, a model's weights are matrices. PyTorch tensors deliberately copy NumPy's interface, so everything here transfers directly. If you're fluent with shapes, half of all deep-learning bugs disappear.

---

## 1. Why NumPy?

A Python list of numbers is a list of pointers to separate number objects (m34). A NumPy **array** is one contiguous block of raw numbers of a single type, processed by fast C loops:

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
a * 2              # array([2., 4., 6.])   no Python loop
a.sum()            # 6.0
```

Operations on whole arrays are typically 10–100× faster than the equivalent Python loop.

## 2. Creating arrays

```python
np.array([[1, 2], [3, 4]])         # from nested lists
np.zeros((2, 3))                   # shape (2, 3), all 0.0
np.ones(5), np.full((2, 2), 7)
np.eye(3)                          # identity matrix
np.arange(0, 10, 2)                # [0 2 4 6 8]   like range
np.linspace(0, 1, 5)               # [0. 0.25 0.5 0.75 1.]   5 evenly spaced points

rng = np.random.default_rng(seed=42)   # the modern way to make random numbers
rng.normal(0, 1, size=(2, 3))          # Gaussian samples
rng.integers(0, 10, size=5)
```

Always create a `Generator` with a seed for reproducible experiments.

## 3. Shape, dtype and friends

```python
x = np.zeros((32, 3, 28, 28), dtype=np.float32)   # a batch of 32 RGB 28×28 images
x.shape      # (32, 3, 28, 28)
x.ndim       # 4
x.size       # 75264 numbers
x.dtype      # float32
x.nbytes     # 301056 bytes (4 bytes per float32)
```

**dtype** matters in ML: `float32` is the default for training (half the memory of `float64`), `float16`/`bfloat16` for even cheaper training (m01's go-deeper!), `int64` for token IDs and labels, `uint8` for raw image pixels (0–255). `x.astype(np.float32)` converts (and copies).

## 4. Indexing and slicing

```python
m = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
m[1, 2]          # 6        row 1, column 2
m[0]             # [0 1 2 3]   the first row
m[:, 1]          # [1 5 9]     the second column
m[1:, ::2]       # rows 1.., every other column: [[4 6] [8 10]]
m[-1, -1]        # 11
m[[0, 2], [1, 3]]    # "fancy" indexing: elements (0,1) and (2,3) → [1 11]
```

In n dimensions, give one index or slice per axis, separated by commas. `...` (ellipsis) means "all the remaining axes": `x[..., 0]` takes the first item of the last axis.

## 5. Boolean masks

```python
scores = np.array([55, 91, 78, 30])
mask = scores >= 60          # [False  True  True False]
scores[mask]                 # [91 78]
scores[scores < 60] = 0      # assign through a mask
(scores > 50).sum()          # count: True counts as 1
np.where(scores > 60, "pass", "fail")
```

Combine conditions with `&`, `|`, `~` and **parentheses**: `(a > 0) & (a < 10)`. (Python's `and`/`or` don't work on arrays: remember m03's go-deeper question!)

## 6. Axis: which direction to reduce

```python
m.sum()           # everything: 66
m.sum(axis=0)     # down the rows → one value per column: [12 15 18 21]
m.sum(axis=1)     # across the columns → one value per row: [ 6 22 38]
m.mean(axis=1, keepdims=True)   # shape (3, 1) instead of (3,): handy for broadcasting (m37)
```

**Mnemonic:** `axis=k` is the axis that *disappears*.

## 7. Reshaping and transposing

```python
a = np.arange(6)
a.reshape(2, 3)          # same data, new shape
a.reshape(-1, 2)         # -1 means "work it out": shape (3, 2)
m.T                      # transpose (2-D)
x.transpose(0, 2, 3, 1)  # reorder axes: (N, C, H, W) → (N, H, W, C)
x.reshape(len(x), -1)    # flatten each image into one row
a[:, np.newaxis]         # add an axis: shape (6,) → (6, 1)
```

PyTorch uses "channels first" `(N, C, H, W)`; many image libraries use "channels last" `(N, H, W, C)`. Converting between them is a transpose.

## 8. Views vs copies

Slicing and reshaping usually return a **view**: a new array object looking at the *same* memory. Changing the view changes the original!

```python
a = np.arange(5)
v = a[1:3]       # a view
v[0] = 99
a                # [ 0 99  2  3  4]   ← the original changed

c = a[1:3].copy()     # an independent copy
np.shares_memory(a, v)   # True
```

Fancy indexing and boolean masks return **copies**. When in doubt, check with `np.shares_memory`, and call `.copy()` if you'll modify the result. (Remember aliasing in m01 and m05: same idea, new place.)

---

## Problem-solving habit #33: write the shapes down

When working with arrays, write the shape of every variable as a comment: `# x: (N, 784)`, `# W: (784, 10)`, `# logits: (N, 10)`. Most shape bugs become obvious on paper, and the comments double as documentation for your future self.

## Go deeper (optional, research-level)

1. Arrays have **strides**: how many bytes to step in memory to move one position along each axis. Print `np.arange(12).reshape(3, 4).strides` and `.T.strides`. How can a transpose be free (no copying)?
2. Why is `float16` risky for training while `bfloat16` is usually fine? Compare their exponent and mantissa bits.
3. Read the NumPy paper, *Array programming with NumPy* (Harris et al., Nature 2020). Which design ideas did PyTorch and JAX copy?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Several exercises forbid Python loops and comprehensions: use array operations.
