# m26 · The Python data model

**By the end you can:** make your own classes work with Python's built-in syntax (`+`, `==`, `len()`, `[]`, `for`, `in`, `@`, `sorted()`), and understand that this is how NumPy arrays and PyTorch tensors feel so natural.

**Why it matters for AI:** when you write `y = W @ x + b` with PyTorch tensors, Python calls `W.__matmul__(x)` and then `__add__`. PyTorch's `Dataset` works with any class that defines `__len__` and `__getitem__`. Knowing the data model lets you read library code and build your own tools that feel native. In this module you'll write a small `Vector` and `Matrix` class, a mini version of what NumPy does.

---

## 1. Special methods ("dunder" methods)

Python's syntax is implemented by methods with **d**ouble **under**scores. You never call them directly; Python does:

| You write | Python calls |
|---|---|
| `len(v)` | `v.__len__()` |
| `v[i]` | `v.__getitem__(i)` |
| `a + b` | `a.__add__(b)` |
| `a == b` | `a.__eq__(b)` |
| `a < b` | `a.__lt__(b)` |
| `for x in v` | `v.__iter__()` |
| `x in v` | `v.__contains__(x)` |
| `abs(v)` | `v.__abs__()` |
| `a @ b` | `a.__matmul__(b)` |
| `repr(v)` / `str(v)` | `v.__repr__()` / `v.__str__()` |
| `bool(v)`, `if v:` | `v.__bool__()` (or `__len__`) |
| `hash(v)`, dict keys | `v.__hash__()` |
| `f(x)` on an object | `f.__call__(x)` |

## 2. Representation

```python
class Vector:
    def __init__(self, *components):
        self._c = tuple(components)

    def __repr__(self):                 # for developers: unambiguous, ideally valid code
        return f"Vector{self._c}"

    def __str__(self):                  # for users (print); falls back to __repr__
        return "<" + ", ".join(map(str, self._c)) + ">"
```

## 3. Sequences: __len__ and __getitem__

```python
    def __len__(self):
        return len(self._c)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Vector(*self._c[index])   # slicing returns the same type
        return self._c[index]
```

With just these two, Python can already **iterate** (`for x in v`), check **membership** (`3 in v`) and unpack (`x, y = v`), by calling `__getitem__(0)`, `__getitem__(1)`, … until `IndexError`. Defining `__iter__` is faster and clearer, so do that too.

This is the PyTorch **Dataset** protocol: any object with `__len__` and `__getitem__` can be fed to a `DataLoader`.

## 4. Operators

```python
    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented            # "I don't know how": lets Python try other.__radd__
        if len(self) != len(other):
            raise ValueError("dimension mismatch")
        return Vector(*(a + b for a, b in zip(self, other)))

    def __mul__(self, scalar):               # v * 3
        return Vector(*(a * scalar for a in self))

    def __rmul__(self, scalar):              # 3 * v  (int doesn't know about Vector,
        return self * scalar                 #         so Python asks Vector)
```

Return **new** objects from operators: `a + b` must not change `a`. Return `NotImplemented` (not raise) for types you don't support, so Python can try the other operand's reflected method or give a clean `TypeError`.

## 5. Equality and hashing

```python
    def __eq__(self, other):
        return isinstance(other, Vector) and self._c == other._c

    def __hash__(self):
        return hash(self._c)
```

Rule: **objects that are equal must have equal hashes.** If you define `__eq__` without `__hash__`, Python makes your objects unhashable (they can't be dict keys or set members). Only make objects hashable if they're immutable; otherwise a key could change after it's stored.

## 6. Ordering

Define `__eq__` and `__lt__`, and `functools.total_ordering` fills in `<=`, `>`, `>=`. Then `sorted()`, `min()` and `max()` just work.

```python
from functools import total_ordering

@total_ordering
class Version:
    ...
    def __lt__(self, other):
        return self.parts < other.parts      # tuples compare element by element
```

## 7. Callable objects

```python
class Polynomial:
    def __init__(self, *coeffs):
        self.coeffs = coeffs
    def __call__(self, x):
        return sum(c * x ** i for i, c in enumerate(self.coeffs))

p = Polynomial(1, 0, 2)    # 1 + 2x²
p(3)                       # 19
```

This is why you can write `model(x)` in PyTorch: `nn.Module` defines `__call__`, which runs hooks and then your `forward` (you did a mini version in m25).

---

## Problem-solving habit #24: make it behave like a built-in

When designing a class, ask: "what built-in type is this most like?" A `Vector` is like a tuple of numbers, a `Dataset` like a list, a `Config` like a dict. Then implement the protocol of that type, so users can apply everything they already know.

## Go deeper (optional, research-level)

1. Read the "Data model" chapter of the Python Language Reference, section 3.3.8 ("Emulating numeric types"). When exactly does Python call `__radd__` instead of `__add__`?
2. NumPy **broadcasting** lets you add a vector to every row of a matrix. Read the broadcasting rules in the NumPy docs and extend your `Matrix.__add__` to accept a row `Vector`.
3. Your `Matrix @ Matrix` is O(n³). Look up Strassen's algorithm (≈ O(n^2.81)) and why GPUs still use the straightforward algorithm in practice.

## Your turn

Open the **Exercises** tab, solve each class, then press **Run tests**.
