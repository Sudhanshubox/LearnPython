# m34 · Performance and how CPython works

**By the end you can:** measure performance properly, find the real bottleneck with a profiler, apply the optimizations that matter, and explain what Python does under the hood: bytecode, the evaluation loop, reference counting, memory layout and the GIL.

**Why it matters for AI:** data pipelines that are 50× too slow leave your GPU idle and your experiments queued. Knowing *why* a Python loop is slow, and why the same operation in NumPy or PyTorch is 100× faster, is the key to writing efficient ML code. Research-level understanding means being able to reason from first principles about what the machine is doing.

---

## 1. Measure first

"Premature optimization is the root of all evil" (Donald Knuth). Most code isn't slow; the slow part is usually somewhere you didn't expect. So:

1. Make it **correct** (with tests, m31).
2. **Measure** to find out *if* and *where* it's slow.
3. Optimize that part, and measure again.

```python
import time
start = time.perf_counter()        # a high-resolution clock for measuring durations
work()
print(time.perf_counter() - start)
```

Timings are noisy (other programs, CPU frequency changes, caches). Repeat several times and take the **minimum**: the run with the least interference. That's what `timeit` does:

```bash
python -m timeit -s "data = list(range(1000))" "sum(data)"
```

## 2. Profiling: where does the time go?

```python
import cProfile
import pstats

cProfile.run("main()", "profile.out")
pstats.Stats("profile.out").sort_stats("cumulative").print_stats(10)
```

The profiler counts calls and time per function. Look at **cumulative time** (time in a function *including* what it calls) to find the expensive branch, and **total time** (excluding callees) to find the hot function itself. `py-spy` and `scalene` are excellent tools for real programs.

## 3. The optimizations that matter (in order)

1. **A better algorithm or data structure.** O(n²) → O(n) beats any micro-tuning (Phase 2). The classic: `x in list` inside a loop → use a set.
2. **Do less work.** Cache repeated results (m29), stop early, avoid recomputing in loops, stream instead of loading everything (m28).
3. **Use built-ins and libraries written in C.** `sum`, `sorted`, `"".join`, `collections.Counter`, `str` methods, and above all **NumPy vectorization** (Phase 4) run in C instead of the Python interpreter loop.
4. **Micro-optimizations** (local variables instead of globals, avoiding attribute lookups in tight loops): small gains, only in measured hot spots.
5. **Parallelism or the GPU** for what's left (m33, Phase 6).

## 4. What actually runs: bytecode

CPython compiles your source code to **bytecode**, instructions for a virtual machine, then executes them one at a time in a big loop written in C:

```python
import dis

def add(a, b):
    return a + b

dis.dis(add)
#  LOAD_FAST  a
#  LOAD_FAST  b
#  BINARY_OP  +
#  RETURN_VALUE
```

Each instruction costs dozens of machine instructions: fetching it, looking up the operands' types, dispatching to the right C function, and so on. That **interpretive overhead** is why a Python loop over a million numbers is slow, and why NumPy, which runs one C loop over raw numbers, is so much faster.

- `LOAD_FAST` (local variables) is cheaper than `LOAD_GLOBAL` (globals and built-ins, which need a dictionary lookup).
- Python 3.11+ **specializes** hot instructions at runtime (for example, `BINARY_OP` becomes a fast int-add when it keeps seeing ints). This is why recent Python versions got 25–60% faster.

## 5. Everything is an object

```python
import sys
sys.getsizeof(1)          # 28 bytes for a small int!
sys.getsizeof(1.0)        # 24 bytes
sys.getsizeof([])         # 56 bytes, before any items
```

Each Python object carries a header with a **reference count** and a **type pointer**. A list doesn't hold numbers; it holds *pointers* to number objects scattered around memory. A NumPy array of a million `float64`s is one block of 8 MB. A Python list of the same numbers is about 8 MB of pointers plus 24 MB of float objects, and the CPU cache can't help much.

**`__slots__`** removes the per-instance `__dict__`, which saves memory when you create millions of small objects:

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y
```

## 6. Memory management

- **Reference counting:** each object counts how many references point to it; at zero, it's freed immediately. `sys.getrefcount(obj)` shows the count (plus one for the call itself).
- **Cycles** (`a.other = b; b.other = a`) never reach zero on their own, so a **cyclic garbage collector** (`gc` module) runs periodically to find them.
- **Lists over-allocate:** `append` reserves extra space, so most appends are O(1) (m05).
- `tracemalloc` measures how much memory your code allocates.

## 7. The GIL, revisited

The Global Interpreter Lock (m33) exists largely because of reference counting: incrementing and decrementing counts from many threads at once would need a lock on every object. One big lock was simpler and faster for single-threaded code. C extensions like NumPy release the GIL during long computations, which is why they *can* use multiple cores.

---

## Problem-solving habit #32: estimate before you optimize

Use back-of-the-envelope numbers: Python does roughly 10⁷ simple operations per second, C about 10⁹, a GPU about 10¹². Reading 1 GB from an SSD takes about a second. Estimating "how long *should* this take?" tells you whether there's anything to gain, and where.

## Go deeper (optional, research-level)

1. Run `dis.dis` on a list comprehension and on the equivalent `for` loop with `append`. Why is the comprehension faster? Check how Python 3.12's PEP 709 changed comprehensions.
2. Read *Latency Numbers Every Programmer Should Know*. How many times slower is a main-memory reference than an L1 cache hit? Relate that to Python lists of pointers versus NumPy arrays.
3. Read the "Faster CPython" project's ideas (PEP 659: specializing adaptive interpreter). What is "quickening"?

## Your turn

Open the **Exercises** tab, solve each part, then press **Run tests**. Some tests require your version to be many times faster than a slow reference implementation.
