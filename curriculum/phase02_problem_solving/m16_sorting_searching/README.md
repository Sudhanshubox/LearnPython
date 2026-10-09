# m16 · Sorting and binary search

**By the end you can:** implement and analyze the classic sorting algorithms, explain why O(n log n) is the limit for comparison sorting, and use binary search far beyond "find x in a sorted list", including searching for an *answer*.

**Why it matters for AI:** sorting is everywhere: ranking search results, top-k sampling of the next token, evaluating retrieval (precision@k), sorting sequences by length to batch them efficiently. Binary search on the answer is a real engineering tool: finding the biggest batch size that fits in GPU memory, or the smallest learning rate that diverges, in a handful of trial runs instead of hundreds.

---

## 1. Quadratic sorts: easy, slow

**Insertion sort** grows a sorted prefix: take the next item and slide it left until it's in place, like sorting playing cards in your hand.

```python
def insertion_sort(nums):
    a = list(nums)
    for i in range(1, len(a)):
        x = a[i]
        j = i - 1
        while j >= 0 and a[j] > x:
            a[j + 1] = a[j]          # shift bigger items right
            j -= 1
        a[j + 1] = x
    return a
```

O(n²) in the worst case, but O(n) on almost-sorted data, and very fast for tiny lists. (Timsort uses it for small runs.)

## 2. Merge sort: divide and conquer

1. Split the list in half.
2. Sort each half (recursively).
3. **Merge** the two sorted halves (m05 exercise 9!).

```
[5, 2, 4, 1]  →  [5, 2] [4, 1]  →  [5] [2] [4] [1]
              →  [2, 5] [1, 4]  →  [1, 2, 4, 5]
```

There are log₂ n levels of splitting, and each level does O(n) merging work: **O(n log n)** always. It's **stable** (equal items keep their order) and needs O(n) extra memory.

## 3. Quicksort: partition around a pivot

1. Pick a **pivot**.
2. Partition into items `< pivot`, `== pivot`, `> pivot`.
3. Recursively sort the `<` and `>` parts.

Average O(n log n), and very fast in practice. But a bad pivot (always the smallest item, e.g. the first item of already-sorted data) gives O(n²). Fix: choose the pivot **at random**. Keeping an `==` group handles lists with many duplicates.

## 4. Why not faster than n log n?

A comparison sort learns one bit per comparison ("is a < b?"). There are n! possible orderings to tell apart, which needs log₂(n!) ≈ n log₂ n bits. So **any** comparison sort needs Ω(n log n) comparisons in the worst case. (Counting sort and radix sort beat this by *not* comparing, when keys are small integers.)

## 5. Sorting in practice

```python
sorted(items)                               # Timsort: O(n log n), stable, adaptive
sorted(people, key=lambda p: p["age"])
sorted(words, key=len, reverse=True)
```

In real code, always use the built-in sort. Write your own only to learn, or for special cases (like counting inversions, exercise 4).

## 6. Binary search

In a sorted list, compare with the middle; discard the half that can't contain the target. log₂(1,000,000) ≈ 20 steps.

Two versions to know by heart. **Lower bound**: the first index whose value is `>= target` (where you'd insert `target` to keep the list sorted):

```python
def lower_bound(nums, target):
    lo, hi = 0, len(nums)            # answer is somewhere in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1             # answer is to the right of mid
        else:
            hi = mid                 # mid might be the answer
    return lo
```

The loop keeps an **invariant**: the answer is always within `[lo, hi]`, and the range shrinks every step. Most binary-search bugs come from violating it (`hi = mid - 1` when mid might be the answer, or `<=` vs `<`). The standard library has the same thing as `bisect.bisect_left`.

## 7. Binary search on the answer

Binary search works on anything **monotonic**, not just lists. If a yes/no question flips from yes to no exactly once as a number grows, you can binary-search for the flip point.

- "Does batch size b fit in GPU memory?" (yes, yes, …, yes, no, no): find the largest yes.
- "Is x² ≤ n?": integer square root.
- "Can we ship all packages within D days with capacity C?": find the smallest C.

Each check might be expensive (a training step, a simulation), so needing log₂(range) checks instead of `range` checks is a huge win.

---

## Problem-solving habit #15: look for monotonicity

When asked for "the minimum X such that…" or "the maximum X such that…", ask whether the condition is monotonic in X. If it is, the problem becomes "write a check function, then binary search."

## Go deeper (optional, research-level)

1. Read about Timsort (Tim Peters' `listsort.txt` in the CPython repo). What is a "run", and why is Timsort O(n) on already-sorted data?
2. Top-k sampling in LLMs needs only the k largest of ~100,000 token probabilities, not a full sort. Compare: full sort O(n log n), a heap O(n log k) (next module), quickselect O(n) average. What do GPU implementations actually do?
3. The number of inversions (exercise 4) measures how unsorted a list is. It's related to **Kendall's tau**, a correlation between two rankings, used to compare a model's ranking with a human's. Work out the formula connecting them.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Don't use `sorted()`, `.sort()` or `bisect` unless an exercise says you may.
