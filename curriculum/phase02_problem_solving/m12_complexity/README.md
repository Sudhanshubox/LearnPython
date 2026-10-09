# m12 · Complexity and a problem-solving framework

**By the end you can:** predict how an algorithm's running time grows with its input, describe it in Big-O notation, measure it experimentally, and attack an unfamiliar problem with a repeatable process.

**Why it matters for AI:** "attention is O(n²) in sequence length" is *the* reason LLMs have context limits, and why a whole research field works on cheaper attention. Scaling laws, the cost of training runs and the speed of data pipelines are all complexity questions. And every technical interview tests this.

---

## 1. Counting steps, not seconds

Seconds depend on your laptop. What we want is how the work **grows** as the input grows.

```python
def total(nums):                 # n = len(nums)
    s = 0                        # 1 step
    for x in nums:               # n times:
        s += x                   #   1 step
    return s                     # 1 step
```

About `n + 2` steps. Double the input, double the work: **linear**.

```python
def has_duplicate(nums):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False
```

About `n²/2` comparisons in the worst case. Double the input, **4×** the work: **quadratic**.

## 2. Big-O: keep the part that dominates

Big-O describes the growth rate for large n, ignoring constant factors and smaller terms:

- `3n + 5` → **O(n)**
- `n²/2 + 100n` → **O(n²)**
- `500` → **O(1)**

| Big-O | Name | n = 1,000,000 needs about | Typical example |
|---|---|---|---|
| O(1) | constant | 1 step | dict lookup, `lst[i]` |
| O(log n) | logarithmic | 20 steps | binary search |
| O(n) | linear | 10⁶ | one loop over the data |
| O(n log n) | linearithmic | 2×10⁷ | sorting |
| O(n²) | quadratic | 10¹² (hours) | all pairs |
| O(2ⁿ) | exponential | never finishes | all subsets |

**Rule of thumb:** Python does roughly 10⁷ simple operations per second. Compare that with the table to know if your idea is fast enough *before* you write it.

## 3. Reading complexity from code

- Sequential blocks **add**: O(n) then O(n²) is O(n²).
- Nested loops **multiply**: a loop of n around a loop of m is O(n·m).
- A loop that **halves** the problem each time is O(log n).
- Watch for hidden loops: `x in list`, `list.index`, `list.insert(0, …)`, slicing `lst[a:b]` and `sorted()` all do work proportional to their size.

```python
for x in a:          # n times
    if x in b:       # hidden O(m) loop!   -> O(n·m) overall
        ...

b_set = set(b)       # O(m) once
for x in a:
    if x in b_set:   # O(1)                -> O(n + m) overall
        ...
```

## 4. Space complexity

Memory counts too. `has_duplicate` above uses O(1) extra space; the set version uses O(n) extra space but O(n) time. Trading memory for speed is the most common optimization there is.

## 5. Best, average, worst case

Searching an unsorted list is O(1) if the item is first, O(n) if it's last or missing. Unless told otherwise, Big-O means the **worst case**. Dict lookups are O(1) on *average* (a bad hash function could make them O(n)).

## 6. Measuring growth experimentally

You can check your analysis: count operations (or time) at several sizes, then look at the ratio. If doubling n multiplies the work by 2^k, the algorithm is about O(n^k). On a log-log plot, the slope of the line *is* the exponent:

```
k ≈ log(work₂ / work₁) / log(n₂ / n₁)
```

This is exactly how researchers find **scaling laws**: plot loss against compute on log-log axes and read off the slope.

## 7. A framework for any problem

Use these steps every time. They feel slow at first and quickly become automatic.

1. **Understand.** Restate the problem in your own words. What are the inputs, the output, the constraints (how big is n)?
2. **Examples.** Work 2–3 small examples by hand, including edge cases (empty, one item, duplicates, negatives).
3. **Brute force.** What's the most obvious correct solution? What's its complexity? Is it already fast enough for the constraints?
4. **Optimize.** Where is repeated work? Can a set/dict, sorting, two pointers or a running total remove it?
5. **Code** it cleanly.
6. **Test** with your examples and edge cases.
7. **Analyze** the final time and space complexity.

---

## Problem-solving habit #11: let the constraints pick the algorithm

If n ≤ 20, exponential is fine. If n ≤ 10⁴, O(n²) is fine. If n ≤ 10⁶, you need O(n log n) or better. Competitive programmers read the constraints *first* and work backwards to the algorithm.

## Go deeper (optional, research-level)

1. Self-attention compares every token with every other token: O(n²) time and memory in sequence length n. Read the abstract of *Efficient Transformers: A Survey* (Tay et al., 2020) and list three ways people reduce that cost.
2. Read about the "Master Theorem" for recursive algorithms like merge sort: T(n) = 2T(n/2) + O(n). Why does it give O(n log n)?
3. *Scaling Laws for Neural Language Models* (Kaplan et al., 2020) found loss follows a power law in model size, data and compute. Look at Figure 1. Which slope did they measure for compute, and what does it mean in plain words?

## Your turn

Open the **Exercises** tab. Part A asks you to analyze code; part B to write efficient algorithms; part C to measure growth.
