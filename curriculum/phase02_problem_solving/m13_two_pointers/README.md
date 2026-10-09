# m13 · Two pointers and sliding windows

**By the end you can:** recognize problems that two pointers or a sliding window can solve, and turn O(n²) "check every pair / every subarray" solutions into O(n).

**Why it matters for AI:** sequence models live on windows. An LLM's context window, a 1-D convolution, n-gram features and rolling statistics over time series are all sliding windows. The patterns here are also some of the most frequently asked in coding interviews.

---

## 1. Two pointers from opposite ends

In a **sorted** list, to find two numbers that add up to a target:

```python
def pair_sum(nums, target):          # nums is sorted
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return left, right
        if s < target:
            left += 1                # need a bigger sum: move the small end up
        else:
            right -= 1               # need a smaller sum: move the big end down
    return None
```

**Why it's correct:** if `nums[left] + nums[right]` is too small, then `nums[left]` plus *anything* to the left of `right` is also too small, so `left` can never be part of a solution and we can discard it. Each step discards one candidate, so it takes at most n steps: **O(n)** instead of O(n²).

The same shape reverses a list in place, checks palindromes, and solves "container with most water".

## 2. Two pointers in the same direction (read/write)

A slow pointer marks where to *write*; a fast pointer *reads* every item:

```python
def remove_zeros(nums):              # in place, keeps order
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write] = nums[read]
            write += 1
    return write                     # nums[:write] is the answer
```

O(n) time, O(1) extra space. Useful for filtering, deduplicating sorted data and partitioning.

## 3. Fixed-size sliding window

Maximum sum of any k consecutive numbers. Don't re-add each window; **slide** it: add the item entering, subtract the item leaving.

```python
def max_window(nums, k):
    window = sum(nums[:k])
    best = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i - k]
        best = max(best, window)
    return best
```

O(n) instead of O(n·k). You did this already in m05's moving average.

## 4. Variable-size sliding window

"Longest substring without repeating characters." Grow the window on the right; when it breaks the rule, shrink it from the left until it's valid again:

```python
def longest_unique(s):
    last_seen = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1     # jump past the previous copy
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

Each index enters and leaves the window at most once, so the whole thing is O(n) even though there's a loop inside a loop in some versions.

**Template:**

```
for right in range(n):
    add item[right] to the window
    while the window is invalid:
        remove item[left] from the window; left += 1
    update the answer with the current window
```

## 5. Sort first, then two pointers

Many "find pairs/triples" problems become two-pointer problems after sorting (O(n log n)). **3-sum** (find all triples that sum to 0): fix the first number, then run the opposite-ends search on the rest. O(n²) instead of O(n³).

## 6. How to recognize these problems

| Clue in the problem | Try |
|---|---|
| sorted array, find a pair | opposite-ends two pointers |
| in place, O(1) extra space | read/write pointers |
| "contiguous subarray/substring" of size k | fixed window |
| "longest/shortest subarray such that…" | variable window |
| triples or more, order doesn't matter | sort, then two pointers |

---

## Problem-solving habit #12: prove why you can discard

Two-pointer and window solutions are fast because each step *throws away* candidates without checking them. Always ask: "why can I safely discard this?" If you can't answer, the algorithm is probably wrong. If you can, you've also written its correctness proof.

## Go deeper (optional, research-level)

1. Sliding-window attention (used in Longformer and Mistral) lets each token attend only to the previous w tokens. Its cost is O(n·w) instead of O(n²). What information can't flow between distant tokens in one layer, and how do stacked layers fix this?
2. The "sliding window maximum" problem (max of every window of size k) can be solved in O(n) with a deque. Look it up after attempting it, and explain why each element is pushed and popped at most once.
3. Prove that the variable-window algorithm for `min_subarray_len` (exercise 6) is correct only because all numbers are positive. Construct a counterexample with a negative number.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
