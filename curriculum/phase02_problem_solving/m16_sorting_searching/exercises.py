"""m16 exercises: sorting and binary search.

Don't use sorted(), .sort(), heapq or bisect in this module: write the algorithms yourself.
Sorting functions return a NEW list and don't change the input.
"""

import random


# 1. Insertion sort. O(n²) worst case, O(n) on already-sorted input.
def insertion_sort(nums):
    raise NotImplementedError


# 2. Merge sort, recursive. O(n log n).
def merge_sort(nums):
    raise NotImplementedError


# 3. Quicksort with a RANDOM pivot and three groups (<, ==, >) so it stays fast
#    on sorted input and on lists with many duplicates.
def quicksort(nums):
    raise NotImplementedError


# 4. Count inversions: pairs (i < j) with nums[i] > nums[j]. O(n log n), by
#    adapting merge sort: while merging, when you take an item from the right half,
#    every item still waiting in the left half forms an inversion with it.
#    count_inversions([2, 4, 1, 3, 5]) -> 3   ((2,1), (4,1), (4,3))
def count_inversions(nums):
    raise NotImplementedError


# 5. First index i with nums[i] >= target in a sorted list (len(nums) if none). O(log n).
#    lower_bound([1, 2, 4, 4, 7], 4) -> 2, lower_bound([1, 2], 9) -> 2
def lower_bound(nums, target):
    raise NotImplementedError


# 6. First and last index of target in a sorted list, or (-1, -1). O(log n).
#    search_range([5, 7, 7, 8, 8, 10], 8) -> (3, 4)
def search_range(nums, target):
    raise NotImplementedError


# 7. A sorted list of DISTINCT numbers was rotated (e.g. [4, 5, 6, 7, 0, 1, 2]).
#    Return the index of target, or -1. O(log n).
def search_rotated(nums, target):
    raise NotImplementedError


# 8. Integer square root: the largest r with r * r <= n, by binary search. O(log n).
#    No ** 0.5, math.sqrt or math.isqrt. int_sqrt(17) -> 4
def int_sqrt(n):
    raise NotImplementedError


# 9. `fits(b)` is True for small batch sizes and False from some point on.
#    Return the largest b in [lo, hi] with fits(b) True (or lo - 1 if none fit).
#    Each call to fits() is expensive (imagine a GPU training step), so you must
#    call it only O(log(hi - lo)) times.
def max_batch_size(fits, lo, hi):
    raise NotImplementedError
