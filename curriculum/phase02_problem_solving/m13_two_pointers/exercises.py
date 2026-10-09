"""m13 exercises: two pointers and sliding windows.

Every function must run in O(n) time unless it says otherwise.
"""


# 1. In a SORTED list, return indices (i, j) with i < j and nums[i] + nums[j] == target,
#    or None. O(n) time, O(1) extra space (no sets or dicts).
#    pair_sum_sorted([1, 3, 4, 6, 9], 10) -> (1, 4)  or (2, 3): either is fine
def pair_sum_sorted(nums, target):
    raise NotImplementedError


# 2. Reverse a list IN PLACE using two pointers and swaps. Don't use slicing,
#    reverse(), or reversed(). Return None.
def reverse_in_place(items):
    raise NotImplementedError


# 3. A SORTED list may contain duplicates. Remove them IN PLACE so the first k
#    items are the unique values in order, and return k.
#    nums = [1, 1, 2, 3, 3, 3]; k = dedupe_sorted(nums) -> 3, nums[:3] == [1, 2, 3]
def dedupe_sorted(nums):
    raise NotImplementedError


# 4. Maximum sum of k consecutive numbers (1 <= k <= len(nums)).
#    max_window_sum([2, 1, 5, 1, 3, 2], 3) -> 9   (5 + 1 + 3)
def max_window_sum(nums, k):
    raise NotImplementedError


# 5. Length of the longest substring with no repeated characters.
#    longest_unique("abcabcbb") -> 3 ("abc"), longest_unique("pwwkew") -> 3 ("wke")
def longest_unique(s):
    raise NotImplementedError


# 6. Given POSITIVE numbers, the length of the shortest contiguous run whose sum is
#    >= target, or 0 if there isn't one.
#    min_subarray_len([2, 3, 1, 2, 4, 3], 7) -> 2   ([4, 3])
def min_subarray_len(nums, target):
    raise NotImplementedError


# 7. Container with most water: heights[i] is a vertical line at x = i. Pick two
#    lines; the water between them is (j - i) * min(heights[i], heights[j]).
#    Return the maximum. O(n) with two pointers (think: which pointer should move?).
#    max_water([1, 8, 6, 2, 5, 4, 8, 3, 7]) -> 49
def max_water(heights):
    raise NotImplementedError


# 8. 3-sum: all unique triples [a, b, c] (a <= b <= c) from nums with a + b + c == 0,
#    as a sorted list. O(n²): sort, fix one number, two-pointer the rest.
#    three_sum([-1, 0, 1, 2, -1, -4]) -> [[-1, -1, 2], [-1, 0, 1]]
def three_sum(nums):
    raise NotImplementedError
