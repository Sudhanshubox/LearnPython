"""m10 exercises: debugging.

Every function below has EXACTLY ONE bug. Run the tests, read the failures,
find each bug, fix it with the smallest change you can, and add a comment
explaining what was wrong. Don't rewrite the functions from scratch.
"""


# 1. Return the average of a non-empty list of numbers.
def average(nums):
    return sum(nums) // len(nums)


# 2. Return the last n items of a list (all of them if n >= len(items)).
def last_n(items, n):
    return items[len(items) - n + 1:]


# 3. Count how many times each word appears.
def count_words(words):
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 1) + 1
    return counts


# 4. Return True if the list is sorted in non-decreasing order.
def is_sorted(nums):
    for i in range(len(nums)):
        if nums[i] > nums[i + 1]:
            return False
    return True


# 5. Remove all negative numbers from the list IN PLACE and return it.
def remove_negatives(nums):
    for n in nums:
        if n < 0:
            nums.remove(n)
    return nums


# 6. Scale values to the range 0..1: (v - min) / (max - min).
#    If all values are equal, return a list of 0.0s.
def normalize(values):
    lo, hi = min(values), max(values)
    if lo == hi:
        return [0.0] * len(values)
    return [(v - lo) / hi - lo for v in values]


# 7. Return a NEW dict with base's settings overridden by override's.
#    `base` must not be changed.
def merge_settings(base, override):
    result = base
    for key, value in override.items():
        result[key] = value
    return result


# 8. Return a list of functions where the i-th function multiplies its input by i.
#    make_multipliers(3)[2](10) should be 20.
def make_multipliers(n):
    return [lambda x: x * i for i in range(n)]


# 9. Binary search: return the index of target in the sorted list, or -1.
def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# 10. Return the smallest number in a non-empty list (without using min()).
def smallest(nums):
    best = 0
    for n in nums:
        if n < best:
            best = n
    return best
