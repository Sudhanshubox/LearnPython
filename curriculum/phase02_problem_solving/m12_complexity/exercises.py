"""m12 exercises: complexity. Replace each `raise NotImplementedError` with your solution."""

import math

# ---------------------------------------------------------------- Part A: analyze
# For each function below (n = len(nums)), write its worst-case TIME complexity in
# the COMPLEXITY dict, using exactly one of:
#   "O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)"


def snippet_a(nums):
    return nums[0] + nums[-1]


def snippet_b(nums):
    total = 0
    for x in nums:
        for y in nums:
            total += x * y
    return total


def snippet_c(nums):
    i = len(nums)
    steps = 0
    while i > 1:
        i //= 2
        steps += 1
    return steps


def snippet_d(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False


def snippet_e(nums):
    result = []
    for x in nums:
        if x not in result:      # careful!
            result.append(x)
    return result


def snippet_f(nums):
    nums = sorted(nums)
    return nums[len(nums) // 2]


def snippet_g(nums):
    if not nums:
        return [[]]
    rest = snippet_g(nums[1:])
    return rest + [[nums[0]] + s for s in rest]


def snippet_h(nums):
    for i in range(10):
        for j in range(len(nums)):
            nums[j] += i
    return nums


COMPLEXITY = {
    "snippet_a": "",
    "snippet_b": "",
    "snippet_c": "",
    "snippet_d": "",
    "snippet_e": "",
    "snippet_f": "",
    "snippet_g": "",
    "snippet_h": "",
}


# ---------------------------------------------------------------- Part B: write efficient code

# 1. Prefix sums. Build a list `prefix` where prefix[i] is the sum of nums[:i]
#    (so it has len(nums) + 1 entries and prefix[0] == 0). O(n).
#    prefix_sums([3, 1, 4]) -> [0, 3, 4, 8]
def prefix_sums(nums):
    raise NotImplementedError


#    Then answer "sum of nums[left:right]" in O(1) using the prefix list.
#    range_sum([0, 3, 4, 8], 1, 3) -> 5   (1 + 4)
def range_sum(prefix, left, right):
    raise NotImplementedError


# 2. Maximum subarray sum: the largest sum of any non-empty run of consecutive numbers.
#    Must be O(n) (look up Kadane's algorithm after you've tried for 20 minutes).
#    max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) -> 6   (from [4, -1, 2, 1])
def max_subarray(nums):
    raise NotImplementedError


# 3. Given a list of n-1 distinct numbers taken from 1..n, find the missing one
#    in O(n) time and O(1) extra space (no sets or sorting).
#    missing_number([3, 1, 4, 5]) -> 2
def missing_number(nums):
    raise NotImplementedError


# 4. Return how many pairs (i < j) have nums[i] + nums[j] == target, in O(n).
#    count_pairs([1, 5, 7, -1, 5], 6) -> 3   ((1,5), (7,-1), (1,5))
def count_pairs(nums, target):
    raise NotImplementedError


# ---------------------------------------------------------------- Part C: measure growth

# 5. Estimate the exponent k in "work grows like n^k" from two measurements:
#    k = log(work2 / work1) / log(n2 / n1). Round to 1 decimal place.
#    growth_exponent(1000, 500_000, 2000, 2_000_000) -> 2.0
def growth_exponent(n1, work1, n2, work2):
    raise NotImplementedError
