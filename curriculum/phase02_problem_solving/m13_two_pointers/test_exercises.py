import itertools
import random
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    dedupe_sorted,
    longest_unique,
    max_water,
    max_window_sum,
    min_subarray_len,
    pair_sum_sorted,
    reverse_in_place,
    three_sum,
)

rng = random.Random(13)


def timed(func, *args, limit=1.0):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    assert elapsed < limit, f"{func.__name__} took {elapsed:.2f}s: it should be O(n)"
    return result


@pytest.mark.parametrize("nums, target", [([1, 3, 4, 6, 9], 10), ([1, 2], 3), ([-5, -1, 0, 7], 2), ([2, 2, 3], 4)])
def test_pair_sum_sorted_found(nums, target):
    i, j = pair_sum_sorted(nums, target)
    assert i < j and nums[i] + nums[j] == target


@pytest.mark.parametrize("nums, target", [([1, 3, 4], 100), ([], 1), ([5], 10)])
def test_pair_sum_sorted_missing(nums, target):
    assert pair_sum_sorted(nums, target) is None


def test_pair_sum_sorted_constraints():
    assert not uses_any(exercises.pair_sum_sorted, "set", "dict")
    nums = list(range(0, 2_000_000, 2))
    assert timed(pair_sum_sorted, nums, -1) is None


@pytest.mark.parametrize("items", [[1, 2, 3, 4], [1, 2, 3], [], ["x"]])
def test_reverse_in_place(items):
    expected = items[::-1]
    assert reverse_in_place(items) is None
    assert items == expected


def test_reverse_in_place_constraints():
    assert not uses_any(exercises.reverse_in_place, "reverse", "reversed")


@pytest.mark.parametrize("nums, unique", [
    ([1, 1, 2, 3, 3, 3], [1, 2, 3]), ([], []), ([5], [5]), ([2, 2, 2], [2]), ([1, 2, 3], [1, 2, 3]),
])
def test_dedupe_sorted(nums, unique):
    k = dedupe_sorted(nums)
    assert k == len(unique)
    assert nums[:k] == unique


@pytest.mark.parametrize("nums, k, expected", [
    ([2, 1, 5, 1, 3, 2], 3, 9), ([1, 2, 3], 3, 6), ([-1, -2, -3], 1, -1), ([4, -1, 2, 1], 2, 3),
])
def test_max_window_sum(nums, k, expected):
    assert max_window_sum(nums, k) == expected


def test_max_window_sum_is_linear():
    nums = [rng.randint(-100, 100) for _ in range(300_000)]
    timed(max_window_sum, nums, 50_000)


@pytest.mark.parametrize("s, expected", [
    ("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0), ("abba", 2), ("dvdf", 3), ("abcdef", 6),
])
def test_longest_unique(s, expected):
    assert longest_unique(s) == expected


def test_longest_unique_is_linear():
    s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(300_000))
    timed(longest_unique, s)


@pytest.mark.parametrize("nums, target, expected", [
    ([2, 3, 1, 2, 4, 3], 7, 2), ([1, 4, 4], 4, 1), ([1, 1, 1, 1], 11, 0), ([], 3, 0), ([1, 2, 3, 4, 5], 15, 5),
])
def test_min_subarray_len(nums, target, expected):
    assert min_subarray_len(nums, target) == expected


def test_min_subarray_len_is_linear():
    nums = [rng.randint(1, 10) for _ in range(300_000)]
    timed(min_subarray_len, nums, 10**9)


@pytest.mark.parametrize("heights, expected", [([1, 8, 6, 2, 5, 4, 8, 3, 7], 49), ([1, 1], 1), ([4, 3, 2, 1, 4], 16), ([1, 2, 1], 2)])
def test_max_water(heights, expected):
    assert max_water(heights) == expected


def test_max_water_matches_brute_force():
    for _ in range(50):
        h = [rng.randint(0, 20) for _ in range(rng.randint(2, 30))]
        brute = max((j - i) * min(h[i], h[j]) for i, j in itertools.combinations(range(len(h)), 2))
        assert max_water(h) == brute


def test_max_water_is_linear():
    timed(max_water, [rng.randint(0, 10_000) for _ in range(300_000)])


def test_three_sum():
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([1, 2, 3]) == []


def test_three_sum_matches_brute_force():
    for _ in range(30):
        nums = [rng.randint(-6, 6) for _ in range(rng.randint(0, 15))]
        brute = sorted({tuple(sorted(c)) for c in itertools.combinations(nums, 3) if sum(c) == 0})
        assert three_sum(nums) == [list(t) for t in brute]


def test_three_sum_is_quadratic():
    nums = [rng.randint(-10**6, 10**6) for _ in range(1500)]
    timed(three_sum, nums, limit=3.0)
