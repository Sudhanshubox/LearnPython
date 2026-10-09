import math
import random
import time

import pytest

import exercises
from code_checks import is_recursive, uses_any, uses_power_half
from exercises import (
    count_inversions,
    insertion_sort,
    int_sqrt,
    lower_bound,
    max_batch_size,
    merge_sort,
    quicksort,
    search_range,
    search_rotated,
)

rng = random.Random(16)
SORTS = [insertion_sort, merge_sort, quicksort]
FORBIDDEN = ("sorted", "sort", "heapq", "bisect", "bisect_left", "bisect_right")


@pytest.mark.parametrize("name", [
    "insertion_sort", "merge_sort", "quicksort", "count_inversions", "lower_bound",
    "search_range", "search_rotated", "int_sqrt", "max_batch_size",
])
def test_no_library_shortcuts(name):
    func = getattr(exercises, name)
    assert not uses_any(func, *FORBIDDEN, "sqrt", "isqrt"), "write the algorithm yourself"


@pytest.mark.parametrize("sort", SORTS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("nums", [[], [1], [2, 1], [3, 1, 2], [5, 5, 5], [-1, 10, -20, 0], [4, 2, 4, 1, 2]])
def test_sorts_small(sort, nums):
    original = list(nums)
    assert sort(nums) == sorted(original)
    assert nums == original, "return a new list"


@pytest.mark.parametrize("sort", SORTS, ids=lambda f: f.__name__)
def test_sorts_random(sort):
    for _ in range(30):
        nums = [rng.randint(-50, 50) for _ in range(rng.randint(0, 60))]
        assert sort(nums) == sorted(nums)


def test_merge_sort_recursive_and_fast():
    assert is_recursive(exercises.merge_sort)
    nums = [rng.random() for _ in range(100_000)]
    start = time.perf_counter()
    assert merge_sort(nums) == sorted(nums)
    assert time.perf_counter() - start < 3.0


@pytest.mark.parametrize("case", ["random", "sorted", "duplicates"])
def test_quicksort_fast_on_hard_inputs(case):
    nums = {
        "random": [rng.random() for _ in range(100_000)],
        "sorted": list(range(100_000)),
        "duplicates": [rng.randint(0, 3) for _ in range(100_000)],
    }[case]
    start = time.perf_counter()
    assert quicksort(nums) == sorted(nums)
    assert time.perf_counter() - start < 3.0, "random pivot + three groups"


def test_insertion_sort_fast_on_sorted_input():
    nums = list(range(200_000))
    start = time.perf_counter()
    insertion_sort(nums)
    assert time.perf_counter() - start < 1.0, "on sorted input the inner loop should stop immediately"


def brute_inversions(nums):
    return sum(1 for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] > nums[j])


def test_count_inversions():
    assert count_inversions([2, 4, 1, 3, 5]) == 3
    assert count_inversions([]) == 0
    assert count_inversions([5, 4, 3, 2, 1]) == 10
    assert count_inversions([1, 1, 1]) == 0
    for _ in range(30):
        nums = [rng.randint(0, 20) for _ in range(rng.randint(0, 40))]
        assert count_inversions(nums) == brute_inversions(nums)


def test_count_inversions_fast():
    nums = list(range(60_000, 0, -1))
    start = time.perf_counter()
    assert count_inversions(nums) == 60_000 * 59_999 // 2
    assert time.perf_counter() - start < 2.0, "needs O(n log n)"


def test_lower_bound():
    nums = [1, 2, 4, 4, 7]
    assert [lower_bound(nums, t) for t in [0, 1, 3, 4, 5, 7, 8]] == [0, 0, 2, 2, 4, 4, 5]
    assert lower_bound([], 3) == 0
    for _ in range(100):
        data = sorted(rng.randint(0, 30) for _ in range(rng.randint(0, 20)))
        t = rng.randint(-1, 31)
        assert lower_bound(data, t) == next((i for i, x in enumerate(data) if x >= t), len(data))


@pytest.mark.parametrize("nums, target, expected", [
    ([5, 7, 7, 8, 8, 10], 8, (3, 4)), ([5, 7, 7, 8, 8, 10], 6, (-1, -1)), ([], 0, (-1, -1)),
    ([2, 2, 2], 2, (0, 2)), ([1], 1, (0, 0)),
])
def test_search_range(nums, target, expected):
    assert search_range(nums, target) == expected


def test_search_rotated():
    nums = [4, 5, 6, 7, 0, 1, 2]
    for i, x in enumerate(nums):
        assert search_rotated(nums, x) == i
    assert search_rotated(nums, 3) == -1
    assert search_rotated([], 1) == -1
    for _ in range(50):
        base = sorted(rng.sample(range(100), rng.randint(1, 30)))
        k = rng.randrange(len(base))
        rotated = base[k:] + base[:k]
        t = rng.choice(rotated)
        assert rotated[search_rotated(rotated, t)] == t


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 15, 16, 17, 10**12, 10**30 + 7])
def test_int_sqrt(n):
    r = int_sqrt(n)
    assert r * r <= n < (r + 1) * (r + 1)


def test_int_sqrt_no_float_tricks():
    assert not uses_power_half(exercises.int_sqrt)


def test_max_batch_size():
    for limit in [1, 2, 37, 512, 999, 1000]:
        calls = []

        def fits(b):
            calls.append(b)
            return b <= limit

        assert max_batch_size(fits, 1, 1000) == limit
        assert len(calls) <= math.ceil(math.log2(1000)) + 2, f"called fits() {len(calls)} times"


def test_max_batch_size_none_fit():
    assert max_batch_size(lambda b: False, 8, 64) == 7
