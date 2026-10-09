import hashlib
import random
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    COMPLEXITY,
    count_pairs,
    growth_exponent,
    max_subarray,
    missing_number,
    prefix_sums,
    range_sum,
)

# Answers are stored as hashes so they don't spoil the exercise if you open this file.
ANSWERS = {
    "snippet_a": "a841c7a47a786b5f",
    "snippet_b": "9aece5ba4ad9f076",
    "snippet_c": "a2b37f420dc33e15",
    "snippet_d": "9ac0b60fd592ecaa",
    "snippet_e": "952aaa1da85c92b5",
    "snippet_f": "2009a9835600ee23",
    "snippet_g": "b24297c8e3983931",
    "snippet_h": "c7c6139e5018d427",
}


def fingerprint(name, answer):
    normalized = answer.replace(" ", "").replace("**", "^").lower()
    return hashlib.sha256(f"{name}:{normalized}".encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_complexity_answers(name):
    given = COMPLEXITY[name]
    assert fingerprint(name, given) == ANSWERS[name], (
        f"{name}: {given!r} isn't right yet. Ask yourself: if n doubles, how much more work does it do?"
    )


def test_prefix_sums_and_range_sum():
    prefix = prefix_sums([3, 1, 4])
    assert prefix == [0, 3, 4, 8]
    assert range_sum(prefix, 1, 3) == 5
    assert range_sum(prefix, 0, 3) == 8
    assert range_sum(prefix, 2, 2) == 0
    assert prefix_sums([]) == [0]


def test_range_sum_random():
    rng = random.Random(3)
    nums = [rng.randint(-50, 50) for _ in range(200)]
    prefix = prefix_sums(nums)
    for _ in range(200):
        a = rng.randint(0, 200)
        b = rng.randint(a, 200)
        assert range_sum(prefix, a, b) == sum(nums[a:b])


@pytest.mark.parametrize("nums, expected", [
    ([-2, 1, -3, 4, -1, 2, 1, -5, 4], 6),
    ([1], 1),
    ([-3, -1, -2], -1),
    ([5, 4, -1, 7, 8], 23),
])
def test_max_subarray(nums, expected):
    assert max_subarray(nums) == expected


def test_max_subarray_is_linear():
    nums = [random.Random(5).randint(-100, 100) for _ in range(200_000)]
    start = time.perf_counter()
    max_subarray(nums)
    assert time.perf_counter() - start < 1.0, "needs to be O(n)"


@pytest.mark.parametrize("nums, expected", [([3, 1, 4, 5], 2), ([1], 2), ([2], 1), ([], 1), (list(range(1, 1000)), 1000)])
def test_missing_number(nums, expected):
    assert missing_number(nums) == expected


def test_missing_number_constant_space():
    assert not uses_any(exercises.missing_number, "set", "sorted", "sort", "dict", "Counter")


@pytest.mark.parametrize("nums, target, expected", [
    ([1, 5, 7, -1, 5], 6, 3),
    ([1, 1, 1, 1], 2, 6),
    ([], 5, 0),
    ([3], 6, 0),
    ([2, 4, 3, 3], 6, 2),
])
def test_count_pairs(nums, target, expected):
    assert count_pairs(nums, target) == expected


def test_count_pairs_is_linear():
    nums = [random.Random(7).randint(0, 1000) for _ in range(100_000)]
    start = time.perf_counter()
    count_pairs(nums, 1000)
    assert time.perf_counter() - start < 1.0, "needs to be O(n): remember counts in a dict"


@pytest.mark.parametrize("args, expected", [
    ((1000, 500_000, 2000, 2_000_000), 2.0),
    ((10, 10, 1000, 1000), 1.0),
    ((100, 7, 10_000, 7), 0.0),
    ((100, 1_000_000, 200, 8_000_000), 3.0),
])
def test_growth_exponent(args, expected):
    assert growth_exponent(*args) == expected
