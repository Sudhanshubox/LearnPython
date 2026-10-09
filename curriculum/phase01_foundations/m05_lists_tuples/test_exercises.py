import copy
import random

import pytest

import exercises
from code_checks import uses_any
from exercises import batches, dedupe, matmul, merge_sorted, moving_average, rank, rotate, second_largest, transpose


@pytest.mark.parametrize("nums, expected", [
    ([3, 1, 4, 4, 2], 3), ([5, 5], None), ([7], None), ([], None), ([-1, -5, -3], -3), ([2, 1], 1),
])
def test_second_largest(nums, expected):
    original = copy.copy(nums)
    assert second_largest(nums) == expected
    assert nums == original, "don't change the input list"


@pytest.mark.parametrize("items, expected", [
    ([3, 1, 3, 2, 1], [3, 1, 2]), ([], []), (["a", "b", "a"], ["a", "b"]), ([1, 1, 1], [1]),
])
def test_dedupe(items, expected):
    assert dedupe(items) == expected


@pytest.mark.parametrize("items, k, expected", [
    ([1, 2, 3, 4, 5], 2, [4, 5, 1, 2, 3]),
    ([1, 2, 3], -1, [2, 3, 1]),
    ([1, 2, 3], 3, [1, 2, 3]),
    ([1, 2, 3], 7, [3, 1, 2]),
    ([], 5, []),
])
def test_rotate(items, k, expected):
    original = list(items)
    assert rotate(items, k) == expected
    assert items == original, "don't change the input list"


@pytest.mark.parametrize("items, size, expected", [
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]),
    ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
    ([1, 2], 5, [[1, 2]]),
    ([], 3, []),
])
def test_batches(items, size, expected):
    assert batches(items, size) == expected


def test_transpose():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose([[1]]) == [[1]]
    assert transpose([[1, 2]]) == [[1], [2]]


def test_transpose_rows_are_independent():
    result = transpose([[1, 2], [3, 4]])
    result[0][0] = 99
    assert result[1][0] == 2


def test_matmul():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert matmul(a, b) == [[19, 22], [43, 50]]
    assert matmul([[1, 2, 3]], [[4], [5], [6]]) == [[32]]
    assert matmul([[1], [2]], [[3, 4]]) == [[3, 4], [6, 8]]
    identity = [[1, 0], [0, 1]]
    assert matmul(a, identity) == a


def test_matmul_shape_mismatch():
    assert matmul([[1, 2]], [[1, 2]]) is None


@pytest.mark.parametrize("values, window, expected", [
    ([1, 2, 3, 4, 5], 3, [2.0, 3.0, 4.0]),
    ([10, 20], 1, [10.0, 20.0]),
    ([1, 2], 3, []),
    ([2, 4, 6, 8], 4, [5.0]),
])
def test_moving_average(values, window, expected):
    assert moving_average(values, window) == pytest.approx(expected)


def test_rank():
    assert rank([("Ben", 78), ("Asha", 91), ("Chen", 91)]) == [("Asha", 91), ("Chen", 91), ("Ben", 78)]
    assert rank([]) == []


def test_merge_sorted():
    assert merge_sorted([1, 4, 9], [2, 3, 10]) == [1, 2, 3, 4, 9, 10]
    assert merge_sorted([], [1, 2]) == [1, 2]
    assert merge_sorted([1, 1], [1]) == [1, 1, 1]
    rng = random.Random(0)
    for _ in range(50):
        a = sorted(rng.randint(0, 50) for _ in range(rng.randint(0, 20)))
        b = sorted(rng.randint(0, 50) for _ in range(rng.randint(0, 20)))
        assert merge_sorted(a, b) == sorted(a + b)


def test_merge_sorted_does_not_sort():
    assert not uses_any(exercises.merge_sorted, "sorted", "sort")
