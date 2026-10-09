import random

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    average,
    binary_search,
    count_words,
    is_sorted,
    last_n,
    make_multipliers,
    merge_settings,
    normalize,
    remove_negatives,
    smallest,
)


def test_average():
    assert average([1, 2]) == 1.5
    assert average([4]) == 4


@pytest.mark.parametrize("items, n, expected", [
    ([1, 2, 3, 4, 5], 2, [4, 5]),
    ([1, 2, 3], 1, [3]),
    ([1, 2, 3], 3, [1, 2, 3]),
    ([1, 2, 3], 10, [1, 2, 3]),
])
def test_last_n(items, n, expected):
    assert last_n(items, n) == expected


def test_count_words():
    assert count_words(["a", "b", "a"]) == {"a": 2, "b": 1}
    assert count_words([]) == {}


@pytest.mark.parametrize("nums, expected", [
    ([1, 2, 2, 3], True), ([3, 1], False), ([], True), ([5], True), ([1, 3, 2], False),
])
def test_is_sorted(nums, expected):
    assert is_sorted(nums) == expected


@pytest.mark.parametrize("nums, expected", [
    ([1, -2, 3], [1, 3]),
    ([-1, -2, 3, -4, -5], [3]),
    ([-1, -1, -1], []),
    ([1, 2], [1, 2]),
])
def test_remove_negatives(nums, expected):
    result = remove_negatives(nums)
    assert result == expected
    assert result is nums, "change the list in place"


def test_normalize():
    assert normalize([2, 4, 6]) == pytest.approx([0.0, 0.5, 1.0])
    assert normalize([10, 20]) == pytest.approx([0.0, 1.0])
    assert normalize([3, 3]) == [0.0, 0.0]


def test_merge_settings():
    base = {"lr": 0.1, "epochs": 5}
    merged = merge_settings(base, {"lr": 0.01})
    assert merged == {"lr": 0.01, "epochs": 5}
    assert base == {"lr": 0.1, "epochs": 5}, "base must not change"


def test_make_multipliers():
    fns = make_multipliers(4)
    assert [f(10) for f in fns] == [0, 10, 20, 30]


def test_binary_search():
    items = [1, 3, 5, 7, 9, 11]
    for i, x in enumerate(items):
        assert binary_search(items, x) == i
    assert binary_search(items, 4) == -1
    assert binary_search([], 1) == -1
    assert binary_search([2], 2) == 0
    rng = random.Random(0)
    for _ in range(100):
        data = sorted(rng.sample(range(100), rng.randint(1, 30)))
        target = rng.choice(data)
        assert data[binary_search(data, target)] == target


def test_smallest():
    assert smallest([3, 1, 2]) == 1
    assert smallest([5, 9, 7]) == 5
    assert smallest([-2, -8]) == -8
    assert not uses_any(exercises.smallest, "min", "sorted")
