import itertools
import random
import time

import pytest

from exercises import climb, coin_change, count_ways, edit_distance, knapsack, lcs, lis, min_path_sum, rob, word_break

rng = random.Random(21)


def fast(func, *args, limit=2.0):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    assert elapsed < limit, f"{func.__name__} took {elapsed:.2f}s: memoize or build a table"
    return result


@pytest.mark.parametrize("n, expected", [(0, 1), (1, 1), (2, 2), (3, 4), (4, 7), (10, 274)])
def test_climb(n, expected):
    assert climb(n) == expected


def test_climb_fast():
    a, b, c = 1, 1, 2          # climb(0), climb(1), climb(2)
    for _ in range(500 - 2):
        a, b, c = b, c, a + b + c
    assert fast(climb, 500) == c


@pytest.mark.parametrize("houses, expected", [([2, 7, 9, 3, 1], 12), ([1, 2, 3, 1], 4), ([], 0), ([5], 5), ([2, 1, 1, 2], 4)])
def test_rob(houses, expected):
    assert rob(houses) == expected


def test_rob_fast():
    fast(rob, [rng.randint(0, 100) for _ in range(100_000)])


@pytest.mark.parametrize("coins, amount, expected", [
    ([1, 5, 6], 10, 2), ([2], 3, -1), ([1], 0, 0), ([1, 2, 5], 11, 3), ([186, 419, 83, 408], 6249, 20),
])
def test_coin_change(coins, amount, expected):
    assert fast(coin_change, coins, amount) == expected


@pytest.mark.parametrize("coins, amount, expected", [([1, 2, 5], 5, 4), ([2], 3, 0), ([10], 10, 1), ([1, 2, 3], 0, 1)])
def test_count_ways(coins, amount, expected):
    assert count_ways(coins, amount) == expected


def test_count_ways_fast():
    assert fast(count_ways, [1, 2, 5, 10, 20, 50, 100, 200], 200) == 73682


def brute_lis(nums):
    best = 0
    for mask in range(1 << len(nums)):
        seq = [nums[i] for i in range(len(nums)) if mask >> i & 1]
        if all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
            best = max(best, len(seq))
    return best


def test_lis():
    assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis([]) == 0
    assert lis([7, 7, 7]) == 1
    for _ in range(30):
        nums = [rng.randint(0, 20) for _ in range(rng.randint(0, 12))]
        assert lis(nums) == brute_lis(nums)


def test_lis_fast():
    fast(lis, [rng.randint(0, 10**6) for _ in range(1500)], limit=3.0)


@pytest.mark.parametrize("a, b, expected", [
    ("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("same", "same", 0), ("flaw", "lawn", 2), ("intention", "execution", 5),
])
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b) == expected


def test_edit_distance_fast():
    a = "".join(rng.choice("acgt") for _ in range(800))
    b = "".join(rng.choice("acgt") for _ in range(800))
    fast(edit_distance, a, b, limit=3.0)


def is_subsequence(s, t):
    it = iter(t)
    return all(ch in it for ch in s)


@pytest.mark.parametrize("a, b, length", [("ABCBDAB", "BDCABA", 4), ("abc", "def", 0), ("", "x", 0), ("abcde", "ace", 3)])
def test_lcs(a, b, length):
    result = lcs(a, b)
    assert isinstance(result, str)
    assert len(result) == length
    assert is_subsequence(result, a) and is_subsequence(result, b)


def test_lcs_fast():
    a = "".join(rng.choice("abcd") for _ in range(700))
    b = "".join(rng.choice("abcd") for _ in range(700))
    result = fast(lcs, a, b, limit=3.0)
    assert is_subsequence(result, a) and is_subsequence(result, b)


def brute_knapsack(items, cap):
    best = 0
    for r in range(len(items) + 1):
        for combo in itertools.combinations(items, r):
            if sum(w for w, _ in combo) <= cap:
                best = max(best, sum(v for _, v in combo))
    return best


def test_knapsack():
    assert knapsack([(1, 1), (3, 4), (4, 5), (5, 7)], 7) == 9
    assert knapsack([], 10) == 0
    assert knapsack([(5, 10)], 4) == 0
    for _ in range(20):
        items = [(rng.randint(1, 10), rng.randint(1, 30)) for _ in range(rng.randint(0, 10))]
        cap = rng.randint(0, 30)
        assert knapsack(items, cap) == brute_knapsack(items, cap)


def test_knapsack_fast():
    items = [(rng.randint(1, 100), rng.randint(1, 100)) for _ in range(200)]
    fast(knapsack, items, 2000, limit=3.0)


def test_min_path_sum():
    assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7
    assert min_path_sum([[5]]) == 5
    assert min_path_sum([[1, 2, 3]]) == 6
    big = [[rng.randint(0, 9) for _ in range(300)] for _ in range(300)]
    fast(min_path_sum, big)


@pytest.mark.parametrize("text, vocab, expected", [
    ("applepenapple", ["apple", "pen"], True),
    ("catsandog", ["cats", "dog", "sand", "and", "cat"], False),
    ("", ["a"], True),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", ["a", "aa", "aaa", "aaaa"], False),
])
def test_word_break(text, vocab, expected):
    assert fast(word_break, text, vocab) == expected
