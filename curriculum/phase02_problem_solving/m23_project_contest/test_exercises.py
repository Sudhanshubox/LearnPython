import itertools
import random
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    KthLargest,
    can_finish,
    can_partition,
    largest_rectangle,
    longest_consecutive,
    min_window,
    network_delay,
    num_decodings,
    ship_capacity,
    trap,
)

rng = random.Random(23)


def fast(func, *args, limit=2.0):
    start = time.perf_counter()
    result = func(*args)
    elapsed = time.perf_counter() - start
    assert elapsed < limit, f"{func.__name__} took {elapsed:.2f}s on a large input"
    return result


# 1 ------------------------------------------------------------------------

@pytest.mark.parametrize("nums, expected", [
    ([100, 4, 200, 1, 3, 2], 4), ([], 0), ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9), ([1, 2, 0, 1], 3), ([-1, -2, 5], 2),
])
def test_longest_consecutive(nums, expected):
    assert longest_consecutive(nums) == expected


def test_longest_consecutive_fast_and_unsorted():
    assert not uses_any(exercises.longest_consecutive, "sorted", "sort")
    nums = list(range(300_000))
    rng.shuffle(nums)
    assert fast(longest_consecutive, nums) == 300_000


# 2 ------------------------------------------------------------------------

def brute_trap(h):
    return sum(max(0, min(max(h[: i + 1]), max(h[i:])) - h[i]) for i in range(len(h)))


def test_trap():
    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap([4, 2, 0, 3, 2, 5]) == 9
    assert trap([]) == 0
    assert trap([3]) == 0
    for _ in range(50):
        h = [rng.randint(0, 6) for _ in range(rng.randint(0, 20))]
        assert trap(h) == brute_trap(h)


def test_trap_fast():
    fast(trap, [rng.randint(0, 1000) for _ in range(300_000)])


# 3 ------------------------------------------------------------------------

@pytest.mark.parametrize("digits, expected", [
    ("12", 2), ("226", 3), ("06", 0), ("0", 0), ("10", 1), ("27", 1), ("11106", 2), ("", 1), ("100", 0),
])
def test_num_decodings(digits, expected):
    assert num_decodings(digits) == expected


def test_num_decodings_fast():
    # "11...1" (n ones) decodes in Fibonacci(n + 1) ways
    a, b = 1, 1
    for _ in range(800 - 1):
        a, b = b, a + b
    assert fast(num_decodings, "1" * 800) == b


# 4 ------------------------------------------------------------------------

def test_can_finish():
    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False
    assert can_finish(3, []) is True
    assert can_finish(4, [[1, 0], [2, 1], [3, 2], [1, 3]]) is False
    assert can_finish(5, [[1, 0], [2, 0], [3, 1], [3, 2], [4, 3]]) is True


def test_can_finish_fast():
    n = 50_000
    chain = [[i + 1, i] for i in range(n - 1)]
    assert fast(can_finish, n, chain) is True


# 5 ------------------------------------------------------------------------

def brute_ship(weights, days):
    cap = max(weights)
    while True:
        need, load = 1, 0
        for w in weights:
            if load + w > cap:
                need += 1
                load = 0
            load += w
        if need <= days:
            return cap
        cap += 1


def test_ship_capacity():
    assert ship_capacity([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
    assert ship_capacity([3, 2, 2, 4, 1, 4], 3) == 6
    assert ship_capacity([1, 2, 3, 1, 1], 4) == 3
    for _ in range(30):
        w = [rng.randint(1, 20) for _ in range(rng.randint(1, 12))]
        d = rng.randint(1, len(w))
        assert ship_capacity(w, d) == brute_ship(w, d)


def test_ship_capacity_fast():
    weights = [rng.randint(1, 10**6) for _ in range(50_000)]
    fast(ship_capacity, weights, 7)


# 6 ------------------------------------------------------------------------

def test_kth_largest():
    s = KthLargest(3)
    assert [s.add(x) for x in [4, 5, 8, 2, 3, 5, 10, 9, 4]] == [None, None, 4, 4, 4, 5, 5, 8, 8]


def test_kth_largest_fast():
    s = KthLargest(100)
    data = [rng.randint(0, 10**9) for _ in range(200_000)]
    start = time.perf_counter()
    for x in data:
        last = s.add(x)
    assert time.perf_counter() - start < 2.0, "each add should be O(log k)"
    assert last == sorted(data, reverse=True)[99]


# 7 ------------------------------------------------------------------------

def test_network_delay():
    assert network_delay([(2, 1, 1), (2, 3, 1), (3, 4, 1)], 4, 2) == 2
    assert network_delay([(1, 2, 1)], 2, 1) == 1
    assert network_delay([(1, 2, 1)], 2, 2) == -1
    assert network_delay([(1, 2, 10), (1, 3, 1), (3, 2, 1)], 3, 1) == 2
    assert network_delay([], 1, 1) == 0


def test_network_delay_fast():
    n = 20_000
    times = [(i, i + 1, rng.randint(1, 10)) for i in range(1, n)] + [
        (rng.randint(1, n), rng.randint(1, n), rng.randint(1, 100)) for _ in range(60_000)
    ]
    assert fast(network_delay, times, n, 1) > 0


# 8 ------------------------------------------------------------------------

def brute_rect(h):
    best = 0
    for i in range(len(h)):
        low = float("inf")
        for j in range(i, len(h)):
            low = min(low, h[j])
            best = max(best, low * (j - i + 1))
    return best


def test_largest_rectangle():
    assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
    assert largest_rectangle([2, 4]) == 4
    assert largest_rectangle([]) == 0
    assert largest_rectangle([3, 3, 3]) == 9
    for _ in range(50):
        h = [rng.randint(0, 10) for _ in range(rng.randint(0, 25))]
        assert largest_rectangle(h) == brute_rect(h)


def test_largest_rectangle_fast():
    fast(largest_rectangle, list(range(200_000)))


# 9 ------------------------------------------------------------------------

@pytest.mark.parametrize("s, t, expected", [
    ("ADOBECODEBANC", "ABC", "BANC"), ("a", "a", "a"), ("a", "aa", ""), ("aa", "aa", "aa"),
    ("abcabdebac", "cda", "cabd"), ("xyz", "", ""),
])
def test_min_window(s, t, expected):
    assert min_window(s, t) == expected


def test_min_window_fast():
    s = "".join(rng.choice("abcdefgh") for _ in range(200_000)) + "XYZ"
    assert fast(min_window, s, "XYZ") == "XYZ"


# 10 -----------------------------------------------------------------------

def brute_partition(nums):
    total = sum(nums)
    return total % 2 == 0 and any(
        sum(c) == total // 2 for r in range(len(nums) + 1) for c in itertools.combinations(nums, r)
    )


def test_can_partition():
    assert can_partition([1, 5, 11, 5]) is True
    assert can_partition([1, 2, 3, 5]) is False
    assert can_partition([2, 2]) is True
    for _ in range(40):
        nums = [rng.randint(1, 20) for _ in range(rng.randint(1, 12))]
        assert can_partition(nums) == brute_partition(nums)


def test_can_partition_fast():
    nums = [rng.randint(1, 50) for _ in range(200)]
    fast(can_partition, nums)
