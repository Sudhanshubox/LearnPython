import itertools
import random
import time

import pytest

from exercises import (
    can_jump,
    fractional_knapsack,
    greedy_change,
    greedy_is_optimal,
    huffman_lengths,
    max_meetings,
    merge_intervals,
    min_rooms,
)

rng = random.Random(22)


def random_intervals(n):
    out = []
    for _ in range(n):
        s = rng.randint(0, 20)
        out.append([s, s + rng.randint(1, 6)])
    return out


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def brute_max_meetings(intervals):
    for r in range(len(intervals), 0, -1):
        for combo in itertools.combinations(intervals, r):
            if all(not overlaps(a, b) for a, b in itertools.combinations(combo, 2)):
                return r
    return 0


def test_max_meetings():
    assert max_meetings([[1, 3], [2, 4], [3, 5], [6, 7]]) == 3
    assert max_meetings([]) == 0
    assert max_meetings([[1, 10], [2, 3], [4, 5]]) == 2
    for _ in range(40):
        iv = random_intervals(rng.randint(0, 8))
        assert max_meetings(iv) == brute_max_meetings(iv)


def test_merge_intervals():
    assert merge_intervals([[1, 3], [8, 10], [2, 6], [10, 12]]) == [[1, 6], [8, 12]]
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 4], [2, 3]]) == [[1, 4]]
    assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]


def brute_rooms(intervals):
    points = sorted({t for iv in intervals for t in iv})
    return max((sum(1 for s, e in intervals if s <= t < e) for t in points), default=0)


def test_min_rooms():
    assert min_rooms([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_rooms([[7, 10], [2, 4]]) == 1
    assert min_rooms([[1, 5], [5, 10]]) == 1
    assert min_rooms([]) == 0
    for _ in range(40):
        iv = random_intervals(rng.randint(0, 15))
        assert min_rooms(iv) == brute_rooms(iv)


@pytest.mark.parametrize("nums, expected", [
    ([2, 3, 1, 1, 4], True), ([3, 2, 1, 0, 4], False), ([0], True), ([1, 0, 1], False), ([2, 0, 0], True),
])
def test_can_jump(nums, expected):
    assert can_jump(nums) == expected


def test_can_jump_is_linear():
    nums = [1] * 300_000
    start = time.perf_counter()
    assert can_jump(nums) is True
    assert time.perf_counter() - start < 1.0


def test_fractional_knapsack():
    assert fractional_knapsack([(10, 60), (20, 100), (30, 120)], 50) == pytest.approx(240.0)
    assert fractional_knapsack([(10, 60)], 5) == pytest.approx(30.0)
    assert fractional_knapsack([], 10) == 0


def test_greedy_change():
    assert greedy_change([1, 5, 10, 25], 30) == [25, 5]
    assert greedy_change([25, 10, 5, 1], 41) == [25, 10, 5, 1]
    assert greedy_change([1, 5, 6], 10) == [6, 1, 1, 1, 1]
    assert greedy_change([3, 5], 7) is None
    assert greedy_change([2], 0) == []


def check_prefix_lengths(lengths):
    # Kraft's inequality must hold with equality for an optimal (full) prefix code
    return abs(sum(2 ** -n for n in lengths.values()) - 1) < 1e-9 or len(lengths) == 1


def cost(freqs, lengths):
    return sum(freqs[s] * lengths[s] for s in freqs)


def test_huffman_lengths_example():
    freqs = {"a": 5, "b": 9, "c": 12, "d": 13, "e": 16, "f": 45}
    lengths = huffman_lengths(freqs)
    assert set(lengths) == set(freqs)
    assert cost(freqs, lengths) == 224
    assert lengths["f"] == 1
    assert check_prefix_lengths(lengths)


def test_huffman_single_and_pair():
    assert huffman_lengths({"x": 7}) == {"x": 1}
    assert huffman_lengths({"x": 1, "y": 100}) == {"x": 1, "y": 1}


def optimal_cost(freqs):
    import heapq
    h = list(freqs.values())
    heapq.heapify(h)
    total = 0
    while len(h) > 1:
        a, b = heapq.heappop(h), heapq.heappop(h)
        total += a + b
        heapq.heappush(h, a + b)
    return total if len(freqs) > 1 else sum(freqs.values())


def test_huffman_random():
    for _ in range(30):
        n = rng.randint(2, 15)
        freqs = {f"s{i}": rng.randint(1, 100) for i in range(n)}
        lengths = huffman_lengths(freqs)
        assert cost(freqs, lengths) == optimal_cost(freqs)
        assert check_prefix_lengths(lengths)


@pytest.mark.parametrize("coins, up_to, expected", [
    ([1, 5, 10, 25], 100, True),
    ([1, 5, 6], 20, False),
    ([1, 3, 4], 10, False),
    ([1, 2, 5, 10, 20, 50, 100, 200, 500, 2000], 3000, True),
    ([1], 50, True),
])
def test_greedy_is_optimal(coins, up_to, expected):
    assert greedy_is_optimal(coins, up_to) == expected
