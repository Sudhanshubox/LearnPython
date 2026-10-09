import math
import random
import statistics
import time
from collections import Counter

import pytest

import exercises
from code_checks import uses_any
from exercises import MinHeap, k_closest, merge_k, running_median, sample_top_k, schedule, top_k, top_k_probs

rng = random.Random(19)


def test_min_heap_written_from_scratch():
    assert not uses_any(exercises.MinHeap, "heapq", "heappush", "heappop", "heapify", "sorted", "sort", "min")


def test_min_heap_basic():
    h = MinHeap()
    for x in [5, 3, 8, 1, 9, 2]:
        h.push(x)
    assert len(h) == 6
    assert h.peek() == 1
    assert [h.pop() for _ in range(6)] == [1, 2, 3, 5, 8, 9]
    assert len(h) == 0
    with pytest.raises(IndexError):
        h.pop()


def test_min_heap_from_items():
    items = [rng.randint(0, 100) for _ in range(200)]
    h = MinHeap(items)
    assert len(h) == 200
    assert [h.pop() for _ in range(200)] == sorted(items)


def test_min_heap_random_operations():
    h, mirror = MinHeap(), []
    for _ in range(3000):
        if mirror and rng.random() < 0.45:
            assert h.pop() == min(mirror)
            mirror.remove(min(mirror))
        else:
            x = rng.randint(-1000, 1000)
            h.push(x)
            mirror.append(x)
        assert len(h) == len(mirror)


def test_min_heap_is_logarithmic():
    h = MinHeap()
    start = time.perf_counter()
    for i in range(100_000, 0, -1):
        h.push(i)
    for _ in range(100_000):
        h.pop()
    assert time.perf_counter() - start < 3.0


def test_top_k():
    assert top_k([5, 1, 9, 3, 7], 2) == [9, 7]
    assert top_k([1, 2], 5) == [2, 1]
    assert top_k([], 3) == []
    data = [rng.randint(0, 10**6) for _ in range(5000)]
    assert top_k(iter(data), 10) == sorted(data, reverse=True)[:10]


def test_top_k_does_not_sort_everything():
    assert not uses_any(exercises.top_k, "nlargest")


def test_merge_k():
    assert merge_k([[1, 4], [2, 5], [0, 3, 6]]) == [0, 1, 2, 3, 4, 5, 6]
    assert merge_k([[], [1], []]) == [1]
    assert merge_k([]) == []
    lists = [sorted(rng.randint(0, 100) for _ in range(rng.randint(0, 20))) for _ in range(10)]
    assert merge_k(lists) == sorted(x for lst in lists for x in lst)


def test_merge_k_uses_a_heap():
    assert uses_any(exercises.merge_k, "heappush", "heappop", "heapify", "heapreplace", "heappushpop")
    assert not uses_any(exercises.merge_k, "sorted", "sort")


def test_k_closest():
    assert k_closest([(1, 3), (-2, 2), (5, 0)], 2) == [(-2, 2), (1, 3)]
    assert k_closest([(3, 4), (4, 3), (0, 1)], 2) == [(0, 1), (3, 4)]
    assert k_closest([(1, 1)], 3) == [(1, 1)]


def test_running_median():
    assert running_median([5, 15, 1, 3]) == [5, 10.0, 5, 4.0]
    assert running_median([]) == []
    data = [rng.randint(-100, 100) for _ in range(300)]
    result = running_median(data)
    for i in range(len(data)):
        assert result[i] == statistics.median(data[: i + 1])


def test_running_median_is_fast():
    data = [rng.random() for _ in range(100_000)]
    start = time.perf_counter()
    running_median(data)
    assert time.perf_counter() - start < 2.0, "use two heaps: O(log n) per number"


def test_schedule():
    assert schedule([("email", 1, 0), ("deploy", 5, 1), ("backup", 5, 2)]) == ["deploy", "backup", "email"]
    assert schedule([]) == []
    assert schedule([("b", 2, 1), ("a", 2, 0), ("c", 3, 2)]) == ["c", "a", "b"]


def test_top_k_probs():
    probs = top_k_probs({"the": 2.0, "a": 1.0, "cat": 0.1}, 2)
    assert set(probs) == {"the", "a"}
    assert probs["the"] == pytest.approx(math.exp(1) / (math.exp(1) + 1))
    assert sum(probs.values()) == pytest.approx(1.0)


def test_top_k_probs_is_numerically_stable():
    probs = top_k_probs({"x": 1000.0, "y": 999.0, "z": -5.0}, 3)
    assert probs["x"] == pytest.approx(1 / (1 + math.exp(-1) + math.exp(-1005)))


def test_sample_top_k_distribution():
    logits = {"the": 2.0, "a": 1.0, "cat": 0.5, "zebra": -3.0}
    r = random.Random(0)
    counts = Counter(sample_top_k(logits, 2, r) for _ in range(20_000))
    assert set(counts) == {"the", "a"}, "only the top-k tokens can be sampled"
    expected = top_k_probs(logits, 2)["the"]
    assert counts["the"] / 20_000 == pytest.approx(expected, abs=0.02)
