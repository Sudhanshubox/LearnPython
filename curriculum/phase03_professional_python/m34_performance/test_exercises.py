import random
import string
import time

import pytest

import exercises
from exercises import (
    Point,
    SlotPoint,
    best_time,
    bytes_per_instance,
    hottest_functions,
    list_growth_sizes,
    opnames,
    uses_globals,
)

rng = random.Random(34)


def measure(func, *args):
    start = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - start


def test_best_time():
    calls = []

    def work(seconds):
        calls.append(seconds)
        time.sleep(seconds)

    t = best_time(work, 0.02, repeat=3)
    assert len(calls) == 3
    assert 0.015 < t < 0.2


def slow_helper(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def fast_helper(n):
    return n


def program(n):
    for _ in range(3):
        slow_helper(n)
        fast_helper(n)


def test_hottest_functions():
    names = hottest_functions(program, 200_000, n=2)
    assert names[0] == "slow_helper"
    assert len(names) == 2


def random_text(n_words, vocab):
    return " ".join(rng.choice(vocab) for _ in range(n_words))


VOCAB = ["".join(rng.choice(string.ascii_lowercase) for _ in range(5)) for _ in range(3000)]

CASES = {
    "common_words": (lambda: (random_text(4000, VOCAB), random_text(4000, VOCAB)),),
    "build_csv_line": (lambda: ([rng.randint(0, 10**6) for _ in range(20_000)],),),
    "running_max": (lambda: ([rng.randint(0, 10**6) for _ in range(4000)],),),
    "pair_counts": (lambda: ([rng.randint(0, 50) for _ in range(2500)],),),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", sorted(CASES))
def test_fast_versions_match_and_are_10x_faster(name):
    slow = getattr(exercises, f"slow_{name}")
    fast = getattr(exercises, f"fast_{name}")
    args = CASES[name][0]()
    expected, slow_time = measure(slow, *args)
    result, fast_time = measure(fast, *args)
    assert result == expected
    assert fast_time * 10 < slow_time, f"slow: {slow_time:.3f}s, yours: {fast_time:.3f}s (need 10x)"


@pytest.mark.parametrize("name", sorted(CASES))
def test_fast_versions_small_cases(name):
    slow = getattr(exercises, f"slow_{name}")
    fast = getattr(exercises, f"fast_{name}")
    small = {
        "common_words": [("a b c a", "c a d"), ("", "x"), ("Hi hi", "HI")],
        "build_csv_line": [([1, 2, 3],), ([],), (["x"],)],
        "running_max": [([3, 1, 4, 1, 5],), ([],), ([-2, -5],)],
        "pair_counts": [([1, 1, 1],), ([],), (["a", "b", "a"],)],
    }[name]
    for args in small:
        assert fast(*args) == slow(*args)


def test_opnames():
    names = opnames(lambda a, b: a + b)
    assert "RETURN_VALUE" in names
    assert any(n.startswith("BINARY") for n in names)
    assert all(isinstance(n, str) for n in names)


def test_uses_globals():
    def local_only(a, b):
        return a * b

    def uses_len(items):
        return len(items)

    assert uses_globals(uses_len) is True
    assert uses_globals(local_only) is False


def test_list_growth_sizes():
    sizes = list_growth_sizes(64)
    assert sizes == sorted(set(sizes))
    assert 3 <= len(sizes) < 32, "the list should grow in jumps, not on every append"


def test_slot_point():
    p = SlotPoint(1, 2)
    assert (p.x, p.y) == (1, 2)
    assert not hasattr(p, "__dict__")
    with pytest.raises(AttributeError):
        p.z = 3
    assert "__slots__" in vars(SlotPoint)


def test_bytes_per_instance():
    regular = bytes_per_instance(Point)
    slotted = bytes_per_instance(SlotPoint)
    assert regular > 0 and slotted > 0
    assert slotted < regular, f"__slots__ should save memory ({slotted} vs {regular} bytes)"
