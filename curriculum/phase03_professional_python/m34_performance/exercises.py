"""m34 exercises: performance and CPython internals."""

import cProfile
import dis
import io
import pstats
import sys
import time
import tracemalloc


# ---------------------------------------------------------------- measuring

# 1. Call func(*args) `repeat` times and return the MINIMUM duration in seconds,
#    measured with time.perf_counter.
def best_time(func, *args, repeat=5):
    raise NotImplementedError


# 2. Profile func(*args) with cProfile and return the names (co_name, e.g. "slow_helper")
#    of the `n` functions with the highest TOTAL time (time spent in the function itself,
#    excluding the functions it calls), highest first.
#    Hint: pstats.Stats(profiler).stats is a dict mapping (filename, line, name) to
#    (call_count, primitive_calls, total_time, cumulative_time, callers).
def hottest_functions(func, *args, n=3):
    raise NotImplementedError


# ---------------------------------------------------------------- optimizing
# Each slow_* function below is correct but slow. Write the fast_* version: it must
# return exactly the same result and be at least 10x faster on large inputs.

def slow_common_words(text_a, text_b):
    """Words (lowercase) that appear in both texts, sorted, without duplicates."""
    words_b = text_b.lower().split()
    common = []
    for w in text_a.lower().split():
        if w in words_b and w not in common:
            common.append(w)
    return sorted(common)


def fast_common_words(text_a, text_b):
    raise NotImplementedError


def slow_build_csv_line(values):
    """Join values with commas, as one string.

    Each f-string builds a brand-new string, copying everything so far: O(n²).
    (CPython can sometimes optimize `s = s + x` in place, but not this.)"""
    line = ""
    for v in values:
        line = f"{line},{v}" if line else str(v)
    return line


def fast_build_csv_line(values):
    raise NotImplementedError


def slow_running_max(nums):
    """result[i] = max(nums[0..i])."""
    return [max(nums[: i + 1]) for i in range(len(nums))]


def fast_running_max(nums):
    raise NotImplementedError


def slow_pair_counts(items):
    """How many pairs (i < j) have items[i] == items[j]."""
    count = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                count += 1
    return count


def fast_pair_counts(items):
    raise NotImplementedError


# ---------------------------------------------------------------- internals

# 3. Return the list of bytecode operation names (opname) of a function, in order,
#    using dis.get_instructions.
def opnames(func):
    raise NotImplementedError


# 4. Does the function's bytecode load any global or built-in name (LOAD_GLOBAL)?
def uses_globals(func):
    raise NotImplementedError


# 5. Append items to an empty list one at a time, recording sys.getsizeof(list) after
#    each append, and return the sorted list of DISTINCT sizes seen for n appends.
#    (Watch the list over-allocate: the size jumps only now and then.)
def list_growth_sizes(n):
    raise NotImplementedError


# 6. A memory-lean class. Point stores x and y normally. SlotPoint stores them
#    using __slots__ (so it has no __dict__ and can't get new attributes).
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class SlotPoint:
    pass


# 7. Use tracemalloc to measure how many bytes it takes to create n instances of cls
#    (cls(i, i) for i in range(n)), keeping them all alive in a list.
#    Return the bytes per instance (an int: traced difference // n).
#    Hint: tracemalloc.start(); tracemalloc.get_traced_memory()[0] before and after;
#    tracemalloc.stop().
def bytes_per_instance(cls, n=10_000):
    raise NotImplementedError
