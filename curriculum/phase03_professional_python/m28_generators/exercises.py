"""m28 exercises: iterators and generators.

Many tests pass INFINITE iterators to your functions, so never call list() on an
input unless the exercise says so. Don't use itertools in exercises 2-7 (write the
logic yourself); you may use it in 8 and 9.
"""

import json
import random
from collections import deque


# 1. An iterator CLASS (no yield allowed) that counts down from start to 1.
#    list(Countdown(3)) -> [3, 2, 1]
class Countdown:
    def __init__(self, start):
        raise NotImplementedError


# 2. An infinite generator of Fibonacci numbers: 0, 1, 1, 2, 3, 5, ...
def fibonacci():
    raise NotImplementedError


# 3. Yield the first n items of any iterable (stop early: it may be infinite).
#    list(take(3, fibonacci())) -> [0, 1, 1]
def take(n, iterable):
    raise NotImplementedError


# 4. Yield lists of `size` items; the last may be shorter. Works on infinite inputs.
#    list(chunked(range(5), 2)) -> [[0, 1], [2, 3], [4]]
def chunked(iterable, size):
    raise NotImplementedError


# 5. Yield tuples of n consecutive items (a sliding window). Use a deque(maxlen=n).
#    list(windows([1, 2, 3, 4], 3)) -> [(1, 2, 3), (2, 3, 4)]
def windows(iterable, n):
    raise NotImplementedError


# 6. Yield the running mean after each item.
#    list(running_mean([2, 4, 6])) -> [2.0, 3.0, 4.0]
def running_mean(iterable):
    raise NotImplementedError


# 7. Flatten arbitrarily nested lists and tuples lazily, using `yield from`.
#    Strings are NOT nested (yield them whole).
#    list(flatten([1, [2, (3, [4])], "ab"])) -> [1, 2, 3, 4, "ab"]
def flatten(nested):
    raise NotImplementedError


# 8. A lazy data pipeline over JSON Lines text. Each function takes an iterable and
#    returns a generator:
#    - parse_jsonl(lines): skip blank lines; yield json.loads(line) for the rest.
#    - valid_examples(records): yield records that have a non-empty string "text"
#      and an int "label".
#    - tokenize(records): yield (lowercased list of words of text, label) pairs.
#    pipeline(lines) chains all three.
def parse_jsonl(lines):
    raise NotImplementedError


def valid_examples(records):
    raise NotImplementedError


def tokenize(records):
    raise NotImplementedError


def pipeline(lines):
    raise NotImplementedError


# 9. A data loader. For each epoch, shuffle the indices 0..n_items-1 with
#    random.Random(seed + epoch).shuffle, then yield (epoch, batch_of_indices) for each
#    batch of `batch_size` indices (the last batch may be smaller unless drop_last=True).
#    Every epoch must visit every index exactly once (except dropped ones).
def data_loader(n_items, batch_size, epochs, seed=0, drop_last=False):
    raise NotImplementedError
