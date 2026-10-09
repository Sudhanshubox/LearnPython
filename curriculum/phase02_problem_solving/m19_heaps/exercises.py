"""m19 exercises: heaps. From exercise 2 on, use heapq."""

import heapq
import math
import random


# 1. A min-heap stored in a Python list, written from scratch (no heapq!).
#    Children of index i: 2i + 1 and 2i + 2. Parent: (i - 1) // 2.
class MinHeap:
    def __init__(self, items=()):
        """Build the heap from any iterable in O(n): copy it, then sift down every
        non-leaf, from the last one back to the root."""
        raise NotImplementedError

    def push(self, item):
        """Append, then sift up. O(log n)."""
        raise NotImplementedError

    def pop(self):
        """Remove and return the smallest item. O(log n). Raise IndexError if empty."""
        raise NotImplementedError

    def peek(self):
        """Return the smallest item without removing it."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# 2. The k largest numbers from an iterable (possibly a huge stream), largest first.
#    Use a min-heap of size k: O(n log k) time, O(k) memory. Don't sort the whole input.
#    top_k([5, 1, 9, 3, 7], 2) -> [9, 7]
def top_k(stream, k):
    raise NotImplementedError


# 3. Merge k sorted lists into one sorted list using a heap. O(N log k).
#    merge_k([[1, 4], [2, 5], [0, 3, 6]]) -> [0, 1, 2, 3, 4, 5, 6]
def merge_k(lists):
    raise NotImplementedError


# 4. The k points closest to the origin (x, y), closest first.
#    Ties: keep the one that appears first in the input.
#    k_closest([(1, 3), (-2, 2), (5, 0)], 2) -> [(-2, 2), (1, 3)]
def k_closest(points, k):
    raise NotImplementedError


# 5. Running median: return a list with the median after each new number.
#    Use two heaps (a max-heap via negation for the lower half, a min-heap for the upper).
#    running_median([5, 15, 1, 3]) -> [5, 10.0, 5, 4.0]
#    (Odd count: the middle number. Even count: the average of the two middles, as a float.)
def running_median(stream):
    raise NotImplementedError


# 6. Task scheduler: tasks are (name, priority, arrival_order) tuples. Return the names
#    in the order they'd be run: highest priority first; equal priorities by arrival_order.
#    schedule([("email", 1, 0), ("deploy", 5, 1), ("backup", 5, 2)])
#    -> ["deploy", "backup", "email"]
def schedule(tasks):
    raise NotImplementedError


# 7. Top-k sampling for text generation. `logits` maps each token to a score.
#    Keep only the k highest-scoring tokens, turn their scores into probabilities with
#    softmax (p_i = exp(s_i) / sum_j exp(s_j); subtract the max score first for
#    numerical stability), and return a dict token -> probability.
#    top_k_probs({"the": 2.0, "a": 1.0, "cat": 0.1}, 2) -> {"the": 0.731..., "a": 0.268...}
def top_k_probs(logits, k):
    raise NotImplementedError


#    Then sample one token from those probabilities using `rng` (a random.Random).
#    Use rng.random() once and walk the cumulative probabilities.
def sample_top_k(logits, k, rng):
    raise NotImplementedError
