import random
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    LRUCache,
    Node,
    from_list,
    has_cycle,
    insert_at,
    length,
    merge_sorted,
    middle,
    remove_all,
    remove_nth_from_end,
    reverse,
    to_list,
)

rng = random.Random(17)


@pytest.mark.parametrize("name", ["length", "insert_at", "remove_all", "reverse", "middle", "has_cycle", "merge_sorted", "remove_nth_from_end"])
def test_works_on_nodes_not_python_lists(name):
    assert not uses_any(getattr(exercises, name), "to_list", "from_list", "list", "sorted", "set")


def test_length():
    assert length(from_list([4, 5, 6])) == 3
    assert length(None) == 0
    assert length(Node(1)) == 1


@pytest.mark.parametrize("values, index, value, expected", [
    ([1, 2, 3], 0, 9, [9, 1, 2, 3]),
    ([1, 2, 3], 1, 9, [1, 9, 2, 3]),
    ([1, 2, 3], 3, 9, [1, 2, 3, 9]),
    ([1, 2, 3], 99, 9, [1, 2, 3, 9]),
    ([], 0, 9, [9]),
])
def test_insert_at(values, index, value, expected):
    assert to_list(insert_at(from_list(values), index, value)) == expected


@pytest.mark.parametrize("values, target, expected", [
    ([1, 2, 1, 3, 1], 1, [2, 3]), ([1, 1], 1, []), ([], 1, []), ([2, 3], 1, [2, 3]), ([5, 6, 6], 6, [5]),
])
def test_remove_all(values, target, expected):
    assert to_list(remove_all(from_list(values), target)) == expected


@pytest.mark.parametrize("values", [[], [1], [1, 2], [1, 2, 3, 4, 5]])
def test_reverse(values):
    head = from_list(values)
    nodes = []
    node = head
    while node:
        nodes.append(node)
        node = node.next
    new_head = reverse(head)
    assert to_list(new_head) == values[::-1]
    if nodes:
        assert new_head is nodes[-1], "reuse the existing nodes"


@pytest.mark.parametrize("values, expected", [([1, 2, 3], 2), ([1, 2, 3, 4], 3), ([7], 7), ([1, 2], 2)])
def test_middle(values, expected):
    assert middle(from_list(values)).value == expected


def test_has_cycle():
    assert has_cycle(None) is False
    assert has_cycle(from_list([1, 2, 3])) is False
    head = from_list([1, 2, 3, 4, 5])
    tail = head.next.next.next.next
    tail.next = head.next.next          # 5 -> 3
    assert has_cycle(head) is True
    loop = Node(1)
    loop.next = loop
    assert has_cycle(loop) is True


def test_merge_sorted():
    assert to_list(merge_sorted(from_list([1, 4, 9]), from_list([2, 3, 10]))) == [1, 2, 3, 4, 9, 10]
    assert to_list(merge_sorted(None, from_list([1]))) == [1]
    assert merge_sorted(None, None) is None
    for _ in range(30):
        a = sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 10)))
        b = sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 10)))
        assert to_list(merge_sorted(from_list(a), from_list(b))) == sorted(a + b)


@pytest.mark.parametrize("values, n, expected", [
    ([1, 2, 3, 4, 5], 2, [1, 2, 3, 5]), ([1], 1, []), ([1, 2], 1, [1]), ([1, 2], 2, [2]),
])
def test_remove_nth_from_end(values, n, expected):
    assert to_list(remove_nth_from_end(from_list(values), n)) == expected


def test_lru_cache():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1          # a is now most recent
    cache.put("c", 3)                   # evicts b
    assert cache.get("b") is None
    cache.put("a", 10)                  # update a, most recent
    cache.put("d", 4)                   # evicts c
    assert cache.get("c") is None
    assert cache.get("a") == 10
    assert cache.get("d") == 4


def test_lru_cache_matches_reference():
    cache = LRUCache(5)
    reference = []                      # list of (key, value), most recent last
    for _ in range(2000):
        key = rng.randint(0, 9)
        if rng.random() < 0.5:
            value = rng.randint(0, 100)
            cache.put(key, value)
            reference = [kv for kv in reference if kv[0] != key] + [(key, value)]
            if len(reference) > 5:
                reference.pop(0)
        else:
            found = next((v for k, v in reference if k == key), None)
            assert cache.get(key) == found
            if found is not None:
                reference = [kv for kv in reference if kv[0] != key] + [(key, found)]


def test_lru_cache_is_constant_time():
    cache = LRUCache(50_000)
    start = time.perf_counter()
    for i in range(200_000):
        cache.put(i, i)
        cache.get(i - 1000)
    assert time.perf_counter() - start < 2.0, "get and put must be O(1)"
