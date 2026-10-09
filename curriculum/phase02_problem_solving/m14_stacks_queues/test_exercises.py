import random
import time

import pytest

from exercises import (
    MinStack,
    TwoStackQueue,
    balanced,
    calculate,
    days_until_warmer,
    eval_rpn,
    hot_potato,
    simplify_path,
    to_rpn,
)


@pytest.mark.parametrize("text, expected", [
    ("f(a[1]) {ok}", True), ("", True), ("(]", False), ("((", False), ("))", False),
    ("{[()()]}", True), ("([)]", False), ("no brackets", True),
])
def test_balanced(text, expected):
    assert balanced(text) == expected


@pytest.mark.parametrize("path, expected", [
    ("/a/./b/../../c/", "/c"), ("/../", "/"), ("/home//user/", "/home/user"), ("/", "/"),
    ("/a/b/c/../..", "/a"), ("/...", "/..."),
])
def test_simplify_path(path, expected):
    assert simplify_path(path) == expected


@pytest.mark.parametrize("tokens, expected", [
    (["2", "1", "+", "3", "*"], 9),
    (["4", "13", "5", "/", "+"], 6),
    (["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"], 22),
    (["7"], 7),
    (["7", "-2", "/"], -3),
])
def test_eval_rpn(tokens, expected):
    assert eval_rpn(tokens) == expected


@pytest.mark.parametrize("tokens, expected", [
    (["3", "+", "4", "*", "2"], ["3", "4", "2", "*", "+"]),
    (["(", "1", "+", "2", ")", "*", "3"], ["1", "2", "+", "3", "*"]),
    (["8", "-", "3", "-", "2"], ["8", "3", "-", "2", "-"]),
    (["8", "/", "2", "*", "3"], ["8", "2", "/", "3", "*"]),
    (["5"], ["5"]),
])
def test_to_rpn(tokens, expected):
    assert to_rpn(tokens) == expected


@pytest.mark.parametrize("expression, expected", [
    ("2 * ( 3 + 4 ) - 5", 9), ("8 - 3 - 2", 3), ("( ( 2 ) )", 2), ("100 / 7 / 2", 7), ("1 + 2 * 3 - 4 / 2", 5),
])
def test_calculate(expression, expected):
    assert calculate(expression) == expected


def test_days_until_warmer():
    assert days_until_warmer([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert days_until_warmer([30, 60, 90]) == [1, 1, 0]
    assert days_until_warmer([90, 60, 30]) == [0, 0, 0]
    assert days_until_warmer([]) == []


def test_days_until_warmer_is_linear():
    temps = list(range(200_000, 0, -1)) + [10**9]
    start = time.perf_counter()
    result = days_until_warmer(temps)
    assert time.perf_counter() - start < 1.5, "use a monotonic stack: O(n)"
    assert result[0] == 200_000


def test_min_stack():
    s = MinStack()
    s.push(5)
    s.push(3)
    s.push(7)
    s.push(3)
    assert s.get_min() == 3
    assert s.pop() == 3
    assert s.get_min() == 3
    assert s.pop() == 7
    assert s.top() == 3
    assert s.pop() == 3
    assert s.get_min() == 5


def test_min_stack_random():
    rng = random.Random(14)
    s, mirror = MinStack(), []
    for _ in range(2000):
        if mirror and rng.random() < 0.4:
            assert s.pop() == mirror.pop()
        else:
            v = rng.randint(-100, 100)
            s.push(v)
            mirror.append(v)
        if mirror:
            assert s.get_min() == min(mirror)
            assert s.top() == mirror[-1]


def test_two_stack_queue():
    q = TwoStackQueue()
    for x in [1, 2, 3]:
        q.enqueue(x)
    assert q.dequeue() == 1
    q.enqueue(4)
    assert [q.dequeue(), q.dequeue(), q.dequeue()] == [2, 3, 4]
    assert len(q) == 0


def test_two_stack_queue_is_fast():
    q = TwoStackQueue()
    start = time.perf_counter()
    for i in range(200_000):
        q.enqueue(i)
    assert len(q) == 200_000
    assert [q.dequeue() for _ in range(200_000)] == list(range(200_000))
    assert time.perf_counter() - start < 2.0, "dequeue should be O(1) amortized"


def test_hot_potato():
    assert hot_potato(["A", "B", "C", "D"], 1) == ["B", "D", "C", "A"]
    assert hot_potato(["A", "B", "C"], 0) == ["A", "B", "C"]
    assert hot_potato(["Solo"], 5) == ["Solo"]
    assert hot_potato(list("ABCDEFG"), 2) == list("CFBGEAD")
