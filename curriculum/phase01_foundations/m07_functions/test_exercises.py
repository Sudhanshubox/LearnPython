import inspect
import math
import time

import pytest

import exercises
from code_checks import is_recursive, uses_any
from exercises import (
    add_task,
    average,
    compose,
    derivative,
    factorial,
    fib,
    flatten,
    format_price,
    make_counter,
    permutations,
)


def test_average():
    assert average(1, 2, 3) == 2.0
    assert average(5) == 5.0
    assert average() == 0.0
    assert average(*range(101)) == 50.0


def test_format_price():
    assert format_price(12.5) == "Rs 12.50"
    assert format_price(3, "$", decimals=0) == "$ 3"
    assert format_price(0.125, "€", decimals=3) == "€ 0.125"


def test_format_price_decimals_is_keyword_only():
    param = inspect.signature(format_price).parameters["decimals"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        format_price(1, "Rs", 2)


def test_add_task_has_no_shared_default():
    assert add_task("read") == ["read"]
    assert add_task("code") == ["code"]
    mine = ["plan"]
    assert add_task("ship", mine) == ["plan", "ship"]
    assert mine == ["plan", "ship"]


def test_compose():
    inc = lambda x: x + 1
    dbl = lambda x: x * 2
    assert compose(inc, dbl)(5) == 11
    assert compose(dbl, inc)(5) == 12
    assert compose(str, abs)(-3) == "3"


def test_make_counter():
    a = make_counter()
    b = make_counter()
    assert [a(), a(), a()] == [1, 2, 3]
    assert b() == 1
    assert a() == 4


@pytest.mark.parametrize("n, expected", [(0, 1), (1, 1), (5, 120), (10, 3628800)])
def test_factorial(n, expected):
    assert factorial(n) == expected


def test_factorial_is_recursive():
    assert is_recursive(exercises.factorial)
    assert not uses_any(exercises.factorial, "math")


def test_fib_values():
    assert [fib(n) for n in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


def test_fib_is_recursive_and_fast():
    assert is_recursive(exercises.fib) or uses_any(exercises.fib, "cache", "lru_cache")
    start = time.perf_counter()
    assert fib(300) == 222232244629420445529739893461909967206666939096499764990979600
    assert time.perf_counter() - start < 0.5, "memoize: remember answers you've already computed"


def test_flatten():
    assert flatten([1, [2, [3, [4]], 5]]) == [1, 2, 3, 4, 5]
    assert flatten([]) == []
    assert flatten([[[]]]) == []
    assert flatten(["a", ["b"]]) == ["a", "b"]


def test_permutations():
    assert permutations("abc") == ["abc", "acb", "bac", "bca", "cab", "cba"]
    assert permutations("aab") == ["aab", "aba", "baa"]
    assert permutations("a") == ["a"]
    assert len(permutations("abcdef")) == 720


def test_permutations_without_itertools():
    assert not uses_any(exercises.permutations, "itertools")


@pytest.mark.parametrize("f, x, expected", [
    (lambda x: x ** 2, 3, 6.0),
    (lambda x: x ** 3, 2, 12.0),
    (math.sin, 0, 1.0),
    (math.exp, 1, math.e),
    (lambda x: 7, 100, 0.0),
])
def test_derivative(f, x, expected):
    assert derivative(f, x) == pytest.approx(expected, rel=1e-6, abs=1e-6)
