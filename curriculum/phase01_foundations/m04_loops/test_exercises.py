import inspect
import time

import pytest

import exercises
from exercises import (
    collatz_steps,
    gcd,
    is_prime,
    newton_sqrt,
    primes_up_to,
    reverse_digits,
    staircase,
    sum_of_multiples,
)


@pytest.mark.parametrize("limit, expected", [(10, 23), (1, 0), (16, 60), (1000, 233168)])
def test_sum_of_multiples(limit, expected):
    assert sum_of_multiples(limit) == expected


@pytest.mark.parametrize("n, expected", [(1, 0), (2, 1), (6, 8), (27, 111), (97, 118)])
def test_collatz_steps(n, expected):
    assert collatz_steps(n) == expected


@pytest.mark.parametrize("n, expected", [
    (-7, False), (0, False), (1, False), (2, True), (3, True), (4, False),
    (25, False), (97, True), (7919, True), (1_000_000_007, True), (1_000_000_008, False),
])
def test_is_prime(n, expected):
    assert is_prime(n) == expected


def test_primes_up_to():
    assert primes_up_to(20) == [2, 3, 5, 7, 11, 13, 17, 19]
    assert primes_up_to(1) == []
    assert primes_up_to(2) == [2]


def test_primes_up_to_is_fast():
    start = time.perf_counter()
    primes = primes_up_to(200_000)
    elapsed = time.perf_counter() - start
    assert len(primes) == 17984
    assert primes[-1] == 199999
    assert elapsed < 1.0, f"took {elapsed:.2f}s; try the Sieve of Eratosthenes"


@pytest.mark.parametrize("n, expected", [(1234, 4321), (1200, 21), (0, 0), (7, 7), (1000000001, 1000000001)])
def test_reverse_digits(n, expected):
    assert reverse_digits(n) == expected


def test_reverse_digits_uses_arithmetic():
    assert "str(" not in inspect.getsource(exercises.reverse_digits)


@pytest.mark.parametrize("a, b, expected", [(48, 18, 6), (18, 48, 6), (17, 5, 1), (0, 9, 9), (9, 0, 9), (270, 192, 6)])
def test_gcd(a, b, expected):
    assert gcd(a, b) == expected


def test_gcd_uses_a_loop():
    source = inspect.getsource(exercises.gcd)
    assert "math.gcd" not in source and "while" in source


@pytest.mark.parametrize("x", [0, 1, 2, 9, 0.25, 1e-4, 12345.678, 1e10])
def test_newton_sqrt(x):
    result = newton_sqrt(x)
    assert abs(result * result - x) <= 1e-9 * x


def test_newton_sqrt_does_not_cheat():
    source = inspect.getsource(exercises.newton_sqrt)
    assert "** 0.5" not in source and "**0.5" not in source and "sqrt(" not in source.split(":", 1)[1]


def test_staircase():
    assert staircase(3) == "  #\n ##\n###"
    assert staircase(1) == "#"
    assert staircase(5).split("\n")[0] == "    #"
