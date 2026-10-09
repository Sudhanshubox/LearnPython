import subprocess
import sys
from pathlib import Path

import pytest

from exercises import (
    celsius_to_fahrenheit,
    digit_sum,
    last_digit_of_power,
    nearly_equal,
    receipt_line,
    seconds_to_hms,
)

HERE = Path(__file__).parent


@pytest.mark.parametrize("c, f", [(0, 32), (100, 212), (-40, -40), (37, 98.6)])
def test_celsius_to_fahrenheit(c, f):
    assert abs(celsius_to_fahrenheit(c) - f) < 1e-9


@pytest.mark.parametrize("total, expected", [
    (3725, (1, 2, 5)),
    (0, (0, 0, 0)),
    (59, (0, 0, 59)),
    (3600, (1, 0, 0)),
    (86399, (23, 59, 59)),
])
def test_seconds_to_hms(total, expected):
    assert seconds_to_hms(total) == expected


def test_receipt_line():
    assert receipt_line("Notebook", 45.5, 3) == "Notebook x3 = Rs 136.50"
    assert receipt_line("Pen", 10, 1) == "Pen x1 = Rs 10.00"
    assert receipt_line("Chai", 0.1, 3) == "Chai x3 = Rs 0.30"


def test_nearly_equal():
    assert nearly_equal(0.1 + 0.2, 0.3)
    assert nearly_equal(1.0, 1.0)
    assert not nearly_equal(1.0, 1.1)
    assert nearly_equal(1.0, 1.05, tolerance=0.1)
    assert nearly_equal(-0.5, -0.5 + 1e-12)


@pytest.mark.parametrize("n, expected", [(9045, 18), (0, 0), (7, 7), (10**20, 1), (999_999, 54)])
def test_digit_sum(n, expected):
    assert digit_sum(n) == expected


@pytest.mark.parametrize("base, exponent, expected", [
    (7, 3, 3),
    (2, 10, 4),
    (10, 5, 0),
    (3, 0, 1),
    (123456789, 2, 1),
])
def test_last_digit_of_power(base, exponent, expected):
    assert last_digit_of_power(base, exponent) == expected


def test_last_digit_of_huge_power():
    # A naive base ** exponent would never finish, so run it in a child process with a time limit.
    code = "from exercises import last_digit_of_power as f; print(f(7, 10**18), f(2, 10**18 + 1))"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=5, cwd=HERE,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Too slow: base ** exponent is far too big to compute. Look for a pattern in the last digits!")
    assert result.stdout.split() == ["1", "2"], result.stderr
