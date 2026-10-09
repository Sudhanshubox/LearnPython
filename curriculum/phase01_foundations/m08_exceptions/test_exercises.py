import ast

import pytest

import exercises
from code_checks import contains, uses_any
from exercises import InsufficientFunds, get_nested, load_people, parse_age, parse_int, retry, safe_divide, withdraw


def test_safe_divide():
    assert safe_divide(10, 4) == 2.5
    assert safe_divide(1, 0) is None
    assert safe_divide(0, 5) == 0


def test_safe_divide_uses_try():
    assert contains(exercises.safe_divide, ast.Try)
    assert uses_any(exercises.safe_divide, "ZeroDivisionError")


@pytest.mark.parametrize("text, expected", [(" 42 ", 42), ("-7", -7), ("4.2", None), ("", None), ("abc", None)])
def test_parse_int(text, expected):
    assert parse_int(text) == expected


def test_parse_int_default():
    assert parse_int("x", default=0) == 0


@pytest.mark.parametrize("text, expected", [("30", 30), ("0", 0), ("150", 150), (" 7 ", 7)])
def test_parse_age_valid(text, expected):
    assert parse_age(text) == expected


@pytest.mark.parametrize("text, message", [
    ("abc", "not a number"), ("2.5", "not a number"), ("", "not a number"),
    ("-1", "out of range"), ("151", "out of range"),
])
def test_parse_age_invalid(text, message):
    with pytest.raises(ValueError, match=message):
        parse_age(text)


def test_withdraw():
    assert withdraw(100, 30) == 70
    assert withdraw(100, 100) == 0
    with pytest.raises(InsufficientFunds):
        withdraw(50, 80)
    with pytest.raises(ValueError):
        withdraw(50, 0)
    with pytest.raises(ValueError):
        withdraw(50, -5)


def test_insufficient_funds_is_an_exception():
    assert issubclass(InsufficientFunds, Exception)
    assert not issubclass(InsufficientFunds, ValueError)


def test_load_people():
    lines = ["Asha,21", "Ben,abc", "", "Chen, 30"]
    assert load_people(lines) == ([("Asha", 21), ("Chen", 30)], [2])


def test_load_people_edge_cases():
    lines = [" Dev , 40 ", ",20", "Eve,20,extra", "Fay,-3", "Gus", "   ", "Hal,151", "Ivy,0"]
    records, bad = load_people(lines)
    assert records == [("Dev", 40), ("Ivy", 0)]
    assert bad == [2, 3, 4, 5, 7]


def test_retry_succeeds_after_failures():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("try again")
        return "ok"

    assert retry(flaky, attempts=3) == "ok"
    assert len(calls) == 3


def test_retry_reraises_last_exception():
    calls = []

    def always_fails():
        calls.append(1)
        raise ConnectionError(f"failure {len(calls)}")

    with pytest.raises(ConnectionError, match="failure 4"):
        retry(always_fails, attempts=4)
    assert len(calls) == 4


def test_retry_returns_immediately_on_success():
    calls = []

    def works():
        calls.append(1)
        return 42

    assert retry(works) == 42
    assert len(calls) == 1


def test_get_nested():
    config = {"model": {"layers": {"hidden": 128}, "name": "tiny"}}
    assert get_nested(config, "model", "layers", "hidden") == 128
    assert get_nested(config, "model", "name") == "tiny"
    assert get_nested(config, "model", "dropout") is None
    assert get_nested(config, "model", "name", "x") is None
    assert get_nested(config) == config


def test_get_nested_is_eafp():
    assert contains(exercises.get_nested, ast.Try)
