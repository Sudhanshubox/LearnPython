import inspect

import pytest

import exercises
from exercises import day_type, fizzbuzz, grade, is_leap_year, largest, outcome, triangle_type


@pytest.mark.parametrize("score, expected", [
    (100, "A"), (90, "A"), (89.9, "B"), (80, "B"), (79, "C"), (70, "C"),
    (69, "D"), (60, "D"), (59, "F"), (0, "F"), (-1, "invalid"), (101, "invalid"),
])
def test_grade(score, expected):
    assert grade(score) == expected


@pytest.mark.parametrize("year, expected", [
    (2000, True), (1900, False), (2024, True), (2023, False), (2100, False), (2400, True),
])
def test_is_leap_year(year, expected):
    assert is_leap_year(year) == expected


@pytest.mark.parametrize("n, expected", [
    (1, "1"), (3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz"), (30, "FizzBuzz"), (98, "98"), (99, "Fizz"),
])
def test_fizzbuzz(n, expected):
    assert fizzbuzz(n) == expected


@pytest.mark.parametrize("sides, expected", [
    ((3, 3, 3), "equilateral"),
    ((3, 3, 5), "isosceles"),
    ((5, 3, 3), "isosceles"),
    ((3, 5, 3), "isosceles"),
    ((3, 4, 5), "scalene"),
    ((1, 2, 3), "invalid"),
    ((1, 1, 5), "invalid"),
    ((0, 0, 0), "invalid"),
    ((-3, 4, 5), "invalid"),
])
def test_triangle_type(sides, expected):
    assert triangle_type(*sides) == expected


@pytest.mark.parametrize("predicted, actual, expected", [
    (True, True, "TP"), (True, False, "FP"), (False, False, "TN"), (False, True, "FN"),
])
def test_outcome(predicted, actual, expected):
    assert outcome(predicted, actual) == expected


@pytest.mark.parametrize("nums, expected", [
    ((1, 2, 3), 3), ((3, 2, 1), 3), ((2, 3, 1), 3), ((5, 5, 1), 5), ((-1, -2, -3), -1), ((2.5, 2.4, 2.45), 2.5),
])
def test_largest(nums, expected):
    assert largest(*nums) == expected


def test_largest_does_not_use_max_or_sorted():
    source = inspect.getsource(exercises.largest)
    assert "max(" not in source and "sorted(" not in source and ".sort(" not in source


@pytest.mark.parametrize("day, expected", [
    ("Saturday", "weekend"), ("sunday", "weekend"), ("MONDAY", "weekday"),
    ("Wednesday", "weekday"), ("friday", "weekday"), ("Funday", "invalid"), ("", "invalid"),
])
def test_day_type(day, expected):
    assert day_type(day) == expected


def test_day_type_uses_match():
    assert "match " in inspect.getsource(exercises.day_type)
