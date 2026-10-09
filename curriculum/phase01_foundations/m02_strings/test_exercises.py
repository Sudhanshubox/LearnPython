import pytest

from exercises import caesar, count_vowels, initials, is_palindrome, rle_decode, rle_encode, text_size


@pytest.mark.parametrize("name, expected", [
    ("ada lovelace", "A.L."),
    ("  guido   van rossum ", "G.V.R."),
    ("Linus", "L."),
])
def test_initials(name, expected):
    assert initials(name) == expected


@pytest.mark.parametrize("text, expected", [
    ("A man, a plan, a canal: Panama", True),
    ("racecar", True),
    ("No 'x' in Nixon", True),
    ("hello", False),
    ("", True),
    ("ab", False),
])
def test_is_palindrome(text, expected):
    assert is_palindrome(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Programming in Python", 5),
    ("AEIOU aeiou", 10),
    ("rhythm", 0),
    ("", 0),
])
def test_count_vowels(text, expected):
    assert count_vowels(text) == expected


@pytest.mark.parametrize("text, shift, expected", [
    ("Hello, World!", 3, "Khoor, Zruog!"),
    ("abc", -1, "zab"),
    ("xyz", 3, "abc"),
    ("Python 3.13", 26, "Python 3.13"),
    ("Mixed Case", 52 + 1, "Njyfe Dbtf"),
])
def test_caesar(text, shift, expected):
    assert caesar(text, shift) == expected


def test_caesar_round_trip():
    message = "Attack at dawn! 123"
    assert caesar(caesar(message, 7), -7) == message


@pytest.mark.parametrize("text, encoded", [
    ("aaabccdddd", "a3b1c2d4"),
    ("", ""),
    ("a", "a1"),
    ("abab", "a1b1a1b1"),
    ("z" * 12, "z12"),
])
def test_rle_encode(text, encoded):
    assert rle_encode(text) == encoded


@pytest.mark.parametrize("encoded, text", [
    ("a3b1c2d4", "aaabccdddd"),
    ("x12", "x" * 12),
    ("", ""),
    ("a1b10c2", "a" + "b" * 10 + "cc"),
])
def test_rle_decode(encoded, text):
    assert rle_decode(encoded) == text


@pytest.mark.parametrize("text, expected", [
    ("hi", (2, 2)),
    ("नमस्ते", (6, 18)),
    ("café", (4, 5)),
    ("👍", (1, 4)),
    ("", (0, 0)),
])
def test_text_size(text, expected):
    assert text_size(text) == expected
