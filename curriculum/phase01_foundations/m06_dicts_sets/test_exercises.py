import random
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    bigram_model,
    build_vocab,
    common,
    encode,
    first_unique,
    group_by_length,
    has_pair_with_sum,
    invert,
    most_likely_next,
    top_words,
    word_counts,
)


def test_word_counts():
    assert word_counts("The cat. The hat!") == {"the": 2, "cat": 1, "hat": 1}
    assert word_counts("") == {}
    assert word_counts("Wait; what? WAIT: what, wait!") == {"wait": 3, "what": 2}


def test_word_counts_by_hand():
    assert not uses_any(exercises.word_counts, "Counter", "defaultdict")


def test_top_words():
    assert top_words("b a b c a b", 2) == [("b", 3), ("a", 2)]
    assert top_words("z y x", 2) == [("x", 1), ("y", 1)]
    assert top_words("one", 5) == [("one", 1)]


def test_group_by_length():
    assert group_by_length(["hi", "cat", "yo", "dog"]) == {2: ["hi", "yo"], 3: ["cat", "dog"]}
    assert group_by_length([]) == {}


def test_invert():
    assert invert({"a": 1, "b": 2, "c": 1}) == {1: ["a", "c"], 2: ["b"]}
    assert invert({}) == {}


@pytest.mark.parametrize("a, b, expected", [
    ([3, 1, 2, 2], [2, 3, 5], [2, 3]),
    ([1, 2], [3, 4], []),
    (["b", "a"], ["a", "b", "a"], ["a", "b"]),
])
def test_common(a, b, expected):
    assert common(a, b) == expected


@pytest.mark.parametrize("nums, target, expected", [
    ([8, 3, 5, 1], 9, True),
    ([4], 8, False),
    ([4, 4], 8, True),
    ([1, 2, 3], 7, False),
    ([], 0, False),
    ([-3, 10, 7], 4, True),
])
def test_has_pair_with_sum(nums, target, expected):
    assert has_pair_with_sum(nums, target) == expected


def test_has_pair_with_sum_is_linear():
    nums = random.Random(1).sample(range(10**9), 20000)
    start = time.perf_counter()
    assert has_pair_with_sum(nums, -1) is False
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"took {elapsed:.2f}s; use a set to remember numbers you've seen"


@pytest.mark.parametrize("s, expected", [("leetcode", 0), ("loveleetcode", 2), ("aabb", -1), ("", -1), ("z", 0)])
def test_first_unique(s, expected):
    assert first_unique(s) == expected


def test_build_vocab_and_encode():
    vocab = build_vocab(["I like cats", "I like dogs"])
    assert vocab == {"<unk>": 0, "i": 1, "like": 2, "cats": 3, "dogs": 4}
    assert encode("I like birds", vocab) == [1, 2, 0]
    assert encode("", vocab) == []


def test_bigram_model():
    model = bigram_model(["i", "like", "cats", "i", "like", "dogs", "i", "run"])
    assert set(model) == {"i", "like", "cats", "dogs"}
    assert model["i"] == pytest.approx({"like": 2 / 3, "run": 1 / 3})
    assert model["like"] == pytest.approx({"cats": 0.5, "dogs": 0.5})
    assert model["cats"] == {"i": 1.0}
    for followers in model.values():
        assert sum(followers.values()) == pytest.approx(1.0)


def test_most_likely_next():
    model = bigram_model(["i", "like", "cats", "i", "like", "dogs", "i", "run"])
    assert most_likely_next(model, "i") == "like"
    assert most_likely_next(model, "like") == "cats"
    assert most_likely_next(model, "run") is None
    assert most_likely_next(model, "zebra") is None
