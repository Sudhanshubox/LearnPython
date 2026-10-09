import inspect
import itertools
import json
import types

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    Countdown,
    chunked,
    data_loader,
    fibonacci,
    flatten,
    parse_jsonl,
    pipeline,
    running_mean,
    take,
    tokenize,
    valid_examples,
    windows,
)


def naturals():
    n = 0
    while True:
        yield n
        n += 1


@pytest.mark.parametrize("name", ["fibonacci", "take", "chunked", "windows", "running_mean", "flatten"])
def test_hand_written(name):
    assert not uses_any(getattr(exercises, name), "itertools", "islice", "batched", "accumulate")


@pytest.mark.parametrize("name", [
    "fibonacci", "take", "chunked", "windows", "running_mean", "flatten",
    "parse_jsonl", "valid_examples", "tokenize", "pipeline", "data_loader",
])
def test_returns_generators(name):
    func = getattr(exercises, name)
    args = {
        "fibonacci": (), "take": (2, [1, 2, 3]), "chunked": ([1], 1), "windows": ([1, 2], 1),
        "running_mean": ([1],), "flatten": ([1],), "parse_jsonl": ([],), "valid_examples": ([],),
        "tokenize": ([],), "pipeline": ([],), "data_loader": (4, 2, 1),
    }[name]
    assert isinstance(func(*args), types.GeneratorType), f"{name} should be lazy (a generator)"


def test_countdown():
    assert list(Countdown(3)) == [3, 2, 1]
    assert list(Countdown(0)) == []
    it = iter(Countdown(2))
    assert next(it) == 2 and next(it) == 1
    with pytest.raises(StopIteration):
        next(it)
    assert "yield" not in inspect.getsource(Countdown)


def test_fibonacci_and_take():
    assert list(take(10, fibonacci())) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert list(take(0, naturals())) == []
    assert list(take(5, [1, 2])) == [1, 2]


def test_take_does_not_over_consume():
    source = naturals()
    list(take(3, source))
    assert next(source) == 3, "take(3, ...) should read exactly 3 items"


def test_chunked():
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(chunked([], 3)) == []
    assert list(take(2, chunked(naturals(), 3))) == [[0, 1, 2], [3, 4, 5]]


def test_windows():
    assert list(windows([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]
    assert list(windows([1, 2], 3)) == []
    assert list(take(2, windows(naturals(), 2))) == [(0, 1), (1, 2)]


def test_running_mean():
    assert list(running_mean([2, 4, 6])) == [2.0, 3.0, 4.0]
    assert list(take(3, running_mean(naturals()))) == [0.0, 0.5, 1.0]


def test_flatten():
    assert list(flatten([1, [2, (3, [4])], "ab"])) == [1, 2, 3, 4, "ab"]
    assert list(flatten([])) == []
    assert "yield from" in inspect.getsource(exercises.flatten)


LINES = [
    '{"text": "Great Movie", "label": 1}',
    "",
    '{"text": "", "label": 0}',
    '{"text": "Terrible plot", "label": 0}',
    '{"text": "no label"}',
    '{"text": "Okay I guess", "label": "1"}',
]


def test_pipeline_stages():
    records = list(parse_jsonl(LINES))
    assert len(records) == 5
    assert [r["text"] for r in valid_examples(records)] == ["Great Movie", "Terrible plot"]
    assert list(tokenize([{"text": "Hi There", "label": 1}])) == [(["hi", "there"], 1)]


def test_pipeline_end_to_end():
    assert list(pipeline(LINES)) == [(["great", "movie"], 1), (["terrible", "plot"], 0)]


def test_pipeline_is_lazy():
    def endless():
        for i in itertools.count():
            yield json.dumps({"text": f"Example {i}", "label": i % 2})

    first = list(take(3, pipeline(endless())))
    assert first == [(["example", "0"], 0), (["example", "1"], 1), (["example", "2"], 0)]


def test_data_loader_covers_every_index_each_epoch():
    batches = list(data_loader(10, 3, epochs=2, seed=1))
    assert [e for e, _ in batches] == [0, 0, 0, 0, 1, 1, 1, 1]
    for epoch in (0, 1):
        idx = [i for e, b in batches if e == epoch for i in b]
        assert sorted(idx) == list(range(10))
    assert [len(b) for _, b in batches[:4]] == [3, 3, 3, 1]


def test_data_loader_shuffles_differently_each_epoch_and_is_reproducible():
    a = list(data_loader(20, 20, epochs=2, seed=7))
    assert a[0][1] != a[1][1]
    assert a == list(data_loader(20, 20, epochs=2, seed=7))
    import random
    expected = list(range(20))
    random.Random(7).shuffle(expected)
    assert a[0][1] == expected


def test_data_loader_drop_last():
    batches = list(data_loader(10, 3, epochs=1, drop_last=True))
    assert [len(b) for _, b in batches] == [3, 3, 3]
