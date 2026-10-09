import dataclasses
import subprocess
import sys
import typing
from pathlib import Path

import pytest

import exercises
from exercises import Prediction, Split, SupportsPredict, Task, TrainingConfig, best, evaluate, group_by, parse_split

HERE = Path(__file__).parent

# ---------------------------------------------------------------- TrainingConfig


def test_config_is_a_frozen_dataclass():
    assert dataclasses.is_dataclass(TrainingConfig)
    cfg = TrainingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.lr = 0.5


def test_config_defaults():
    cfg = TrainingConfig()
    assert (cfg.model_name, cfg.lr, cfg.batch_size, cfg.epochs, cfg.hidden_sizes) == ("tiny", 0.001, 32, 10, [64, 64])
    assert TrainingConfig(lr=0.01) == TrainingConfig(lr=0.01)
    assert "lr=0.01" in repr(TrainingConfig(lr=0.01))


def test_config_lists_not_shared():
    a, b = TrainingConfig(), TrainingConfig()
    a.hidden_sizes.append(128)
    assert b.hidden_sizes == [64, 64]


@pytest.mark.parametrize("bad", [{"lr": 0}, {"lr": -1}, {"batch_size": 0}, {"epochs": 0}])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        TrainingConfig(**bad)


def test_config_with_changes_and_to_dict():
    cfg = TrainingConfig()
    new = cfg.with_changes(lr=0.1, epochs=3)
    assert (new.lr, new.epochs, cfg.lr) == (0.1, 3, 0.001)
    with pytest.raises(ValueError):
        cfg.with_changes(lr=-5)
    assert cfg.to_dict() == {
        "model_name": "tiny", "lr": 0.001, "batch_size": 32, "epochs": 10, "hidden_sizes": [64, 64],
    }


# ---------------------------------------------------------------- Split


def test_split_enum():
    assert [s.value for s in Split] == ["train", "validation", "test"]
    assert Split("train") is Split.TRAIN


@pytest.mark.parametrize("text, expected", [
    ("train", "TRAIN"), (" Test ", "TEST"), ("VALIDATION", "VALIDATION"), ("val", "VALIDATION"), ("valid", "VALIDATION"),
])
def test_parse_split(text, expected):
    assert parse_split(text) is Split[expected]


@pytest.mark.parametrize("text", ["training", "", "dev"])
def test_parse_split_rejects(text):
    with pytest.raises(ValueError):
        parse_split(text)


# ---------------------------------------------------------------- Prediction & Task


def test_best():
    preds = [Prediction("cat", 0.2), Prediction("dog", 0.7), Prediction("cow", 0.1)]
    assert best(preds) == Prediction("dog", 0.7)
    assert best([]) is None


def test_task_sorting():
    assert dataclasses.is_dataclass(Task)
    tasks = [Task(2, "write"), Task(1, "test"), Task(1, "plan")]
    assert [t.name for t in sorted(tasks)] == ["plan", "test", "write"]


# ---------------------------------------------------------------- generics & protocols


def test_group_by():
    words = ["apple", "bob", "avocado", "cat", "banana"]
    assert group_by(words, lambda w: w[0]) == {"a": ["apple", "avocado"], "b": ["bob", "banana"], "c": ["cat"]}
    assert group_by(range(6), lambda n: n % 2) == {0: [0, 2, 4], 1: [1, 3, 5]}
    assert group_by([], len) == {}


def test_protocol_duck_typing():
    class Doubler:                      # does NOT inherit from SupportsPredict
        def predict(self, x: float) -> float:
            return 2 * x

    class NotAModel:
        pass

    assert isinstance(Doubler(), SupportsPredict)
    assert not isinstance(NotAModel(), SupportsPredict)
    assert evaluate(Doubler(), [1, 2, 3], [2, 4, 7]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- types


@pytest.mark.timeout(120)
def test_mypy_passes():
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--disallow-untyped-defs", "--no-incremental", "--ignore-missing-imports", "exercises.py"],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 0, "mypy found problems:\n" + result.stdout
