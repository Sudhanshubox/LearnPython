"""m27 exercises: dataclasses, enums and type hints.

Annotate EVERY function and method you write (parameters and return type):
the last test runs mypy with --disallow-untyped-defs on this file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import NamedTuple, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
K = TypeVar("K")


# 1. A FROZEN dataclass TrainingConfig with these fields and defaults:
#      model_name: str = "tiny"
#      lr: float = 0.001
#      batch_size: int = 32
#      epochs: int = 10
#      hidden_sizes: list[int] = [64, 64]   (each config gets its own list!)
#    - __post_init__ raises ValueError if lr <= 0, batch_size < 1 or epochs < 1.
#    - with_changes(**changes) returns a NEW config with those fields changed
#      (use dataclasses.replace; validation must run again).
#    - to_dict() returns a plain dict of all fields (dataclasses.asdict).
#    Replace `pass` with the fields and methods, and add the decorator.
class TrainingConfig:
    pass


# 2. An Enum Split with members TRAIN = "train", VALIDATION = "validation", TEST = "test".
#    Then parse_split(text) returns the member for a value in any case and with
#    surrounding spaces ignored: parse_split(" Test ") -> Split.TEST.
#    Also accept the short forms "val" and "valid" for VALIDATION.
#    Anything else raises ValueError.
class Split(Enum):
    pass


def parse_split(text: str) -> Split:
    raise NotImplementedError


# 3. A Prediction NamedTuple with fields label (str) and score (float).
#    best(predictions) returns the Prediction with the highest score, or None if
#    the list is empty.
class Prediction(NamedTuple):
    label: str
    score: float


def best(predictions: list[Prediction]) -> Prediction | None:
    raise NotImplementedError


# 4. A dataclass Task with fields priority (int) and name (str) that can be SORTED
#    (lower priority first, then by name). Use @dataclass(order=True).
class Task:
    pass


# 5. A generic function. group_by(items, key) returns a dict from each key(item) to
#    the list of items with that key, in their original order. Keep the annotations
#    given: they let mypy know the dict's key and value types.
def group_by(items: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    raise NotImplementedError


# 6. A Protocol for "anything with predict(x: float) -> float". Make it
#    runtime_checkable. Then evaluate(model, xs, ys) returns the mean squared error of
#    the model's predictions. Any class with a matching predict method must work,
#    without inheriting from SupportsPredict.
@runtime_checkable
class SupportsPredict(Protocol):
    def predict(self, x: float) -> float: ...


def evaluate(model: SupportsPredict, xs: list[float], ys: list[float]) -> float:
    raise NotImplementedError
