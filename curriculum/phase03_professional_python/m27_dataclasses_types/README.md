# m27 · Dataclasses, enums and type hints

**By the end you can:** define data-holding classes in a few lines with `@dataclass`, replace "magic strings" with enums, annotate your code with type hints (including generics and protocols), and let `mypy` find bugs before you run anything.

**Why it matters for AI:** ML code is full of configuration (learning rate, batch size, model sizes) and records (examples, predictions, metrics). Dataclasses make them clean and safe. Libraries like Hugging Face `transformers` define every config as a dataclass, and PyTorch, FastAPI and Pydantic rely on type hints heavily. Type checking catches the shape and type mix-ups that otherwise only show up an hour into a training run.

---

## 1. Dataclasses

Writing `__init__`, `__repr__` and `__eq__` by hand for a class that just holds data is boring and error-prone. `@dataclass` writes them for you from type-annotated fields:

```python
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 32
    layers: list[int] = field(default_factory=lambda: [64, 64])   # NOT layers = [64, 64]!

cfg = TrainingConfig(lr=0.01)
cfg                                    # TrainingConfig(lr=0.01, batch_size=32, layers=[64, 64])
cfg == TrainingConfig(lr=0.01)         # True
```

`field(default_factory=...)` exists for the same reason as m07's mutable-default trap: each instance needs its own new list.

**Validation** goes in `__post_init__`, which runs right after the generated `__init__`:

```python
    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be positive")
```

**Options:**

- `@dataclass(frozen=True)`: immutable (assigning raises an error) and hashable. Great for configs: nobody can change the learning rate halfway through.
- `@dataclass(order=True)`: adds `<`, `>` etc., comparing fields in order.
- `dataclasses.replace(cfg, lr=0.1)`: a copy with some fields changed (the way to "modify" a frozen dataclass).
- `dataclasses.asdict(cfg)`: convert to a dict, for example to save as JSON.

## 2. Enums: named constants

```python
from enum import Enum

class Split(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

Split.TRAIN.value          # "train"
Split("test")              # Split.TEST: look up by value
Split["TRAIN"]             # look up by name
list(Split)                # all members
```

With plain strings, `"valdiation"` (a typo) silently goes through. With an enum, `Split.VALDIATION` fails immediately, and your editor can autocomplete the options.

## 3. Type hints

```python
def mean(values: list[float]) -> float:
    return sum(values) / len(values)

name: str = "Asha"
scores: dict[str, int] = {}
maybe: int | None = None          # "int or None" (Python 3.10+)
```

Python **ignores** type hints at runtime; they're documentation that tools can check. Common types:

| Hint | Meaning |
|---|---|
| `list[int]`, `dict[str, float]`, `tuple[int, int]`, `set[str]` | containers |
| `X \| None` | optional |
| `Iterable[T]`, `Sequence[T]`, `Mapping[K, V]` (from `collections.abc`) | "anything you can loop over / index / look up", more flexible for parameters |
| `Callable[[int, str], bool]` | a function taking (int, str) and returning bool |
| `Any` | opt out of checking |

## 4. Generics: types with type parameters

How do you say "returns the same type as the items in the list"?

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

first([1, 2])        # the checker knows this is an int
first(["a"])         # ... and this is a str
```

(Python 3.12 also allows `def first[T](items: list[T]) -> T:`.)

## 5. Protocols: type-checked duck typing

```python
from typing import Protocol

class SupportsPredict(Protocol):
    def predict(self, x: float) -> float: ...

def evaluate(model: SupportsPredict, xs: list[float], ys: list[float]) -> float:
    ...
```

Any object with a matching `predict` method is accepted, no inheritance needed (m25's duck typing, now checked by tools). Add `@runtime_checkable` to allow `isinstance(obj, SupportsPredict)`.

## 6. Checking with mypy

```bash
pip install mypy
mypy exercises.py
```

```python
def area(w: float, h: float) -> float:
    return w * h

area("3", 4)       # mypy: Argument 1 to "area" has incompatible type "str"; expected "float"
```

mypy found the bug without running the code. Start by annotating function signatures; mypy infers most types inside functions.

**Tip:** in VS Code, the Pylance extension type-checks as you type.

---

## Problem-solving habit #25: make illegal states unrepresentable

Design types so invalid data can't even be created: an enum instead of a free string, a frozen dataclass that validates in `__post_init__`, `int | None` instead of "use -1 to mean missing". Every bug ruled out by the types is a bug you never debug.

## Go deeper (optional, research-level)

1. Read PEP 484 (type hints) and PEP 544 (protocols). Why did Python choose *gradual* typing instead of making types mandatory?
2. Pydantic validates data at **runtime** using type hints (FastAPI uses it for every API request). How is that different from what mypy does? When do you need each?
3. Look at how Hugging Face defines `TrainingArguments` (a dataclass with about 100 fields). How do they turn it into command-line arguments automatically? (Hint: `HfArgumentParser` reads the dataclass fields.)

## Your turn

Open the **Exercises** tab, solve each part, then press **Run tests**. The last test runs `mypy` on your file, so annotate everything you write.
