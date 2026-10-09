# m29 · Decorators and context managers

**By the end you can:** write decorators (with and without arguments) that add behaviour to functions without touching their code, and context managers that guarantee setup and cleanup happen, even when errors occur.

**Why it matters for AI:** you've already used decorators: `@property`, `@dataclass`, `@cache`. PyTorch uses `@torch.no_grad()` to switch off gradient tracking, and `with torch.autocast(...)` for mixed precision. Experiment code uses decorators to time functions, retry flaky API calls, cache expensive results and register models by name. Context managers manage files, GPU memory, random seeds and database connections.

---

## 1. Decorators are functions that wrap functions

Functions are values (m07), so a function can take a function and return a new one:

```python
import functools
import time

def timed(func):
    @functools.wraps(func)                  # copy the name and docstring onto the wrapper
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)      # call the original
        print(f"{func.__name__} took {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timed
def train():
    ...
```

`@timed` above `def train` means exactly `train = timed(train)`. The name `train` now refers to `wrapper`, which calls the original `train` inside.

- `*args, **kwargs` let the wrapper accept and pass on **any** arguments.
- **Always** use `functools.wraps`; without it, `train.__name__` becomes `"wrapper"`, which confuses debugging, logging and documentation tools.

## 2. Decorators with state

The wrapper is a closure, and functions can have attributes:

```python
def count_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper
```

## 3. Decorators with arguments

`@retry(times=3)` needs **three** levels: a function that takes the arguments and returns a decorator, which takes the function and returns the wrapper.

```python
def retry(times):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == times - 1:
                        raise
        return wrapper
    return decorator

@retry(times=3)
def call_api(): ...
```

`@retry(times=3)` means `call_api = retry(times=3)(call_api)`.

## 4. A registry decorator

A decorator doesn't have to wrap; it can just record the function and return it unchanged:

```python
ACTIVATIONS = {}

def register(name):
    def decorator(func):
        ACTIVATIONS[name] = func
        return func
    return decorator

@register("relu")
def relu(x):
    return max(0.0, x)

ACTIVATIONS["relu"](-2)        # look it up by the name in a config file
```

Hugging Face, timm and many training frameworks register models, losses and optimizers exactly like this.

## 5. Context managers: guaranteed cleanup

```python
with open("data.txt", encoding="utf-8") as f:
    ...
# the file is closed here, even if the block raised an exception
```

A context manager is any object with `__enter__` and `__exit__`:

```python
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self                         # bound to the name after "as"

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        return False                        # False: don't swallow exceptions

with Timer() as t:
    train()
print(t.elapsed)
```

`__exit__` receives the exception (if any). Returning `True` suppresses it; returning `False` lets it propagate. Suppress only exceptions you've decided to handle.

## 6. contextlib.contextmanager: the easy way

Write a generator: everything before `yield` is setup, after is cleanup. Put the cleanup in `finally` so it always runs:

```python
from contextlib import contextmanager
import os

@contextmanager
def working_directory(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)        # runs even if the with-block raised
```

## 7. Reproducibility: a seed context

Machine-learning results depend on random numbers (weight initialization, shuffling, dropout). To make an experiment reproducible, you fix the **seed**. A context manager can fix it temporarily and then restore the previous random state so the rest of the program isn't affected. That's exercise 9.

---

## Problem-solving habit #27: separate the "what" from the "around"

Timing, logging, retrying, caching, permission checks and cleanup are concerns that wrap around many different functions. When you notice the same before/after code copied into several functions, move it into a decorator or context manager, so each function contains only its own logic.

## Go deeper (optional, research-level)

1. Class decorators receive a class instead of a function: `@dataclass` is one. Write `@add_repr` that adds a `__repr__` to any class.
2. `contextlib.ExitStack` manages a *variable* number of context managers (for example, opening N files). Read its docs and think about where a training script might need it.
3. Read how `torch.no_grad` works: it's usable both as a decorator and as a context manager. How can one object be both? (Hint: `contextlib.ContextDecorator`.)

## Your turn

Open the **Exercises** tab, solve each part, then press **Run tests**.
