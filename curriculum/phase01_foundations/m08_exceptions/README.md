# m08 · Errors and exceptions

**By the end you can:** read a traceback in seconds, handle errors deliberately instead of crashing, raise clear errors from your own code, and build code that survives messy real-world input.

**Why it matters for AI:** real datasets are full of broken rows, missing values and corrupt files. API calls time out. GPUs run out of memory. Production AI code is mostly about failing gracefully. In exercise 5 you'll write a loader that keeps the good records and reports the bad ones, exactly what every data pipeline needs.

---

## 1. Reading a traceback

```
Traceback (most recent call last):
  File "train.py", line 12, in <module>
    avg = mean(scores)
  File "train.py", line 4, in mean
    return sum(xs) / len(xs)
ZeroDivisionError: division by zero
```

Read it **from the bottom up**:
1. The last line says *what* went wrong: the exception type and message.
2. The lines above show *where*: the innermost call last. Here, `mean` was called with an empty list.

**Tip:** paste the *whole* traceback when asking for help (including to your mentor). The bottom line alone often isn't enough.

## 2. Exceptions you'll meet constantly

| Exception | Typical cause |
|---|---|
| `NameError` | typo in a variable name |
| `TypeError` | wrong type: `"3" + 4`, wrong number of arguments |
| `ValueError` | right type, bad value: `int("abc")` |
| `IndexError` / `KeyError` | list index / dict key doesn't exist |
| `AttributeError` | `None.upper()`, often a function that returned `None` |
| `ZeroDivisionError` | dividing by 0 |
| `FileNotFoundError` | wrong path |

## 3. try / except / else / finally

```python
try:
    value = int(text)
except ValueError:
    print(f"{text!r} is not a number")
else:
    print("parsed", value)          # runs only if no exception happened
finally:
    print("always runs")            # cleanup: closing files, releasing locks
```

Catch **specific** exceptions. A bare `except:` or `except Exception:` also hides bugs you didn't expect, like typos.

```python
try:
    ratio = hits / total
except ZeroDivisionError:
    ratio = 0.0

try:
    ...
except (KeyError, IndexError) as e:   # several types; `e` is the exception object
    print(f"missing: {e}")
```

## 4. Raising exceptions

When your function gets input it can't handle, fail loudly and early with a clear message:

```python
def set_learning_rate(lr):
    if not 0 < lr < 1:
        raise ValueError(f"learning rate must be between 0 and 1, got {lr}")
    ...
```

Inside an `except` block, a bare `raise` re-raises the same exception after you've logged or cleaned up.

## 5. Your own exception types

```python
class InsufficientFunds(Exception):
    """Raised when a withdrawal is larger than the balance."""

raise InsufficientFunds("balance is 50, tried to withdraw 80")
```

Custom exceptions let callers catch *your* specific problem without catching everything else.

## 6. EAFP vs LBYL

Two styles for risky operations:

```python
# LBYL: Look Before You Leap
if key in config:
    value = config[key]

# EAFP: Easier to Ask Forgiveness than Permission (very Pythonic)
try:
    value = config[key]
except KeyError:
    value = default
```

EAFP avoids race conditions (a file can disappear between checking and opening) and is often clearer when failure is rare.

## 7. assert: for bugs, not for input

```python
assert len(batch) > 0, "batch should never be empty here"
```

Use `assert` to state things that must be true if *your* code is correct. Don't use it to validate user input: asserts are removed when Python runs with `-O`.

---

## Problem-solving habit #8: decide what "failure" means first

Before writing a function, decide what it does with bad input: return a default? return `None`? raise? skip and report? Write that decision in the docstring. Most bugs in data code come from never deciding.

## Go deeper (optional, research-level)

1. Exceptions are objects with a class hierarchy. Print `ZeroDivisionError.__mro__`. Why does `except ArithmeticError:` also catch `ZeroDivisionError`?
2. What's the difference between `raise NewError(...) from e` and raising without `from`? Look at the traceback each one produces ("The above exception was the direct cause..." vs "During handling...").
3. Python 3.11 added `ExceptionGroup` and `except*`. Read PEP 654. Why did async code (many tasks failing at once) need this?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
