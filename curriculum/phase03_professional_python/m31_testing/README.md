# m31 · Testing with pytest

**By the end you can:** write clear, thorough tests with pytest (assertions, `pytest.raises`, `pytest.approx`, parametrize, fixtures, `tmp_path`, `monkeypatch`), choose test cases that actually find bugs, and measure how good a test suite is.

**Why it matters for AI:** ML code fails *silently*: a wrong normalization or an off-by-one in a tokenizer won't crash; it just makes the model worse. Tests on the deterministic parts (data processing, metrics, tokenizers, configs) are your main defence. Every serious AI library (PyTorch, transformers, scikit-learn) has thousands of tests, and you can't contribute to them without writing tests yourself.

**This module flips the usual setup:** the code is already written (`shop.py`), and *you* write the tests, in `exercises.py`.

---

## 1. A test is a function that asserts

```python
# exercises.py
from shop import apply_discount

def test_ten_percent_off():
    assert apply_discount(200, 10) == 180
```

pytest finds functions named `test_*`, runs them, and reports any `assert` that fails, showing the values on both sides. Run it from a terminal with `pytest exercises.py -v`.

**Arrange, act, assert:** set up the input, call the code, check the result. One behaviour per test, with a name that says which one.

## 2. Testing errors and floats

```python
import pytest

def test_rejects_negative_percent():
    with pytest.raises(ValueError):
        apply_discount(100, -5)

def test_float_result():
    assert 0.1 + 0.2 == pytest.approx(0.3)      # never compare floats with == (m01)
```

`pytest.raises(ValueError, match="percent")` also checks the message.

## 3. Parametrize: many cases, one test

```python
@pytest.mark.parametrize("price, percent, expected", [
    (100, 0, 100),
    (100, 100, 0),
    (19.99, 15, 16.99),
])
def test_discount_cases(price, percent, expected):
    assert apply_discount(price, percent) == expected
```

Each row runs as a separate test, and failures tell you exactly which row broke. You've seen this pattern in every module so far: the tests you've been passing were written this way.

## 4. Fixtures: reusable setup

```python
@pytest.fixture
def stocked():
    inv = Inventory()
    inv.add("pen", 10)
    return inv

def test_remove_reduces_quantity(stocked):     # pytest passes the fixture's result in
    stocked.remove("pen", 3)
    assert stocked.quantity("pen") == 7
```

Each test gets a **fresh** fixture, so tests can't affect each other. pytest also has built-in fixtures:

- `tmp_path`: a new empty folder for each test (for file I/O).
- `monkeypatch`: temporarily replace functions, attributes or environment variables (for example, to fake the current time or a network call).
- `capsys`: capture what the code prints.

## 5. Which cases should you test?

Good test cases come from thinking about where code typically breaks:

- **Normal** cases: the typical use.
- **Boundaries**: 0, 1, the exact limit (100%), one past it, empty input.
- **Invalid input**: negative, wrong type, malformed text. Does it raise the *documented* error?
- **Properties**: things that must always hold, like "the parts of a split bill add up to the total", or "save then load gives back the same data" (a **round trip**).
- **Exact rules from the spec**: who gets the extra cent? What's rounded, and how?

A test suite that only checks the happy path will pass on almost any buggy version.

## 6. Measuring test quality: mutation testing

How do you know your tests are good? **Coverage** (which lines ran) is a weak signal: a test can run a line without checking its result. **Mutation testing** is stronger: make small bugs ("mutants") in the code and see whether your tests notice. If a mutant survives, a bug like it could slip through.

That's exactly how this module is graded: `test_exercises.py` runs your tests against the real `shop.py` (they must all pass) and against 12 mutants, each with one realistic bug (they must each make at least one of your tests fail). Try not to read the mutants in `test_exercises.py` before you're done: designing tests that catch *unknown* bugs is the skill.

## 7. Test-driven development (TDD)

An effective habit: write a failing test for the next small behaviour, write just enough code to pass it, clean up, repeat. Tests then describe exactly what the code is supposed to do, and you never have untested code.

---

## Problem-solving habit #29: read the spec as a list of test cases

Every sentence in a docstring is a claim you can test: "percent must be between 0 and 100" → tests at -1, 0, 100 and 101. "Remainder cents go to the first people" → a test with a remainder. Turning specs into tests is also the fastest way to find out a spec is ambiguous.

## Go deeper (optional, research-level)

1. Install `hypothesis` and write a **property-based** test: `@given(st.integers(min_value=1), st.integers(min_value=1, max_value=50))` for `split_bill`, asserting the parts always sum to the total. Hypothesis generates hundreds of inputs and shrinks failures to the smallest example.
2. Run `pip install pytest-cov` then `pytest --cov=shop exercises.py`. Do you have 100% line coverage? Construct a buggy version of `shop.py` that your tests *don't* catch despite 100% coverage.
3. How do people test ML models, where outputs aren't exact? Read about invariance tests and directional expectation tests in *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList* (Ribeiro et al., 2020).

## Your turn

Read `shop.py` carefully (use the file picker), then write tests in `exercises.py`. The grading also checks that you used `@pytest.mark.parametrize`, a fixture, `pytest.raises`, `pytest.approx` and `tmp_path`.
