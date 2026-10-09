"""m08 exercises: exceptions. Replace each `raise NotImplementedError` with your solution."""


# 1. Return a / b, or None if b is zero. Catch ZeroDivisionError specifically
#    (use try/except, not an if-check).
def safe_divide(a, b):
    raise NotImplementedError


# 2. Convert text to an int, or return `default` if it isn't a valid integer.
#    Surrounding spaces are fine. parse_int(" 42 ") -> 42, parse_int("4.2") -> None
def parse_int(text, default=None):
    raise NotImplementedError


# 3. Parse an age. Return it as an int, or raise ValueError with a helpful message if:
#    - it isn't a whole number           -> message contains "not a number"
#    - it's negative or greater than 150 -> message contains "out of range"
#    parse_age("30") -> 30; parse_age("abc") raises ValueError("'abc' is not a number")
def parse_age(text):
    raise NotImplementedError


# 4. Define a custom exception InsufficientFunds (a subclass of Exception), then
#    write withdraw(balance, amount) that returns the new balance. It raises:
#    - ValueError if amount <= 0
#    - InsufficientFunds if amount > balance
class InsufficientFunds(Exception):
    pass  # this one is already done for you


def withdraw(balance, amount):
    raise NotImplementedError


# 5. Parse CSV-like lines of the form "name,age". Return (records, bad_lines):
#    - records: list of (name, age) tuples for valid lines, names stripped, ages as int
#    - bad_lines: list of 1-based line numbers that were invalid
#    A line is invalid if it doesn't have exactly 2 fields, the name is empty,
#    or the age isn't a valid age (reuse parse_age!). Skip completely blank lines silently.
#    load_people(["Asha,21", "Ben,abc", "", "Chen, 30"])
#    -> ([("Asha", 21), ("Chen", 30)], [2])
def load_people(lines):
    raise NotImplementedError


# 6. Call func() up to `attempts` times. Return its result as soon as a call
#    succeeds. If every call raises, re-raise the LAST exception.
#    (Real code does this for flaky network calls, like an AI API that times out.)
def retry(func, attempts=3):
    raise NotImplementedError


# 7. Look up a value in nested dicts by a sequence of keys, EAFP style
#    (try/except, not `in` checks). Return None if any key is missing
#    or something along the way isn't a dict.
#    config = {"model": {"layers": {"hidden": 128}}}
#    get_nested(config, "model", "layers", "hidden") -> 128
#    get_nested(config, "model", "dropout") -> None
def get_nested(data, *keys):
    raise NotImplementedError
