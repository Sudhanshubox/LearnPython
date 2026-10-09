# m11 · Project: command-line expense tracker

**This is your first real program.** No new concepts: you'll combine everything from m01–m10 (strings, conditionals, loops, lists, dicts, functions, exceptions, files, JSON) into a tool you can actually use every day.

**What you'll build:**

```bash
$ python exercises.py add 250 food 2026-10-01 "lunch with team"
Added: 2026-10-01  food  Rs 250.00  lunch with team

$ python exercises.py add 1200 travel 2026-10-03 "train to Pune"
$ python exercises.py add 90 food 2026-11-02

$ python exercises.py report
Category          Total
travel          1200.00
food             340.00
-----------------------
TOTAL           1540.00

$ python exercises.py report --month 2026-10
Category          Total
travel          1200.00
food             250.00
-----------------------
TOTAL           1450.00
```

Expenses are saved to `expenses.json` (or any file given with `--file`), so they're still there tomorrow.

---

## How to approach a project

Professional engineers don't start by writing `main()`. They build and test small pieces, then connect them:

1. **Data model first.** One expense is a dict: `{"amount": 250.0, "category": "food", "date": "2026-10-01", "note": "lunch"}`. The whole tracker is a list of these.
2. **Pure functions next.** `make_expense`, `total`, `by_category` and `by_month` take data and return data. No printing, no files. These are easy to test.
3. **Input/output last.** `save`, `load` and `format_report` touch the outside world. `main` just wires everything together.

`exercises.py` is laid out in this order. The tests check each piece separately, so you can make progress one function at a time.

## New tools you'll need

**Dates.** Python's `datetime` module can validate a date string for you:

```python
from datetime import date

date.fromisoformat("2026-10-01")   # OK, returns a date object
date.fromisoformat("2026-13-01")   # ValueError: month must be in 1..12
"2026-10-01"[:7]                    # "2026-10": the month part, handy for grouping
```

**Command-line arguments.** `sys.argv` holds the words typed after `python`. The `argparse` module turns them into a structured object and writes the `--help` text for you:

```python
import argparse

parser = argparse.ArgumentParser(description="Track expenses")
parser.add_argument("--file", default="expenses.json")
sub = parser.add_subparsers(dest="command", required=True)

add = sub.add_parser("add")
add.add_argument("amount", type=float)
add.add_argument("category")
add.add_argument("date")
add.add_argument("note", nargs="?", default="")   # optional positional

args = parser.parse_args(["add", "250", "food", "2026-10-01"])
args.command, args.amount   # ('add', 250.0)
```

**Money and floats.** Remember m01: `0.1 + 0.2 != 0.3`. Round money to 2 decimal places when you total it: `round(x, 2)`. (Real banking software uses integer cents or `decimal.Decimal`.)

## Requirements

Implement every function in `exercises.py`. Its comments are the specification, and the tests check every rule. When all tests pass, try the tool for real from a terminal in this folder.

## Stretch goals (optional)

Once the tests pass, extend it. The tests don't check these, so ask your mentor to review your design:

- `list` command with `--category` and `--month` filters
- `delete <index>` command
- monthly budget per category, with a warning when you go over
- a simple text bar chart in the report: `food   ████████ 340.00`
- export to CSV

## Go deeper (optional, research-level)

1. Your `save` overwrites `expenses.json` directly. If the program crashes halfway through writing, your data is gone. Implement "write to a temporary file, then `os.replace`" (m09 go-deeper) and explain why it's safe.
2. Compare storing expenses in JSON vs SQLite (`import sqlite3`, built in). At how many expenses does reading the whole JSON file on every command become slow? Measure it.
