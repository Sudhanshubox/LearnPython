"""m11 project: a command-line expense tracker.

Implement the functions in order; the tests check each one separately.
When everything passes, try it: python exercises.py add 250 food 2026-10-01 "lunch"
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------- data model

# 1. Create and validate one expense. Return a dict:
#    {"amount": float, "category": str, "date": str, "note": str}
#    - amount must be a number > 0, otherwise raise ValueError
#    - category is stripped and lowercased; it can't be empty (ValueError)
#    - date must be a valid "YYYY-MM-DD" date (use date.fromisoformat), else ValueError
#    - note is stripped
#    make_expense(250, " Food ", "2026-10-01", "lunch ")
#    -> {"amount": 250.0, "category": "food", "date": "2026-10-01", "note": "lunch"}
def make_expense(amount, category, date_text, note=""):
    raise NotImplementedError


# ---------------------------------------------------------------- analysis

# 2. Total amount, optionally only for one category and/or one month ("YYYY-MM").
#    Round the result to 2 decimal places.
def total(expenses, category=None, month=None):
    raise NotImplementedError


# 3. Total per category, as a dict ordered from the largest total to the smallest
#    (ties: alphabetical). Values rounded to 2 decimals.
#    -> {"travel": 1200.0, "food": 340.0}
def by_category(expenses):
    raise NotImplementedError


# 4. Total per month, as a dict ordered by month (oldest first). Values rounded.
#    -> {"2026-10": 1450.0, "2026-11": 90.0}
def by_month(expenses):
    raise NotImplementedError


# ---------------------------------------------------------------- input/output

# 5. Save the list of expenses as JSON (indent=2). load() returns the list,
#    or [] if the file doesn't exist yet.
def save(path, expenses):
    raise NotImplementedError


def load(path):
    raise NotImplementedError


# 6. Build the report text shown in the README:
#    - a header line: "Category" left-aligned in 10 characters, then "Total" right-aligned in 13
#    - one line per category (largest first): name left-aligned in 10, amount right-aligned
#      in 13 with 2 decimals
#    - a line of 23 dashes
#    - "TOTAL" left-aligned in 10, the grand total right-aligned in 13 with 2 decimals
#    Lines are joined with "\n". For no expenses, return "No expenses yet."
#    Hint: f"{name:<10}{amount:>13.2f}"
def format_report(expenses):
    raise NotImplementedError


# ---------------------------------------------------------------- command line

# 7. The command-line interface. `argv` is the list of arguments (like sys.argv[1:]).
#    Commands:
#      add AMOUNT CATEGORY DATE [NOTE]   validate with make_expense, append, save,
#                                        print "Added: DATE  CATEGORY  Rs AMOUNT  NOTE"
#                                        (amount with 2 decimals; strip trailing spaces)
#      report [--month YYYY-MM]          print format_report for (that month's) expenses
#    Option before the command:  --file PATH  (default "expenses.json")
#    Return 0 on success. If make_expense raises ValueError, print "Error: <message>"
#    and return 1 without saving anything.
def main(argv):
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
