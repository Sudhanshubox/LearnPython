import json

import pytest

from exercises import by_category, by_month, format_report, load, main, make_expense, save, total

SAMPLE = [
    {"amount": 250.0, "category": "food", "date": "2026-10-01", "note": "lunch"},
    {"amount": 1200.0, "category": "travel", "date": "2026-10-03", "note": "train"},
    {"amount": 90.0, "category": "food", "date": "2026-11-02", "note": ""},
]


# ---------------------------------------------------------------- make_expense

def test_make_expense_normalizes():
    assert make_expense(250, " Food ", "2026-10-01", "lunch ") == {
        "amount": 250.0, "category": "food", "date": "2026-10-01", "note": "lunch",
    }
    assert make_expense(9.5, "tea", "2026-01-31")["note"] == ""
    assert isinstance(make_expense(5, "x", "2026-01-01")["amount"], float)


@pytest.mark.parametrize("amount, category, date_text", [
    (0, "food", "2026-10-01"),
    (-5, "food", "2026-10-01"),
    (10, "   ", "2026-10-01"),
    (10, "food", "2026-13-01"),
    (10, "food", "01-10-2026"),
    (10, "food", "yesterday"),
])
def test_make_expense_rejects_bad_input(amount, category, date_text):
    with pytest.raises(ValueError):
        make_expense(amount, category, date_text)


# ---------------------------------------------------------------- analysis

def test_total():
    assert total(SAMPLE) == 1540.0
    assert total(SAMPLE, category="food") == 340.0
    assert total(SAMPLE, month="2026-10") == 1450.0
    assert total(SAMPLE, category="food", month="2026-11") == 90.0
    assert total(SAMPLE, category="rent") == 0
    assert total([]) == 0


def test_total_rounds_money():
    tiny = [make_expense(0.1, "x", "2026-01-01"), make_expense(0.2, "x", "2026-01-01")]
    assert total(tiny) == 0.3


def test_by_category():
    result = by_category(SAMPLE)
    assert result == {"travel": 1200.0, "food": 340.0}
    assert list(result) == ["travel", "food"]


def test_by_category_ties_alphabetical():
    data = [make_expense(5, "b", "2026-01-01"), make_expense(5, "a", "2026-01-01")]
    assert list(by_category(data)) == ["a", "b"]


def test_by_month():
    data = SAMPLE + [make_expense(10, "food", "2025-12-31")]
    result = by_month(data)
    assert result == {"2025-12": 10.0, "2026-10": 1450.0, "2026-11": 90.0}
    assert list(result) == ["2025-12", "2026-10", "2026-11"]


# ---------------------------------------------------------------- files

def test_save_and_load(tmp_path):
    path = tmp_path / "expenses.json"
    save(path, SAMPLE)
    assert json.loads(path.read_text(encoding="utf-8")) == SAMPLE
    assert load(path) == SAMPLE


def test_load_missing_file(tmp_path):
    assert load(tmp_path / "nope.json") == []


# ---------------------------------------------------------------- report

def test_format_report():
    expected = "\n".join([
        "Category          Total",
        "travel          1200.00",
        "food             340.00",
        "-----------------------",
        "TOTAL           1540.00",
    ])
    assert format_report(SAMPLE) == expected


def test_format_report_empty():
    assert format_report([]) == "No expenses yet."


# ---------------------------------------------------------------- command line

def test_main_add_and_report(tmp_path, capsys):
    f = str(tmp_path / "e.json")
    assert main(["--file", f, "add", "250", "food", "2026-10-01", "lunch with team"]) == 0
    assert capsys.readouterr().out.strip() == "Added: 2026-10-01  food  Rs 250.00  lunch with team"
    assert main(["--file", f, "add", "1200", "Travel", "2026-10-03"]) == 0
    assert capsys.readouterr().out.strip() == "Added: 2026-10-03  travel  Rs 1200.00"
    assert main(["--file", f, "add", "90", "food", "2026-11-02"]) == 0
    capsys.readouterr()

    assert len(load(f)) == 3

    assert main(["--file", f, "report"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL           1540.00" in out

    assert main(["--file", f, "report", "--month", "2026-10"]) == 0
    out = capsys.readouterr().out
    assert "food             250.00" in out
    assert "TOTAL           1450.00" in out


def test_main_rejects_bad_expense(tmp_path, capsys):
    f = tmp_path / "e.json"
    assert main(["--file", str(f), "add", "-5", "food", "2026-10-01"]) == 1
    assert capsys.readouterr().out.startswith("Error:")
    assert not f.exists() or load(f) == []
