from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    clean_column_names,
    clean_customers,
    clip_iqr,
    impute,
    is_valid_email,
    latest_per_id,
    normalize_text,
    parse_age,
    parse_money,
)

HERE = Path(__file__).parent


@pytest.fixture
def raw():
    return pd.read_csv(HERE / "customers_raw.csv")


@pytest.mark.parametrize("name", ["clean_column_names", "normalize_text", "parse_age", "parse_money", "is_valid_email", "clip_iqr", "latest_per_id", "clean_customers"])
def test_no_loops(name):
    assert not has_loops(getattr(exercises, name))


def test_clean_column_names(raw):
    out = clean_column_names(raw)
    assert list(out.columns) == ["customer_id", "city", "age", "annual_income", "e_mail", "signup_date"]
    assert list(raw.columns)[0] == " Customer ID", "don't modify the input"


def test_normalize_text():
    s = pd.Series(["  new   delhi ", "MUMBAI", "pune", None])
    out = normalize_text(s)
    assert list(out[:3]) == ["New Delhi", "Mumbai", "Pune"]
    assert pd.isna(out[3])


def test_parse_age():
    s = pd.Series(["34", "34 years", " 7", "", "999", "-3", None, "unknown"])
    out = parse_age(s)
    assert list(out[:3]) == [34.0, 34.0, 7.0]
    assert out[3:].isna().all()


def test_parse_money():
    s = pd.Series(["$12,500", "$ 67000", "67000", "", None, "$9,999,999"])
    out = parse_money(s)
    assert list(out[:3]) == [12500.0, 67000.0, 67000.0]
    assert out[3:5].isna().all()
    assert out[5] == 9_999_999.0


def test_is_valid_email():
    s = pd.Series(["a@b.com", "first.last@uni.ac.in", "a.example.com", "a@", "", None, "a b@c.com", "a@@b.com"])
    assert list(is_valid_email(s)) == [True, True, False, False, False, False, False, False]


def test_clip_iqr():
    s = pd.Series([10.0, 11, 12, 13, 14, 1000, np.nan])
    out = clip_iqr(s)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    assert out.max() == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert out[0] == 10.0
    assert pd.isna(out[6])


def test_latest_per_id():
    df = pd.DataFrame({
        "id": [2, 1, 2, 1, 3],
        "date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-01-15", "2026-01-01"]),
        "city": ["old", "newest", "newest", "old", "only"],
    })
    out = latest_per_id(df, "id", "date")
    assert list(out["id"]) == [1, 2, 3]
    assert list(out["city"]) == ["newest", "newest", "only"]
    assert list(out.index) == [0, 1, 2]


def test_impute():
    df = pd.DataFrame({"age": [10.0, np.nan, 30.0, np.nan], "city": ["A", None, "B", "C"], "x": [1, 2, 3, 4]})
    out, filled = impute(df, ["age"], ["city"])
    assert list(out["age"]) == [10.0, 20.0, 30.0, 20.0]
    assert list(out["city"]) == ["A", "unknown", "B", "C"]
    assert filled == {"age": 2, "city": 1}
    assert df["age"].isna().sum() == 2, "don't modify the input"


def test_clean_customers(raw):
    clean = clean_customers(raw)
    assert list(clean.columns) == ["customer_id", "city", "age", "annual_income", "email", "email_valid", "signup_date"]
    assert clean["customer_id"].is_unique
    assert len(clean) == raw[" Customer ID"].nunique() == 60
    assert list(clean["customer_id"]) == sorted(clean["customer_id"])
    assert list(clean.index) == list(range(60))
    assert pd.api.types.is_integer_dtype(clean["customer_id"])
    assert pd.api.types.is_datetime64_any_dtype(clean["signup_date"])
    assert clean["email_valid"].dtype == bool
    assert set(clean["city"]) <= {"Delhi", "Mumbai", "Pune", "Bengaluru", "Chennai", "Unknown", "unknown"}
    assert clean["age"].notna().all() and clean["age"].between(0, 120).all()
    assert clean["annual_income"].notna().all()
    assert clean["annual_income"].max() < 1_000_000, "the $9,999,999 outlier should be clipped"


def test_clean_customers_keeps_latest_record(raw):
    clean = clean_customers(raw)
    df = clean_column_names(raw)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    latest = df.sort_values("signup_date").groupby("customer_id")["signup_date"].last()
    merged = clean.set_index("customer_id")["signup_date"]
    assert (merged == latest.loc[merged.index]).all()
