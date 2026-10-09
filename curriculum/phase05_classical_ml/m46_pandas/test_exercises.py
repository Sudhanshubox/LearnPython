import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

import exercises
from code_checks import has_loops
from exercises import (
    add_category,
    add_region_stats,
    add_revenue,
    filter_orders,
    load_sales,
    monthly_revenue,
    region_product_table,
    region_summary,
    revenue_by_region,
    top_orders,
)

HERE = Path(__file__).parent

# Plain-Python reference data, computed without pandas.
with open(HERE / "sales.csv", encoding="utf-8") as f:
    ROWS = [dict(r, units=int(r["units"]), unit_price=float(r["unit_price"])) for r in csv.DictReader(f)]
for r in ROWS:
    r["revenue"] = r["units"] * r["unit_price"]


@pytest.fixture
def df():
    return add_revenue(load_sales(HERE / "sales.csv"))


@pytest.mark.parametrize("name", [n for n in dir(exercises) if callable(getattr(exercises, n)) and getattr(getattr(exercises, n), "__module__", "") == "exercises"])
def test_no_row_loops(name):
    assert not has_loops(getattr(exercises, name)), f"{name}: use pandas operations, not loops"


def test_load_sales():
    d = load_sales(HERE / "sales.csv")
    assert d.shape == (40, 6)
    assert pd.api.types.is_datetime64_any_dtype(d["date"])


def test_add_revenue_returns_new():
    raw = load_sales(HERE / "sales.csv")
    out = add_revenue(raw)
    assert "revenue" not in raw.columns, "don't modify the input"
    assert out["revenue"].sum() == pytest.approx(sum(r["revenue"] for r in ROWS))


def test_top_orders(df):
    top = top_orders(df, 3)
    assert list(top.columns) == ["order_id", "region", "revenue"]
    assert list(top.index) == [0, 1, 2]
    expected = sorted(ROWS, key=lambda r: -r["revenue"])[:3]
    assert list(top["revenue"]) == [r["revenue"] for r in expected]


def test_filter_orders(df):
    out = filter_orders(df, "west", 10)
    expected = sorted(int(r["order_id"]) for r in ROWS if r["region"] == "west" and r["units"] >= 10)
    assert sorted(out["order_id"]) == expected
    assert (out["region"] == "west").all()


def test_revenue_by_region(df):
    s = revenue_by_region(df)
    totals = defaultdict(float)
    for r in ROWS:
        totals[r["region"]] += r["revenue"]
    assert s.to_dict() == pytest.approx(dict(totals))
    assert list(s.values) == sorted(s.values, reverse=True)


def test_region_summary(df):
    s = region_summary(df)
    assert set(s.columns) >= {"orders", "revenue", "avg_units"}
    west = [r for r in ROWS if r["region"] == "west"]
    assert s.loc["west", "orders"] == len(west)
    assert s.loc["west", "revenue"] == pytest.approx(sum(r["revenue"] for r in west))
    assert s.loc["west", "avg_units"] == pytest.approx(sum(r["units"] for r in west) / len(west))


def test_region_product_table(df):
    t = region_product_table(df)
    assert set(t.index) == {"north", "south", "east", "west"}
    assert set(t.columns) == {"notebook", "pen", "backpack", "calculator", "lamp"}
    for region in t.index:
        for product in t.columns:
            expected = sum(r["revenue"] for r in ROWS if r["region"] == region and r["product"] == product)
            assert t.loc[region, product] == pytest.approx(expected)
    assert not t.isna().any().any()


def test_monthly_revenue(df):
    m = monthly_revenue(df)
    assert list(m.index) == ["2026-01", "2026-02", "2026-03"]
    for month in m.index:
        assert m[month] == pytest.approx(sum(r["revenue"] for r in ROWS if r["date"].startswith(month)))


def test_add_category(df):
    products = pd.read_csv(HERE / "products.csv")
    out = add_category(df, products)
    assert len(out) == len(df)
    lamps = out[out["product"] == "lamp"]
    assert (lamps["category"] == "unknown").all() and len(lamps) > 0
    assert (out.loc[out["product"] == "pen", "category"] == "stationery").all()
    assert "category" not in df.columns


def test_add_region_stats(df):
    out = add_region_stats(df)
    for region, group in out.groupby("region"):
        assert group["region_share"].sum() == pytest.approx(1)
        assert sorted(group["rank_in_region"]) == list(range(1, len(group) + 1))
        best = group.loc[group["revenue"].idxmax()]
        assert best["rank_in_region"] == 1
    assert pd.api.types.is_integer_dtype(out["rank_in_region"])
    assert "region_share" not in df.columns
