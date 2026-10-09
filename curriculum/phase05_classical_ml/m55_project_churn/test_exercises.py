from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from exercises import (
    best_cost_threshold,
    build_pipeline,
    feature_columns,
    leaky_columns,
    load_and_clean,
    split,
    top_risk_customers,
    train_and_evaluate,
    write_report,
)

HERE = Path(__file__).parent
RAW = HERE / "churn_raw.csv"


@pytest.fixture(scope="module")
def df():
    return load_and_clean(RAW)


@pytest.fixture(scope="module")
def trained(df):
    return train_and_evaluate(df)


def test_clean_columns_and_types(df):
    expected = ["customer_id", "signup_date", "plan", "contract", "monthly_charge", "tenure", "support_tickets",
                "days_since_last_login", "devices", "autopay", "payment_method", "region", "satisfaction_1_5",
                "cancellation_reason", "churned"]
    assert list(df.columns) == expected
    assert pd.api.types.is_float_dtype(df["monthly_charge"])
    assert pd.api.types.is_integer_dtype(df["tenure"])
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    assert set(df["churned"].unique()) == {0, 1}
    assert set(df["plan"].unique()) == {"basic", "standard", "premium"}


def test_clean_rows(df):
    raw = pd.read_csv(RAW)
    assert len(df) == len(raw.drop_duplicates()) == 3000
    assert df["customer_id"].is_unique
    assert list(df["customer_id"]) == sorted(df["customer_id"])
    assert list(df.index) == list(range(3000))
    assert df["monthly_charge"].between(5, 100).all()
    assert df["tenure"].between(1, 72).all()


def test_leaky_columns(df):
    assert leaky_columns(df) == ["cancellation_reason"]


def test_feature_columns(df):
    numeric, categorical = feature_columns(df)
    assert numeric == ["monthly_charge", "tenure", "support_tickets", "days_since_last_login", "devices", "satisfaction_1_5"]
    assert categorical == ["plan", "contract", "autopay", "payment_method", "region"]


def test_pipeline_structure(df):
    pipe = build_pipeline(*feature_columns(df))
    assert isinstance(pipe, Pipeline)
    assert [n for n, _ in pipe.steps] == ["prep", "model"]


def test_split(df):
    train, test = split(df)
    assert len(test) == 600
    assert abs(train["churned"].mean() - test["churned"].mean()) < 0.01


def test_honest_metrics(trained):
    _, metrics = trained
    assert set(metrics) >= {"roc_auc", "f1", "accuracy", "baseline_accuracy"}
    assert metrics["roc_auc"] < 0.95, "AUC this high on this data means something is leaking"
    assert metrics["roc_auc"] > 0.8
    assert metrics["accuracy"] > metrics["baseline_accuracy"] + 0.1
    assert metrics["baseline_accuracy"] == pytest.approx(1 - 0.4067, abs=0.01)


def test_best_cost_threshold(df, trained):
    pipe, _ = trained
    _, test = split(df)
    numeric, categorical = feature_columns(df)
    proba = pipe.predict_proba(test[numeric + categorical])[:, 1]
    y = test["churned"].to_numpy()
    t, cost = best_cost_threshold(y, proba)
    cost_at_half = 500 * np.sum((proba < 0.5) & (y == 1)) + 50 * np.sum((proba >= 0.5) & (y == 0))
    assert t < 0.5, "missed churners are expensive, so the threshold should drop"
    assert cost < 0.6 * cost_at_half
    assert best_cost_threshold(np.array([1, 0]), np.array([0.9, 0.1])) == (0.11, 0.0)   # 0.11 is the first threshold above the negative's 0.1


def test_top_risk_customers(df, trained):
    pipe, _ = trained
    ids = top_risk_customers(pipe, df, n=10)
    assert len(ids) == 10 and len(set(ids)) == 10
    risky = df.set_index("customer_id").loc[ids]
    assert risky["churned"].mean() > 0.7
    assert (risky["contract"] == "month-to-month").mean() > 0.7


def test_report(tmp_path, trained):
    _, metrics = trained
    path = tmp_path / "report.md"
    write_report(metrics, path)
    text = path.read_text(encoding="utf-8")
    for heading in ["# Churn model report", "## Data", "## Leakage", "## Results", "## Recommendations"]:
        assert heading in text
    assert f"{metrics['roc_auc']:.3f}" in text or f"{metrics['roc_auc']:.2f}" in text
    assert "cancellation_reason" in text
    assert len(text.split()) > 60, "write real sentences about your findings"
