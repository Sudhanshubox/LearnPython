"""m55 project: churn prediction, end to end. Read the README brief first.

Work through the functions in order; the tests check each step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned"


# 1. Load churn_raw.csv and clean it. The result must have:
#    - snake_case column names (m47's rule: "Satisfaction (1-5)" -> "satisfaction_1_5")
#    - no exact duplicate rows
#    - plan lowercased and stripped ("basic", "standard", "premium")
#    - monthly_charge as float (no "$"), tenure as int months, signup_date as datetime
#    - churned as int 0/1
#    - sorted by customer_id with a fresh 0..n-1 index
def load_and_clean(path):
    raise NotImplementedError


# 2. The list of columns that LEAK the target: information that's only known because the
#    customer already churned. (Explore the data to find it! Don't include identifiers here.)
def leaky_columns(df):
    raise NotImplementedError


# 3. (numeric_cols, categorical_cols) to use as model features: every column except
#    customer_id, signup_date, the target and the leaky columns, split by dtype,
#    in the DataFrame's column order.
def feature_columns(df):
    raise NotImplementedError


# 4. A Pipeline with steps "prep" (a ColumnTransformer: median-impute + scale the numeric
#    columns; most-frequent-impute + one-hot encode (handle_unknown="ignore") the categorical
#    columns) and "model" (LogisticRegression(max_iter=2000)).
def build_pipeline(numeric, categorical):
    raise NotImplementedError


# 5. split(df): train_test_split(df, test_size=0.2, random_state=0, stratify=df[TARGET]).
#    train_and_evaluate(df): fit build_pipeline(...) on the training part, predict
#    probabilities on the test part, and return (fitted_pipeline, metrics) where metrics has
#    "roc_auc", "f1" and "accuracy" (threshold 0.5) and "baseline_accuracy" (always
#    predicting the training set's majority class).
def split(df):
    raise NotImplementedError


def train_and_evaluate(df):
    raise NotImplementedError


# 6. The decision threshold minimizing total cost = cost_fn * (false negatives) +
#    cost_fp * (false positives), predicting churn when proba >= threshold. Try thresholds
#    0.01, 0.02, ..., 0.99 (np.round(np.arange(0.01, 1.0, 0.01), 2)); ties: the smallest.
#    Return (threshold, cost).
def best_cost_threshold(y_true, proba, cost_fn=500.0, cost_fp=50.0):
    raise NotImplementedError


# 7. The customer_ids of the n customers with the highest predicted churn probability,
#    highest first.
def top_risk_customers(pipe, df, n=10):
    raise NotImplementedError


# 8. Write your report (see the README outline) as Markdown to `path`, including the
#    metric values. Write real sentences about YOUR findings in each section.
def write_report(metrics, path):
    raise NotImplementedError
