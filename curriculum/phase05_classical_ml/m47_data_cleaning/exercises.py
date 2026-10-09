"""m47 exercises: data cleaning. Never modify the input: return new objects.
Use vectorized pandas operations (.str, masks, groupby), not loops over rows.
"""

import numpy as np
import pandas as pd


# 1. Normalize column names: strip, lowercase, replace each run of characters that are
#    not a-z or 0-9 with "_", and strip "_" from both ends.
#    " Customer ID" -> "customer_id", "E-mail" -> "e_mail"
def clean_column_names(df):
    raise NotImplementedError


# 2. Normalize a text Series: strip, collapse runs of whitespace into one space,
#    Title Case. Missing values stay missing.
#    "  new   delhi " -> "New Delhi"
def normalize_text(s):
    raise NotImplementedError


# 3. Parse ages from strings like "34", "34 years", " 34", "" into floats: take the first
#    (possibly negative) integer in the text; ages outside 0..120 become NaN; text without
#    a number becomes NaN.
def parse_age(s):
    raise NotImplementedError


# 4. Parse money strings like "$12,500", "$ 67000", "67000", "" into floats
#    (remove "$", "," and spaces). Empty or missing -> NaN.
def parse_money(s):
    raise NotImplementedError


# 5. True where the value looks like an email: something@something.something with no
#    spaces and exactly one "@". Missing values -> False.
def is_valid_email(s):
    raise NotImplementedError


# 6. Clip a numeric Series to [Q1 - k*IQR, Q3 + k*IQR] (quantiles 0.25 and 0.75,
#    ignoring NaN). NaN stays NaN.
def clip_iqr(s, k=1.5):
    raise NotImplementedError


# 7. Keep one row per id: the one with the LATEST value in date_col (a datetime column).
#    Return it sorted by id with a fresh 0..n-1 index.
def latest_per_id(df, id_col, date_col):
    raise NotImplementedError


# 8. Impute missing values: numeric columns with their median, categorical columns with
#    "unknown". Return (new_df, filled) where filled is a dict {column: number of values filled}
#    for every column in the two lists.
def impute(df, numeric_cols, categorical_cols):
    raise NotImplementedError


# 9. The full pipeline from customers_raw.csv (already loaded as `raw`) to a clean table
#    with EXACTLY these columns, in this order:
#      customer_id (int), city (normalized; missing -> "unknown"), age (float; parsed,
#      invalid -> NaN, then median-imputed), annual_income (float; parsed, clipped with
#      clip_iqr(k=3), then median-imputed), email (as given), email_valid (bool),
#      signup_date (datetime)
#    One row per customer_id (the latest signup), sorted by customer_id, index 0..n-1.
#    Do the imputation AFTER deduplication.
def clean_customers(raw):
    raise NotImplementedError
