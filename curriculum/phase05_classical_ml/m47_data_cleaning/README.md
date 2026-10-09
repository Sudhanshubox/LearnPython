# m47 · Data cleaning

**By the end you can:** turn a messy real-world CSV into a clean, typed, trustworthy table: consistent column names, normalized text, parsed numbers and dates, deduplicated records, handled missing values and outliers, all as a reproducible function.

**Why it matters for AI:** "garbage in, garbage out." Practitioners regularly report spending most of their time on data preparation, and that's no accident: a model trained on duplicates, mis-parsed numbers or silently missing values learns the wrong thing, and no architecture can fix it. Even for LLMs, data cleaning and deduplication of the pretraining corpus are among the biggest drivers of quality.

Open `customers_raw.csv` in this folder before reading on. Try to spot every problem yourself.

---

## 1. Column names

Raw headers like `" Customer ID"`, `"City "` and `"E-mail"` are painful to type and easy to get wrong. Normalize them once: strip, lowercase, and replace runs of non-alphanumeric characters with `_`:

```python
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
```

## 2. Text normalization

`"Delhi"`, `"delhi"`, `"DELHI"` and `"  Delhi "` are the same city, but `value_counts()` sees four. Use the vectorized `.str` methods:

```python
s.str.strip().str.replace(r"\s+", " ", regex=True).str.title()
```

## 3. Parsing numbers hidden in text

`"34 years"`, `"$1,20,000"`, `" 42"`: numbers often arrive as strings with units, currency symbols and separators.

```python
s.str.extract(r"(-?\d+)", expand=False).astype(float)          # the first integer in each string
s.str.replace(r"[$,\s]", "", regex=True).replace("", np.nan).astype(float)
pd.to_numeric(s, errors="coerce")       # anything unparseable becomes NaN instead of raising
```

Then apply **domain rules**: an age of 999 or −3 is a data-entry error, not a real age. Set impossible values to missing (NaN) rather than guessing.

## 4. Dates

```python
pd.to_datetime(s, errors="coerce")       # strings → datetime64; bad values → NaT
df["signup_date"].dt.year, .dt.month, .dt.dayofweek
```

## 5. Missing values

```python
df.isna().sum()                          # missing count per column
df.dropna(subset=["email"])              # drop rows missing a key field
df["age"].fillna(df["age"].median())     # impute numbers with the median (robust to outliers)
df["city"].fillna("unknown")             # impute categories with an explicit label
```

Should you drop or impute? Dropping is simple but loses data and can bias the sample (maybe older customers skip the age field). Imputing keeps rows but invents values. Often it's worth adding an `age_missing` indicator column so the model can learn whether missingness itself matters. Whatever you choose, **count what you changed**.

## 6. Duplicates

```python
df.duplicated().sum()                    # exact duplicate rows
df.drop_duplicates()
df.sort_values("signup_date").drop_duplicates("customer_id", keep="last")   # latest record per customer
```

Duplicates are dangerous in ML: if the same example lands in both the training and test sets, your test score is inflated (**leakage**, m49).

## 7. Outliers

An income of $9,999,999 among values around $100,000 might be real or a typo. Common treatments:

- **Investigate** (look at the row; is it plausible?).
- **Clip** to a range based on the interquartile range (IQR): [Q1 − k·IQR, Q3 + k·IQR], with k = 1.5 for "mild" and 3 for "extreme" outliers.
- **Transform** (log scale) so extreme values have less pull.

## 8. Make it a pipeline

Write cleaning as one function, `clean_customers(raw) -> clean_df`, built from small tested steps. Then you can rerun it on next month's data, review it, and trust it. Never clean by hand in a spreadsheet: it can't be reproduced.

---

## Problem-solving habit #43: assert your assumptions about data

After cleaning, write checks that must hold: `assert df["customer_id"].is_unique`, `assert df["age"].between(0, 120).all()`, `assert df["income"].notna().all()`. They document what "clean" means and catch the day the source data changes format.

## Go deeper (optional, research-level)

1. Read section 4 of *Deduplicating Training Data Makes Language Models Better* (Lee et al., 2021). How much of common web datasets was near-duplicate, and what did deduplication change?
2. Missing data comes in three types: missing completely at random (MCAR), at random (MAR), and not at random (MNAR). Find an example of each. Why does median imputation give biased results under MNAR?
3. Look up **data validation** libraries such as Pandera or Great Expectations. How would you express this module's rules as a schema?

## Your turn

Open the **Exercises** tab, solve each function, then build the full `clean_customers` pipeline from them, and press **Run tests**.
