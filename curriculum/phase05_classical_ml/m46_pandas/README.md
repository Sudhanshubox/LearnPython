# m46 · pandas fundamentals

**By the end you can:** load tabular data into a DataFrame, select and filter rows and columns, create new columns, aggregate with `groupby`, reshape with pivot tables, and combine tables with `merge`.

**Why it matters for AI:** most real-world ML starts with tables: customers, transactions, sensor logs, experiment results, evaluation outputs. pandas is the standard tool for exploring and preparing them, and it's how you'll analyze your own experiment logs and model errors. Fluency here makes every later project faster.

---

## 1. Series and DataFrames

A **Series** is a labelled 1-D array; a **DataFrame** is a table of Series sharing one **index** (the row labels).

```python
import pandas as pd

df = pd.read_csv("sales.csv", parse_dates=["date"])
df.head()            # first 5 rows
df.shape             # (40, 6)
df.dtypes            # each column's type
df.info()            # types, non-null counts, memory
df.describe()        # summary statistics of numeric columns
```

## 2. Selecting

```python
df["units"]                      # one column → Series
df[["region", "units"]]          # several columns → DataFrame
df.loc[5, "region"]              # by LABEL: row label 5, column "region"
df.loc[df["units"] > 10, ["order_id", "units"]]   # rows by condition, chosen columns
df.iloc[0:3, 1]                  # by POSITION, like NumPy
```

**Rule of thumb:** use `.loc` with labels and conditions, `.iloc` with integer positions. Avoid chained indexing like `df[df.units > 10]["units"] = 0`: it may modify a temporary copy. (In pandas 3, it never modifies the original, by design.)

## 3. Filtering

```python
big = df[(df["units"] >= 10) & (df["region"] == "north")]   # & | ~ with parentheses (like m36's masks)
df[df["product"].isin(["pen", "lamp"])]
df.query("units >= 10 and region == 'north'")              # the same, as a string
```

## 4. New columns and vectorized operations

```python
df["revenue"] = df["units"] * df["unit_price"]       # vectorized, like NumPy
df["month"] = df["date"].dt.to_period("M")           # .dt for dates
df["product_upper"] = df["product"].str.upper()      # .str for text
df = df.assign(revenue=df.units * df.unit_price)     # returns a new DataFrame instead
```

`df.sort_values("revenue", ascending=False)`, `df.nlargest(5, "revenue")` and `df["region"].value_counts()` are everyday tools.

## 5. groupby: split, apply, combine

```python
df.groupby("region")["revenue"].sum()                      # one number per region
df.groupby("region").agg(orders=("order_id", "count"),
                         revenue=("revenue", "sum"),
                         avg_units=("units", "mean"))      # several named aggregations
df.groupby(["region", "product"])["units"].sum()           # by two keys
```

`transform` returns a result aligned with the *original* rows, which is perfect for "share of group" or "rank within group":

```python
df["region_total"] = df.groupby("region")["revenue"].transform("sum")
df["share"] = df["revenue"] / df["region_total"]
df["rank"] = df.groupby("region")["revenue"].rank(ascending=False, method="first")
```

## 6. Reshaping: pivot tables

```python
df.pivot_table(index="region", columns="product", values="revenue",
               aggfunc="sum", fill_value=0)
```

Rows are regions, columns are products, and cells are summed revenue: the spreadsheet "pivot table".

## 7. Combining tables: merge

```python
products = pd.read_csv("products.csv")
df.merge(products, on="product", how="left")
```

| how | keeps |
|---|---|
| `inner` | only keys found in both tables |
| `left` | every row of the left table (missing matches become NaN) |
| `outer` | every key from either table |

After a merge, always check the row count: an unexpected increase means duplicate keys on the right side, a classic silent bug.

---

## Problem-solving habit #42: look at the data first

Before any analysis, run `df.head()`, `df.info()`, `df.describe()` and a few `value_counts()`. You'll spot wrong types, missing values, typos and surprising distributions in minutes, before they become hours of confusing results.

## Go deeper (optional, research-level)

1. pandas 3 made **copy-on-write** the default. Read the pandas docs page "Copy-on-Write" and explain how it changes `df2 = df[df.units > 5]; df2["units"] = 0`.
2. For data too big for pandas, people use Polars or DuckDB. Rewrite exercise 5 in DuckDB's SQL (`duckdb.sql("SELECT region, sum(...) FROM df GROUP BY region")`). When is SQL clearer than pandas?
3. *Tidy Data* (Hadley Wickham, 2014): what are the three rules of tidy data, and which pandas operations convert between "wide" and "long" formats (`melt`, `pivot`)?

## Your turn

Open the **Exercises** tab. The functions use `sales.csv` and `products.csv` in this folder. None of them should modify the DataFrame passed in.
