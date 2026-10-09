"""m46 exercises: pandas fundamentals.

Functions receive DataFrames and must NOT modify them: return new objects.
Use pandas operations (no Python loops over rows).
"""

import pandas as pd


# 1. Load sales.csv (path given) with the "date" column parsed as datetimes.
def load_sales(path):
    raise NotImplementedError


# 2. Return a copy of df with a new "revenue" column = units * unit_price.
def add_revenue(df):
    raise NotImplementedError


# 3. The n orders with the highest revenue, highest first, with only the columns
#    ["order_id", "region", "revenue"] and a fresh 0..n-1 index. (df has "revenue".)
def top_orders(df, n=5):
    raise NotImplementedError


# 4. Orders from `region` with at least `min_units` units.
def filter_orders(df, region, min_units):
    raise NotImplementedError


# 5. Total revenue per region as a Series, largest first.
def revenue_by_region(df):
    raise NotImplementedError


# 6. A DataFrame with one row per region (the index) and columns
#    "orders" (number of orders), "revenue" (sum) and "avg_units" (mean units).
def region_summary(df):
    raise NotImplementedError


# 7. Pivot table of revenue: index = region, columns = product, values = summed revenue,
#    missing combinations filled with 0.
def region_product_table(df):
    raise NotImplementedError


# 8. Monthly revenue as a Series indexed by month strings like "2026-01", in order.
def monthly_revenue(df):
    raise NotImplementedError


# 9. Add each order's product category from the products table (left merge on "product").
#    Products missing from the table get the category "unknown". Row count must not change.
def add_category(df, products):
    raise NotImplementedError


# 10. Add two columns: "region_share" = the order's revenue / its region's total revenue,
#     and "rank_in_region" = 1 for the region's biggest order, 2 for the next... (ties:
#     earlier row first, i.e. method="first"), as ints.
def add_region_stats(df):
    raise NotImplementedError
