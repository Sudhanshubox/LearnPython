"""m53 exercises: feature engineering and scikit-learn pipelines.

Use scikit-learn here, correctly: anything fitted on data belongs inside a pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Split a DataFrame into (X, y, numeric_cols, categorical_cols): y is the `target` column,
#    X is everything else, numeric_cols are X's numeric columns and categorical_cols the rest,
#    each list in the original column order.
def split_columns(df, target):
    raise NotImplementedError


# 2. A transformer that adds engineered features to a DataFrame (returning a NEW DataFrame):
#      area_per_bedroom = area_sqft / bedrooms
#      log_area = log(area_sqft)
#      is_new = 1.0 if age_years < 5 else 0.0 (missing age -> 0.0)
#      is_ground_floor = 1.0 if floor == 0 else 0.0
#    It learns nothing, so fit just returns self.
class HouseFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


# 3. A ColumnTransformer with two parts:
#    "num": median SimpleImputer then StandardScaler, on numeric_cols
#    "cat": most_frequent SimpleImputer then OneHotEncoder(handle_unknown="ignore"), on categorical_cols
def build_preprocessor(numeric_cols, categorical_cols):
    raise NotImplementedError


# 4. A full Pipeline with steps named "features" (HouseFeatures()), "prep" (the preprocessor,
#    whose numeric columns must include the 4 engineered ones) and "model" (the given model).
def build_pipeline(model, numeric_cols, categorical_cols):
    raise NotImplementedError


# 5. Grid-search the Ridge alpha of build_pipeline(Ridge(), ...) over `alphas` with 5-fold
#    cross-validation and scoring="neg_root_mean_squared_error". Return the FITTED GridSearchCV.
def tune_ridge(X, y, numeric_cols, categorical_cols, alphas=(0.1, 1.0, 10.0, 100.0)):
    raise NotImplementedError


# 6. The feature-selection leakage demo, on PURE NOISE: X = rng.normal(size=(n, d)) and
#    y = rng.integers(0, 2, n) with rng = np.random.default_rng(seed) (X drawn first).
#    leaky: select the k best features with SelectKBest(f_classif, k=k) fitted on ALL of X, y,
#           then 5-fold cross_val_score of LogisticRegression() on the selected columns.
#    honest: 5-fold cross_val_score of a Pipeline(SelectKBest(f_classif, k=k), LogisticRegression())
#           on the full X.
#    Return (mean leaky accuracy, mean honest accuracy).
def selection_leakage_demo(n=100, d=5000, k=20, seed=0):
    raise NotImplementedError
