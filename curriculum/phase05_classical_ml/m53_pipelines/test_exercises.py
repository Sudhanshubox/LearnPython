from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from exercises import (
    HouseFeatures,
    build_pipeline,
    build_preprocessor,
    selection_leakage_demo,
    split_columns,
    tune_ridge,
)

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def houses():
    return pd.read_csv(HERE / "houses.csv")


@pytest.fixture(scope="module")
def split(houses):
    X, y, num, cat = split_columns(houses, "price_lakh")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    return Xtr, Xte, ytr, yte, num, cat


def test_split_columns(houses):
    X, y, num, cat = split_columns(houses, "price_lakh")
    assert "price_lakh" not in X.columns and y.name == "price_lakh"
    assert num == ["area_sqft", "bedrooms", "age_years", "floor"]
    assert cat == ["city", "furnishing"]


def test_house_features(houses):
    X = houses.drop(columns="price_lakh").head(5)
    t = HouseFeatures()
    assert t.fit(X) is t
    out = t.transform(X)
    assert "area_per_bedroom" not in X.columns, "return a new DataFrame"
    np.testing.assert_allclose(out["area_per_bedroom"], X["area_sqft"] / X["bedrooms"])
    np.testing.assert_allclose(out["log_area"], np.log(X["area_sqft"]))
    np.testing.assert_array_equal(out["is_new"], (X["age_years"] < 5).astype(float))
    np.testing.assert_array_equal(out["is_ground_floor"], (X["floor"] == 0).astype(float))
    assert HouseFeatures().fit_transform(X).shape[1] == X.shape[1] + 4


def test_preprocessor(split):
    Xtr, Xte, _, _, num, cat = split
    prep = build_preprocessor(num, cat)
    assert isinstance(prep, ColumnTransformer)
    Z = prep.fit_transform(Xtr)
    assert Z.shape == (len(Xtr), 4 + 4 + 3)                  # 4 numeric + 4 cities + 3 furnishing types
    assert not np.isnan(np.asarray(Z, dtype=float)).any()
    dense = np.asarray(Z.todense() if hasattr(Z, "todense") else Z, dtype=float)
    np.testing.assert_allclose(dense[:, :4].mean(axis=0), 0, atol=1e-10)


def test_preprocessor_handles_unseen_categories(split):
    Xtr, Xte, _, _, num, cat = split
    prep = build_preprocessor(num, cat).fit(Xtr)
    new = Xte.head(2).copy()
    new["city"] = "Chandigarh"
    prep.transform(new)


def test_pipeline_structure_and_quality(split):
    Xtr, Xte, ytr, yte, num, cat = split
    pipe = build_pipeline(Ridge(alpha=1.0), num, cat)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == ["features", "prep", "model"]
    pipe.fit(Xtr, ytr)
    r2 = pipe.score(Xte, yte)
    assert r2 > 0.75
    base = build_preprocessor(num, cat)
    plain = Pipeline([("prep", base), ("model", Ridge(alpha=1.0))]).fit(Xtr, ytr).score(Xte, yte)
    assert r2 > plain, "the engineered features should help"


def test_pipeline_learns_only_from_training_data(split):
    Xtr, _, ytr, _, num, cat = split
    pipe = build_pipeline(Ridge(), num, cat).fit(Xtr, ytr)
    scaler = pipe.named_steps["prep"].named_transformers_["num"][-1]
    feats = HouseFeatures().fit_transform(Xtr)
    expected_cols = num + ["area_per_bedroom", "log_area", "is_new", "is_ground_floor"]
    imputed = feats[expected_cols].fillna(feats[expected_cols].median())
    np.testing.assert_allclose(scaler.mean_, imputed.mean().to_numpy(), rtol=1e-8)


@pytest.mark.timeout(60)
def test_tune_ridge(split):
    Xtr, Xte, ytr, yte, num, cat = split
    search = tune_ridge(Xtr, ytr, num, cat)
    assert isinstance(search, GridSearchCV)
    assert search.best_params_["model__alpha"] in (0.1, 1.0, 10.0, 100.0)
    assert search.cv_results_["mean_test_score"].shape == (4,)
    assert search.best_estimator_.score(Xte, yte) > 0.75


@pytest.mark.timeout(120)
def test_selection_leakage_demo():
    leaky, honest = selection_leakage_demo()
    assert leaky > 0.7, "selecting features on all the data makes pure noise look predictive"
    assert 0.3 < honest < 0.65, "inside the pipeline, the honest estimate is near chance (0.5)"
