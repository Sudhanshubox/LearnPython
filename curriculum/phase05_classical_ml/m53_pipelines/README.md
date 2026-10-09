# m53 · Feature engineering and scikit-learn pipelines

**By the end you can:** turn raw mixed-type columns into model-ready features, build leak-proof preprocessing with `Pipeline` and `ColumnTransformer`, write your own transformers, and tune hyperparameters with cross-validated grid search.

**Why it matters for AI:** in practice, good features and correct preprocessing often matter more than the choice of model. Pipelines guarantee that exactly the same steps run in training and in production, and that nothing from validation or test data leaks into training, the mistake behind a huge share of "too good to be true" results (m49).

The dataset is `houses.csv`: house prices (in lakh rupees) with numeric, categorical and missing values.

---

## 1. Kinds of features and how to encode them

| Column type | Typical preprocessing |
|---|---|
| numeric | impute missing (median), scale (`StandardScaler`) for linear models; maybe log-transform skewed values |
| categorical | impute (most frequent or "missing"), **one-hot encode** (`OneHotEncoder`) |
| ordinal (small < medium < large) | map to ordered integers (`OrdinalEncoder` with explicit categories) |
| dates | extract year, month, day of week, "days since" |
| text | counts or TF-IDF (m35!), or embeddings (Phase 8) |

One-hot encoding turns `city = "Pune"` into columns `city_Delhi, city_Mumbai, city_Pune, …` with a single 1. Use `handle_unknown="ignore"` so a city never seen in training doesn't crash prediction.

## 2. Feature engineering

Create features that make the pattern easier for the model to learn:

- **Ratios and interactions:** `area_per_bedroom = area / bedrooms`.
- **Transforms:** prices and areas are often **log-normal** (multiplicative effects), so `log(area)` relates linearly to `log(price)`. Modelling the log of the target and exponentiating predictions can greatly help linear models.
- **Flags:** `is_new = age < 5`, `is_ground_floor = floor == 0`.
- **Domain knowledge** beats brute force: talk to someone who knows the data.

## 3. Pipelines

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical_cols),
])
model = Pipeline([("prep", preprocess), ("model", Ridge())])

model.fit(X_train, y_train)          # fits imputers, scaler, encoder AND the model on training data only
model.predict(X_test)                # applies the same fitted steps to the test data
```

Because the whole pipeline is one estimator, cross-validation and grid search refit preprocessing **inside each fold**. That's what makes it leak-proof.

## 4. Writing your own transformer

Follow scikit-learn's interface (m50): inherit from `BaseEstimator` and `TransformerMixin`, learn in `fit` (return `self`), apply in `transform`. `TransformerMixin` gives you `fit_transform` for free.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class AddRatio(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["area_per_bedroom"] = X["area_sqft"] / X["bedrooms"]
        return X
```

## 5. Hyperparameter tuning

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(model, {"model__alpha": [0.1, 1, 10]}, cv=5, scoring="neg_root_mean_squared_error")
search.fit(X_train, y_train)
search.best_params_, search.best_score_
```

Parameter names use `step__param` to reach inside the pipeline. Tune on the training data with cross-validation, then evaluate the best pipeline **once** on the test set.

## 6. The feature-selection leakage trap

A famous mistake: select the "most predictive" features using **all** the data, then cross-validate a model on them. Even on **pure random noise**, some features correlate with the labels by chance; selecting them on the full data, including the validation folds, makes cross-validation look great. Putting the selection step *inside* the pipeline fixes it. Exercise 6 makes you measure this effect yourself; it's one of the most important lessons in applied ML.

---

## Problem-solving habit #49: everything learned from data goes in the pipeline

Any step that computes something from the data (means for imputation, scaling statistics, category lists, selected features, PCA components) must be *fitted on training data only*. If it's in the pipeline, that happens automatically. If you catch yourself calling `.fit` on the full dataset outside a pipeline, stop and ask whether it leaks.

## Go deeper (optional, research-level)

1. Read section 7.10.2 of *The Elements of Statistical Learning* (Hastie, Tibshirani & Friedman), "The Wrong and Right Way to Do Cross-validation". It's the source of exercise 6's demo.
2. **Target encoding** replaces a category by the mean target value of that category. It's powerful for high-cardinality columns but leaks easily. How does scikit-learn's `TargetEncoder` use cross-fitting to avoid that?
3. Compare `Ridge` on raw prices vs on `log(price)` (with `TransformedTargetRegressor`). Why does the log help here? (Look at how the data was generated: multiplicative effects.)

## Your turn

Open the **Exercises** tab. Here you *should* use scikit-learn; the point is to use it correctly.
