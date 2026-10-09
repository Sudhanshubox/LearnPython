"""m54 exercises: gradient boosting from scratch (NumPy only)."""

import numpy as np


# 1. The best split for a REGRESSION tree: the (feature, threshold) maximizing
#    gain = SSE(parent) - SSE(left) - SSE(right), where each side must keep at least
#    min_samples_leaf samples. Thresholds are midpoints between consecutive distinct sorted
#    values. Return (feature, threshold, gain), or (None, None, 0.0) if no valid split has
#    positive gain. Use sorting + cumulative sums of y and y**2 (see the README).
def best_regression_split(X, y, min_samples_leaf=1):
    raise NotImplementedError


# 2. A regression tree: leaves predict the mean of their samples. Stop at max_depth, when a
#    node has fewer than 2 * min_samples_leaf samples, or when there's no positive-gain split.
#    X[:, feature] <= threshold goes left.
class RegressionTree:
    def __init__(self, max_depth=3, min_samples_leaf=1):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


# 3. Gradient boosting for regression (squared loss).
#    fit: init_ = mean(y); then n_estimators times, fit a RegressionTree(max_depth,
#    min_samples_leaf) to the residuals y - F and update F += learning_rate * tree prediction.
#    Keep trees_ and train_losses_ (training MSE AFTER each tree).
#    staged_predict(X): a generator yielding the prediction after 1, 2, ..., M trees.
#    predict(X): the final prediction.  score(X, y): R^2.
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def staged_predict(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError


# 4. Gradient boosting for BINARY classification (log-loss).
#    init_ = log(p / (1 - p)) with p = mean(y) (clipped to [1e-6, 1 - 1e-6]); each round, fit a
#    RegressionTree(max_depth) to y - sigmoid(F) and update F += learning_rate * tree prediction.
#    decision_function(X) -> F (log-odds); predict_proba(X) -> sigmoid(F) (1-D);
#    predict(X) -> 1 where proba >= 0.5; score -> accuracy.
class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError


# 5. Early stopping: using model.staged_predict(X_val), return the number of trees (1-based)
#    that gives the lowest validation MSE (ties: the fewest trees).
def best_iteration(model, X_val, y_val):
    raise NotImplementedError
