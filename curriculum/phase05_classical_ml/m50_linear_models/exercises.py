"""m50 exercises: linear models as scikit-learn-style estimators, using only NumPy.

Every fit() returns self. Learned attributes end with "_".
"""

import numpy as np


# 1. Standardize features: fit() learns mean_ and scale_ (the population std; use 1.0
#    where the std is 0), transform() returns (X - mean_) / scale_, and fit_transform()
#    does both.
class StandardScaler:
    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        raise NotImplementedError


# 2. Ordinary least squares with an intercept.
#    fit: coef_ (d,) and intercept_ (float) via np.linalg.lstsq with a column of ones.
#    predict: X @ coef_ + intercept_.  score: R^2 on (X, y).
class LinearRegression:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError


# 3. Ridge regression with penalty alpha on the weights (NOT the intercept), using the
#    closed form on centered data: w = solve(Xc^T Xc + alpha I, Xc^T yc), b = y_mean - x_mean @ w.
#    Same predict/score as LinearRegression (you may inherit from it).
class Ridge:
    def __init__(self, alpha=1.0):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError


# 4. Binary logistic regression trained by full-batch gradient descent.
#    Start from zeros; for `epochs` steps compute p = sigmoid(X @ w + b) and update with
#      dw = X^T (p - y) / n + l2 * w;   db = mean(p - y)
#    Store coef_, intercept_ and losses_ (binary cross-entropy + 0.5 * l2 * ||w||^2 before each step).
#    predict_proba(X) -> probability of class 1 (1-D). predict(X) -> 1 where proba >= 0.5 else 0.
#    score -> accuracy.
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000, l2=0.0):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError


# 5. Softmax (multinomial) regression for labels 0..k-1, trained by full-batch gradient
#    descent from zeros:  P = softmax(X @ W + b);  Y = one-hot labels;
#      dW = X^T (P - Y) / n + l2 * W;   db = mean(P - Y, axis=0)
#    Store coef_ (d, k), intercept_ (k,), classes_ (np.arange(k)) and losses_ (mean cross-entropy
#    + 0.5 * l2 * ||W||^2 before each step). Use a numerically stable softmax.
#    predict_proba -> (n, k); predict -> argmax; score -> accuracy.
class SoftmaxRegression:
    def __init__(self, lr=0.1, epochs=500, l2=0.0):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError
