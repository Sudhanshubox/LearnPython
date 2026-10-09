"""m51 exercises: decision trees and random forests from scratch (NumPy only).

Labels y are int arrays with values 0..k-1.
"""

import numpy as np


# 1. Gini impurity and entropy (in bits) of a label array. Empty arrays -> 0.0.
def gini(y):
    raise NotImplementedError


def entropy(y):
    raise NotImplementedError


# 2. The best split by Gini gain. Consider the given feature indices (all features if
#    `features` is None), in increasing order. Candidate thresholds are midpoints between
#    consecutive DISTINCT sorted values of a feature. Return (feature, threshold, gain) for
#    the largest gain (ties: the first feature checked, then the smallest threshold), or
#    (None, None, 0.0) if no split has a positive gain.
#    Make it fast: sort each feature once and use cumulative class counts (see the lesson).
def best_split(X, y, features=None):
    raise NotImplementedError


# 3. A decision tree classifier.
#    - Stop and make a leaf (predicting the most common class; ties: the smallest label) when
#      depth == max_depth, len(y) < min_samples_split, the node is pure, or no split has
#      positive gain.
#    - Samples with X[:, feature] <= threshold go LEFT.
#    - If max_features is set, each split considers a random subset of that many features,
#      drawn with rng.choice(n_features, max_features, replace=False) where
#      rng = np.random.default_rng(seed) is created once in fit().
#    - After fit: feature_importances_ (sum over splits of gain * n_node / n_total, per
#      feature, normalized to sum to 1; all zeros if the tree is a single leaf).
#    - depth() and n_leaves() describe the fitted tree (a single leaf has depth 0).
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, seed=0):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError

    def depth(self):
        raise NotImplementedError

    def n_leaves(self):
        raise NotImplementedError


# 4. A random forest. In fit, rng = np.random.default_rng(seed); for each of n_estimators
#    trees, draw a bootstrap sample rng.integers(0, n, n) and fit a DecisionTreeClassifier
#    with max_depth, max_features (int(sqrt(n_features)) if "sqrt", at least 1) and
#    seed = seed * 1000 + tree_index. Keep them in trees_.
#    predict: majority vote across trees (ties: the smallest label).
class RandomForestClassifier:
    def __init__(self, n_estimators=50, max_depth=None, max_features="sqrt", seed=0):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError
