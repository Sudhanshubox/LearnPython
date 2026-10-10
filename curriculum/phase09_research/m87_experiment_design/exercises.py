"""m87 exercises: grids, ablations, power analysis, and a pre-registered experiment."""

import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# 1. Every combination of a search space {name: [values]}, as a list of dicts. Keys keep the
#    space's order; combinations come in itertools.product order.
#    grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
def grid(space):
    raise NotImplementedError


# 2. Ablations: [("full", copy of base)] then, for each component in order,
#    ("no_" + component, a copy of base with that component set to False).
def ablations(base, components):
    raise NotImplementedError


# 3. A stable 8-character ID: the first 8 hex characters of the SHA-256 of
#    json.dumps(config, sort_keys=True).
def config_id(config):
    raise NotImplementedError


# 4. Seeds per group to detect an effect of size `effect` when runs vary with standard
#    deviation `std` (README section 4). Use statistics.NormalDist().inv_cdf for the z-values
#    and round UP to an integer.
def seeds_needed(effect, std, alpha=0.05, power=0.8):
    raise NotImplementedError


# 5. The accuracy of always predicting the most common class of y_train on y_test (a float).
def majority_baseline(y_train, y_test):
    raise NotImplementedError


# 6. One training run on the digits. config has "model" ("logreg" or "tree"), "scale" (bool)
#    and "n_train" (int).
#    Split load_digits with train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y);
#    keep only the first n_train training examples; if scale, fit a StandardScaler on the
#    (reduced) training set and transform both sets; model: LogisticRegression(max_iter=2000)
#    or DecisionTreeClassifier(random_state=seed); unknown model -> ValueError.
#    Return the test accuracy as a float.
def train_digits(config, seed):
    raise NotImplementedError


# 7. Run train_fn(config, seed) for every config and every seed (configs outer, seeds inner).
#    Return a list of {"config_id", "config", "seed", "metric"}.
def run_experiment(train_fn, configs, seeds):
    raise NotImplementedError


# 8. {config_id: {"config", "mean", "std" (sample std, 0.0 for one run), "n"}}.
def summarize(results):
    raise NotImplementedError


@dataclass
class Hypothesis:
    """Given: a pre-registered hypothesis: treatment beats control by at least min_effect."""
    statement: str
    treatment: dict
    control: dict
    min_effect: float


# 9. The decision rule (README section 5). From the summary entries of the treatment and the
#    control: diff = mean_t - mean_c, se = sqrt(std_t^2 / n_t + std_c^2 / n_c).
#    "supported" if diff - 2 se >= min_effect; "not supported" if diff + 2 se < min_effect;
#    otherwise "inconclusive".
def decide(hypothesis, summary):
    raise NotImplementedError
