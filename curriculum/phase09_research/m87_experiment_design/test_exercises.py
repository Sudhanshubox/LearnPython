import math

import numpy as np
import pytest

from exercises import (
    Hypothesis,
    ablations,
    config_id,
    decide,
    grid,
    majority_baseline,
    run_experiment,
    seeds_needed,
    summarize,
    train_digits,
)


def test_grid():
    assert grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    g = grid({"model": ["logreg", "tree"], "scale": [True, False], "n_train": [100, 1000]})
    assert len(g) == 8 and g[0] == {"model": "logreg", "scale": True, "n_train": 100}
    assert g[-1] == {"model": "tree", "scale": False, "n_train": 1000}
    assert grid({}) == [{}]


def test_ablations():
    base = {"scale": True, "augment": True, "lr": 0.1}
    runs = ablations(base, ["scale", "augment"])
    assert runs == [("full", base), ("no_scale", {"scale": False, "augment": True, "lr": 0.1}),
                    ("no_augment", {"scale": True, "augment": False, "lr": 0.1})]
    runs[0][1]["lr"] = 99
    assert base["lr"] == 0.1, "return copies"


def test_config_id():
    assert config_id({"a": 1, "b": 2}) == config_id({"b": 2, "a": 1})
    assert len(config_id({"a": 1})) == 8 and config_id({"a": 1}) != config_id({"a": 2})


def test_seeds_needed():
    assert seeds_needed(1.0, 1.0) == 16
    assert seeds_needed(2.0, 1.0) == 4
    assert seeds_needed(0.5, 1.0) == 63
    assert seeds_needed(1.0, 1.0, power=0.9) == 22


def test_majority_baseline():
    assert majority_baseline([1, 1, 2, 3], [1, 2, 1, 1]) == 0.75
    assert isinstance(majority_baseline(np.array([0, 0]), np.array([1])), float)


def test_train_digits():
    acc = train_digits({"model": "logreg", "scale": True, "n_train": 1000}, seed=0)
    assert 0.94 < acc < 0.99
    assert train_digits({"model": "logreg", "scale": True, "n_train": 1000}, 0) == acc
    assert train_digits({"model": "logreg", "scale": True, "n_train": 100}, 0) < acc - 0.03
    assert 0.75 < train_digits({"model": "tree", "scale": False, "n_train": 1000}, 0) < 0.92
    with pytest.raises(ValueError):
        train_digits({"model": "svm", "scale": False, "n_train": 100}, 0)


def fake_train(config, seed):
    return config["x"] * 10 + seed


def test_run_and_summarize():
    results = run_experiment(fake_train, [{"x": 1}, {"x": 2}], seeds=[0, 1, 2])
    assert [(r["config"]["x"], r["seed"], r["metric"]) for r in results] == \
        [(1, 0, 10), (1, 1, 11), (1, 2, 12), (2, 0, 20), (2, 1, 21), (2, 2, 22)]
    assert results[0]["config_id"] == config_id({"x": 1})
    s = summarize(results)
    assert s[config_id({"x": 2})] == {"config": {"x": 2}, "mean": 21, "std": 1.0, "n": 3}
    assert summarize(run_experiment(fake_train, [{"x": 1}], [5]))[config_id({"x": 1})]["std"] == 0.0


def test_decide():
    summary = {config_id({"t": 1}): {"config": {"t": 1}, "mean": 0.95, "std": 0.01, "n": 4},
               config_id({"t": 0}): {"config": {"t": 0}, "mean": 0.90, "std": 0.01, "n": 4}}
    h = lambda effect: Hypothesis("t helps", {"t": 1}, {"t": 0}, effect)
    assert decide(h(0.02), summary) == "supported"
    assert decide(h(0.08), summary) == "not supported"
    assert decide(h(0.045), summary) == "inconclusive"


@pytest.mark.timeout(60)
def test_digits_experiment():
    configs = grid({"model": ["logreg"], "scale": [True, False], "n_train": [100, 1000]})
    summary = summarize(run_experiment(train_digits, configs, seeds=range(5)))
    scaling = Hypothesis("Standardizing features improves logistic regression by >= 2 points",
                         treatment={"model": "logreg", "scale": True, "n_train": 1000},
                         control={"model": "logreg", "scale": False, "n_train": 1000}, min_effect=0.02)
    more_data = Hypothesis("10x more training data improves accuracy by >= 5 points",
                           treatment={"model": "logreg", "scale": True, "n_train": 1000},
                           control={"model": "logreg", "scale": True, "n_train": 100}, min_effect=0.05)
    assert decide(scaling, summary) == "not supported"
    scaling.min_effect = 0.01
    assert decide(scaling, summary) == "inconclusive", "5 seeds can't resolve a 1-point effect"
    assert decide(more_data, summary) == "supported"
