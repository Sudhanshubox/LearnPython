import ast

import numpy as np
import pytest
from sklearn import linear_model as sk
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SkScaler

import exercises
from exercises import LinearRegression, LogisticRegression, Ridge, SoftmaxRegression, StandardScaler


def test_classes_use_numpy_only():
    tree = ast.parse(open(exercises.__file__, encoding="utf-8").read())
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    mods += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m.startswith("sklearn") for m in mods)


@pytest.fixture(scope="module")
def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=0)


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    scaler = SkScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte), ytr, yte


def test_standard_scaler():
    X = np.array([[1.0, 5.0, 2.0], [3.0, 5.0, 4.0], [5.0, 5.0, 9.0]])
    s = StandardScaler()
    assert s.fit(X) is s
    np.testing.assert_allclose(s.transform(X), SkScaler().fit_transform(X))
    np.testing.assert_allclose(s.scale_[1], 1.0)
    np.testing.assert_allclose(StandardScaler().fit_transform(X), s.transform(X))
    np.testing.assert_allclose(s.transform(np.array([[3.0, 5.0, 5.0]])), [[0, 0, 0]], atol=1e-12)


def test_linear_regression_matches_sklearn(diabetes):
    Xtr, Xte, ytr, yte = diabetes
    mine = LinearRegression()
    assert mine.fit(Xtr, ytr) is mine
    ref = sk.LinearRegression().fit(Xtr, ytr)
    np.testing.assert_allclose(mine.coef_, ref.coef_, rtol=1e-6)
    assert mine.intercept_ == pytest.approx(ref.intercept_)
    assert mine.score(Xte, yte) == pytest.approx(ref.score(Xte, yte))


@pytest.mark.parametrize("alpha", [0.01, 1.0, 10.0])
def test_ridge_matches_sklearn(diabetes, alpha):
    Xtr, Xte, ytr, yte = diabetes
    mine = Ridge(alpha=alpha).fit(Xtr, ytr)
    ref = sk.Ridge(alpha=alpha).fit(Xtr, ytr)
    np.testing.assert_allclose(mine.coef_, ref.coef_, rtol=1e-6)
    assert mine.intercept_ == pytest.approx(ref.intercept_)
    np.testing.assert_allclose(mine.predict(Xte), ref.predict(Xte))


def test_ridge_shrinks_weights(diabetes):
    Xtr, _, ytr, _ = diabetes
    norms = [np.linalg.norm(Ridge(alpha=a).fit(Xtr, ytr).coef_) for a in (0.001, 0.1, 10, 1000)]
    assert norms == sorted(norms, reverse=True)


def test_logistic_regression(cancer):
    Xtr, Xte, ytr, yte = cancer
    mine = LogisticRegression(lr=0.1, epochs=2000, l2=0.01)
    assert mine.fit(Xtr, ytr) is mine
    assert mine.score(Xte, yte) > 0.95
    proba = mine.predict_proba(Xte)
    assert proba.shape == (len(Xte),) and ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_array_equal(mine.predict(Xte), (proba >= 0.5).astype(int))
    assert len(mine.losses_) == 2000
    assert mine.losses_[-1] < mine.losses_[0] / 5
    ref = sk.LogisticRegression(C=1 / (0.01 * len(Xtr)), max_iter=5000).fit(Xtr, ytr)
    agreement = np.mean(mine.predict(Xte) == ref.predict(Xte))
    assert agreement > 0.97


def test_logistic_first_loss_is_log2():
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)
    m = LogisticRegression(epochs=5).fit(X, y)
    assert m.losses_[0] == pytest.approx(np.log(2)), "starting from zeros, p = 0.5 for everyone"


def test_softmax_regression_digits():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    s = StandardScaler().fit(Xtr)
    Xtr, Xte = s.transform(Xtr), s.transform(Xte)
    mine = SoftmaxRegression(lr=0.5, epochs=300, l2=1e-3).fit(Xtr, ytr)
    assert mine.coef_.shape == (64, 10) and mine.intercept_.shape == (10,)
    np.testing.assert_array_equal(mine.classes_, np.arange(10))
    P = mine.predict_proba(Xte)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    assert mine.score(Xte, yte) > 0.93
    assert mine.losses_[0] == pytest.approx(np.log(10))
    assert all(a >= b - 1e-9 for a, b in zip(mine.losses_[::50], mine.losses_[50::50]))


def test_softmax_matches_logistic_on_two_classes(cancer):
    Xtr, Xte, ytr, yte = cancer
    two = SoftmaxRegression(lr=0.1, epochs=800).fit(Xtr, ytr)
    one = LogisticRegression(lr=0.1, epochs=800).fit(Xtr, ytr)
    assert np.mean(two.predict(Xte) == one.predict(Xte)) > 0.97


def test_softmax_is_stable():
    X = np.array([[1000.0], [-1000.0]])
    m = SoftmaxRegression(epochs=3).fit(X, np.array([0, 1]))
    assert np.isfinite(m.predict_proba(X)).all()
