import numpy as np
import pytest
from sklearn import metrics
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_digits, make_blobs
from sklearn.decomposition import PCA

import exercises
from code_checks import has_loops
from exercises import (
    KMeans,
    centroid_outliers,
    elbow,
    kmeans_plus_plus,
    purity,
    silhouette_score,
    sq_distances,
    zscore_outliers,
)


@pytest.fixture(scope="module")
def blobs():
    return make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=7)


@pytest.mark.parametrize("name", ["sq_distances", "silhouette_score", "zscore_outliers", "centroid_outliers"])
def test_vectorized(name):
    assert not has_loops(getattr(exercises, name))


def test_sq_distances():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    C = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(sq_distances(X, C), [[0, 2], [25, 13]])


def test_kmeans_plus_plus(blobs):
    X, y = blobs
    C = kmeans_plus_plus(X, 4, np.random.default_rng(0))
    assert C.shape == (4, 2)
    assert all((X == c).all(axis=1).any() for c in C), "centers are chosen from the data points"
    # spread out: the 4 centers usually land in 4 different blobs
    nearest = sq_distances(C, np.array([X[y == i].mean(axis=0) for i in range(4)])).argmin(axis=1)
    assert len(set(nearest)) >= 3


def test_kmeans_finds_blobs(blobs):
    X, y = blobs
    km = KMeans(4, seed=0)
    assert km.fit(X) is km
    assert km.cluster_centers_.shape == (4, 2)
    assert purity(y, km.labels_) > 0.97
    ref = SkKMeans(4, n_init=5, random_state=0).fit(X)
    assert km.inertia_ == pytest.approx(ref.inertia_, rel=0.01)
    assert km.inertia_ == pytest.approx(sq_distances(X, km.cluster_centers_).min(axis=1).sum())
    np.testing.assert_array_equal(km.predict(X), km.labels_)
    assert 1 <= km.n_iter_ <= 100


def test_kmeans_centers_are_cluster_means(blobs):
    X, _ = blobs
    km = KMeans(3, seed=1).fit(X)
    for c in range(3):
        np.testing.assert_allclose(km.cluster_centers_[c], X[km.labels_ == c].mean(axis=0))


def test_elbow(blobs):
    X, _ = blobs
    inertias = elbow(X, [1, 2, 3, 4, 5, 6])
    assert all(a >= b for a, b in zip(inertias, inertias[1:]))
    drops = -np.diff(inertias)
    assert drops[3] < drops[2] / 5, "after the true k=4, adding clusters barely helps (the elbow)"


def test_silhouette_matches_sklearn(blobs):
    X, y = blobs
    assert silhouette_score(X, y) == pytest.approx(metrics.silhouette_score(X, y))
    labels = KMeans(2, seed=0).fit(X).labels_
    assert silhouette_score(X, labels) == pytest.approx(metrics.silhouette_score(X, labels))


def test_silhouette_picks_right_k(blobs):
    X, _ = blobs
    scores = {k: silhouette_score(X, KMeans(k, seed=0).fit(X).labels_) for k in range(2, 7)}
    assert max(scores, key=scores.get) == 4


def test_purity():
    assert purity(np.array([0, 0, 1, 1]), np.array([5, 5, 9, 9])) == 1.0
    assert purity(np.array([0, 1, 0, 1]), np.array([3, 3, 3, 3])) == 0.5


@pytest.mark.timeout(60)
def test_clustering_digits_without_labels():
    X, y = load_digits(return_X_y=True)
    Z = PCA(n_components=20, random_state=0).fit_transform(X)
    km = KMeans(10, n_init=3, seed=0).fit(Z)
    assert purity(y, km.labels_) > 0.7, "k-means on PCA features groups most digits correctly without labels"


def test_zscore_outliers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    X[10] = [0, 9, 0]
    X[20] = [-8, 0, 0]
    mask = zscore_outliers(X, 4.0)
    assert mask[10] and mask[20]
    assert mask.sum() <= 4


def test_centroid_outliers(blobs):
    X, _ = blobs
    X = np.vstack([X, [[40.0, 40.0]]])
    km = KMeans(4, seed=0).fit(X)
    mask = centroid_outliers(X, km.cluster_centers_, km.labels_, fraction=0.01)
    assert mask.sum() == 6
    assert mask[-1]
