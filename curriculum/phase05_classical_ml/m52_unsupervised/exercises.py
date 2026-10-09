"""m52 exercises: clustering and anomaly detection (NumPy only)."""

import numpy as np


# 1. Squared Euclidean distances between every row of X (n, d) and every center (k, d),
#    as an (n, k) array, vectorized (m37).
def sq_distances(X, centers):
    raise NotImplementedError


# 2. k-means++ initialization. With the given rng: choose the first center with
#    rng.integers(n); then repeatedly compute each point's squared distance to its nearest
#    chosen center, and choose the next center with rng.choice(n, p=d2 / d2.sum()).
#    Return the (k, d) array of centers.
def kmeans_plus_plus(X, k, rng):
    raise NotImplementedError


# 3. k-means (Lloyd's algorithm).
#    fit(X): rng = np.random.default_rng(seed); for each of n_init runs, initialize with
#    kmeans_plus_plus(X, k, rng), then alternate assign/update for up to max_iter iterations,
#    stopping early when assignments don't change. A cluster with no points keeps its old
#    center. Keep the run with the lowest inertia.
#    After fit: cluster_centers_ (k, d), labels_ (n,), inertia_ (float), n_iter_ (iterations
#    of the best run). predict(X) -> index of the nearest center.
class KMeans:
    def __init__(self, k, n_init=5, max_iter=100, seed=0):
        raise NotImplementedError

    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


# 4. Inertia for each k in ks: KMeans(k, seed=seed).fit(X).inertia_, as a list.
def elbow(X, ks, seed=0):
    raise NotImplementedError


# 5. Mean silhouette score (see the README), vectorized using the full pairwise distance
#    matrix (Euclidean, NOT squared). Points in a cluster of size 1 get silhouette 0.
def silhouette_score(X, labels):
    raise NotImplementedError


# 6. Purity: for each cluster, count its most common true label; sum those counts and
#    divide by the number of points.
def purity(true_labels, cluster_labels):
    raise NotImplementedError


# 7. Anomalies.
#    zscore_outliers(X, threshold): boolean mask of rows where ANY feature's z-score
#      (using each column's mean and population std) has absolute value > threshold.
#    centroid_outliers(X, centers, labels, fraction): boolean mask marking the given fraction
#      of points (rounded down, at least 1) that are FARTHEST from their own center.
def zscore_outliers(X, threshold=3.0):
    raise NotImplementedError


def centroid_outliers(X, centers, labels, fraction=0.05):
    raise NotImplementedError
