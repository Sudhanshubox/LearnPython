# m52 · Unsupervised learning: clustering and anomalies

**By the end you can:** group unlabelled data with k-means (with smart k-means++ initialization), choose the number of clusters with the elbow method and silhouette scores, evaluate clusters against known labels, and flag anomalies.

**Why it matters for AI:** most data has no labels. Clustering is used to segment customers, organize documents, deduplicate near-identical examples in training data, and explore embedding spaces ("what kinds of questions do users ask our chatbot?"). Anomaly detection catches fraud, sensor failures and broken data. And k-means itself reappears inside modern AI: vector databases use it to partition embeddings (IVF indexes), and it's used to quantize speech into discrete tokens.

---

## 1. k-means

Goal: split n points into k clusters so each point is close to its cluster's **centroid** (mean). It minimizes the **inertia**: the sum of squared distances from each point to its centroid.

**Lloyd's algorithm:**

1. Pick k initial centroids.
2. **Assign** each point to its nearest centroid (pairwise distances, m37).
3. **Update** each centroid to the mean of its assigned points.
4. Repeat 2–3 until the assignments stop changing (or a maximum number of iterations).

Each step can only lower the inertia, so it always converges, but to a *local* minimum that depends on the starting centroids.

**Empty clusters:** if a centroid gets no points, a common fix is to keep its old position (or move it to the point farthest from its centroid).

## 2. k-means++ initialization

Random initial centroids can land in the same blob and give poor results. **k-means++** spreads them out:

1. Choose the first centroid uniformly at random from the points.
2. For each point, compute D(x)², the squared distance to the nearest centroid chosen so far.
3. Choose the next centroid with probability proportional to D(x)² (m42's categorical sampling!): far points are more likely.
4. Repeat until k centroids are chosen.

It's both faster to converge and provably closer to the optimum on average. Running several initializations (`n_init`) and keeping the lowest inertia helps too.

## 3. Choosing k

- **Elbow method:** plot inertia against k. Inertia always decreases as k grows; look for the "elbow" where adding clusters stops helping much.
- **Silhouette score:** for each point, let a = its mean distance to the other points in its own cluster, and b = its mean distance to the points of the *nearest other* cluster. Its silhouette is (b − a) / max(a, b), between −1 (probably in the wrong cluster) and 1 (well clustered). Average over all points; higher is better.

## 4. Evaluating clusters against labels

When true labels exist (for checking), cluster IDs are arbitrary: cluster 3 might be "digit 7". **Purity** assigns each cluster its most common true label and measures the fraction of points matching. (More refined scores, like the adjusted Rand index, also correct for chance.)

## 5. k-means' assumptions

k-means works best for round, similarly sized, well-separated clusters, because it uses Euclidean distance to a single centre. It struggles with elongated or nested shapes (use DBSCAN or spectral clustering) and needs scaled features (m50). In high dimensions, reduce dimensions first (PCA, m39).

## 6. Anomaly detection

Simple, effective approaches:

- **z-scores:** flag points with a feature more than about 3 standard deviations from its mean.
- **Distance to centroid:** after clustering, flag the points farthest from their cluster's centroid.
- More advanced: isolation forests, and the reconstruction error of an autoencoder (Phase 6).

---

## Problem-solving habit #48: look at examples from each cluster

Cluster scores tell you little about *meaning*. Always print or plot a few examples from each cluster and name the clusters yourself. That's where the insight is, and it quickly reveals clusters formed by artefacts (like text length or a missing-value code) instead of real structure.

## Go deeper (optional, research-level)

1. Read *k-means++: The Advantages of Careful Seeding* (Arthur & Vassilvitskii, 2007), the introduction and main theorem. What approximation guarantee does k-means++ give?
2. Vector databases often use **IVF** indexes: k-means partitions the embeddings and a search only looks inside the nearest few clusters. Read the FAISS documentation on IndexIVFFlat. What's the speed vs recall trade-off?
3. Run k-means on m48's moons dataset. Why does it fail? Try `sklearn.cluster.DBSCAN` and explain the difference.

## Your turn

Open the **Exercises** tab. Implement with NumPy (vectorized distances); the tests compare with scikit-learn.
