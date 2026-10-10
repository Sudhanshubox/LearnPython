"""m79 exercises: embeddings and vector search over this course's lessons."""

import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())


# 1. Split text on blank lines (a line containing only whitespace counts as blank) and return
#    the stripped paragraphs with at least min_chars characters.
def split_paragraphs(text, min_chars=80):
    raise NotImplementedError


def load_chunks(pattern="phase0[1-7]*/*/README.md"):
    """Given: (module_id, paragraph) pairs for every lesson, e.g. ("m08", "...")."""
    chunks = []
    for path in sorted(CURRICULUM.glob(pattern)):
        module = path.parent.name.split("_")[0]
        for para in split_paragraphs(path.read_text(encoding="utf-8")):
            chunks.append((module, para))
    return chunks


# 2. L2-normalize the rows of a matrix (or a single vector, along its last axis). Rows of all
#    zeros stay zero (no division by zero).
def normalize(M):
    raise NotImplementedError


# 3. Latent semantic analysis (README section 2).
#    fit(texts): TfidfVectorizer(sublinear_tf=True, stop_words="english") fitted on texts, then
#      TruncatedSVD(self.dim, random_state=self.seed) fitted on the TF-IDF matrix. Return self.
#    encode(texts): transform with both fitted objects and normalize the rows.
class LSAEmbedder:
    def __init__(self, dim=128, seed=0):
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        raise NotImplementedError

    def encode(self, texts):
        raise NotImplementedError


# 4. Exact top-k by dot product for a batch of queries (q, d) against matrix (n, d).
#    Return (indices, scores), both (q, k), best first (k is capped at n). Use np.argpartition
#    so you never fully sort all n scores. No Python loops.
def top_k(queries, matrix, k):
    raise NotImplementedError


# 5. A flat (exact) index. add(vectors, ids) normalizes and appends; search(query, k) returns
#    [(id, score), ...] best first.
class FlatIndex:
    def __init__(self):
        raise NotImplementedError

    def add(self, vectors, ids):
        raise NotImplementedError

    def search(self, query, k=5):
        raise NotImplementedError


# 6. An IVF index (README section 4).
#    train(vectors): KMeans(n_lists, n_init=3, random_state=seed) on the normalized vectors;
#      keep the NORMALIZED cluster centres as self.centroids and empty lists. Return self.
#    add(vectors, ids): normalize and put each vector in the list of the centroid with the
#      highest dot product.
#    search(query, k): scan only the n_probe lists whose centroids score highest for the
#      query; set self.last_scanned to the number of vectors scanned; return [(id, score)]
#      best first, like FlatIndex.
class IVFIndex:
    def __init__(self, n_lists=16, n_probe=1, seed=0):
        self.n_lists, self.n_probe, self.seed = n_lists, n_probe, seed
        self.last_scanned = 0

    def train(self, vectors):
        raise NotImplementedError

    def add(self, vectors, ids):
        raise NotImplementedError

    def search(self, query, k=5):
        raise NotImplementedError


# 7. Mean recall@k: for each query, |approx ∩ exact| / |exact|, averaged over queries.
#    approx and exact are lists of lists of ids.
def recall_at_k(approx, exact):
    raise NotImplementedError


# 8. The fraction of queries whose expected source appears in their list of retrieved sources.
def hit_rate(retrieved_sources, expected):
    raise NotImplementedError
