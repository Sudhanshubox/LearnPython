"""retrievers: keyword (TF-IDF) and vector (LSA) retrievers over chunks, given for m80.

Both have the same interface:
    retriever = KeywordRetriever(chunks)          # chunks: a list of dicts with a "text" key
    retriever.search("question", k=5)             # -> list of chunk indices, best first
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordRetriever:
    def __init__(self, chunks):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in chunks])

    def search(self, query, k=5):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


class VectorRetriever:
    def __init__(self, chunks, dim=128, seed=0):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        X = self.vectorizer.fit_transform([c["text"] for c in chunks])
        self.svd = TruncatedSVD(dim, random_state=seed).fit(X)
        self.vectors = self._normalize(self.svd.transform(X))

    @staticmethod
    def _normalize(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(norms == 0, 1.0, norms)

    def search(self, query, k=5):
        q = self._normalize(self.svd.transform(self.vectorizer.transform([query])))[0]
        return [int(i) for i in np.argsort(-(self.vectors @ q), kind="stable")[:k]]
