"""Scoring functions. Each takes the index, the query TERMS (already tokenized),
and a doc_id, and returns a float. Repeated query terms count once (use set(terms)).
Terms that aren't in the index contribute 0. Use math.log (natural log)."""

import math

from .index import InvertedIndex


def tf_idf(index: InvertedIndex, terms: list[str], doc_id: str) -> float:
    """Sum over the unique query terms t of tf(t, doc) * ln(N / df(t))."""
    raise NotImplementedError


def bm25(index: InvertedIndex, terms: list[str], doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
    """Sum over the unique query terms t present in the document of
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    where idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), dl = the document's length,
    avgdl = the average document length."""
    raise NotImplementedError
