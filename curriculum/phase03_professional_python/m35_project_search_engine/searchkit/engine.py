"""The search engine: documents + index + ranking."""

import json
from dataclasses import dataclass
from pathlib import Path

from .index import InvertedIndex
from .ranking import bm25, tf_idf
from .text import tokenize


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float
    title: str
    snippet: str


class SearchEngine:
    def __init__(self, ranker: str = "bm25") -> None:
        """ranker is "bm25" or "tfidf"; anything else raises ValueError."""
        raise NotImplementedError

    def add_document(self, doc_id: str, text: str, title: str | None = None) -> None:
        """Store and index a document. The title defaults to the doc_id."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of documents."""
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Tokenize the query, score every document that contains at least one query
        term, and return the top k results with score > 0, highest score first
        (ties: doc_id alphabetically). Scores are rounded to 3 decimals.
        The snippet is the first 80 characters of the document's text, with newlines
        replaced by spaces, stripped. An empty query returns []."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save as JSON: {"ranker": ..., "documents": [{"id", "title", "text"}, ...]}
        in the order documents were added."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "SearchEngine":
        """Rebuild an engine saved with save()."""
        raise NotImplementedError
