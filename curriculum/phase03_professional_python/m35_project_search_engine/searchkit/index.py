"""An inverted index: term -> {doc_id: term frequency}."""

from .text import tokenize


class InvertedIndex:
    def __init__(self) -> None:
        raise NotImplementedError

    def add(self, doc_id: str, text: str) -> None:
        """Index a document. Adding an existing doc_id raises ValueError."""
        raise NotImplementedError

    def postings(self, term: str) -> dict[str, int]:
        """{doc_id: how many times term occurs in it}; {} for unknown terms.
        Must not let callers modify the index through the returned dict."""
        raise NotImplementedError

    def doc_freq(self, term: str) -> int:
        """Number of documents containing term."""
        raise NotImplementedError

    @property
    def num_docs(self) -> int:
        raise NotImplementedError

    def doc_length(self, doc_id: str) -> int:
        """Number of terms (after tokenizing) in the document."""
        raise NotImplementedError

    @property
    def avg_doc_length(self) -> float:
        """Average document length (0.0 for an empty index)."""
        raise NotImplementedError

    def __contains__(self, doc_id: object) -> bool:
        raise NotImplementedError
