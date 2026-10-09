"""m35: evaluation metrics for your search engine.

Most of this project lives in the searchkit/ package (use the file picker).
Start with searchkit/text.py and work through the files in the order the README gives.
"""


def precision_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the first k results that are relevant. If there are fewer than k
    results, still divide by k. precision_at_k(["a", "b", "c"], {"a", "c"}, 2) -> 0.5"""
    raise NotImplementedError


def reciprocal_rank(results: list[str], relevant: set[str]) -> float:
    """1 / (1-based rank of the first relevant result), or 0.0 if none is relevant."""
    raise NotImplementedError


def mean_reciprocal_rank(all_results: list[list[str]], all_relevant: list[set[str]]) -> float:
    """Average reciprocal_rank over several queries (0.0 if there are no queries)."""
    raise NotImplementedError
