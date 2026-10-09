"""Turning text into terms."""

import re

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or so such that the "
    "their then there these this to was were which while will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase the text, find runs of letters and digits (regex [a-z0-9]+),
    and drop STOPWORDS. Order and duplicates are kept.

    tokenize("The cat, and THE hat!") -> ["cat", "hat"]
    """
    raise NotImplementedError
