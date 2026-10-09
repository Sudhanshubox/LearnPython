"""Splitting text into words and sentences."""

import re


def words(text):
    """Lowercased words: runs of letters, digits and apostrophes.

    words("It's 2 o'clock, Asha!") -> ["it's", "2", "o'clock", "asha"]
    Hint: re.findall with the pattern r"[a-z0-9']+" on the lowercased text.
    """
    raise NotImplementedError


def sentences(text):
    """Non-empty sentences, split after ".", "!" or "?", stripped of whitespace.

    sentences("Hi there. How are you?  Fine!") -> ["Hi there.", "How are you?", "Fine!"]
    Text after the last punctuation mark counts as a sentence too.
    """
    raise NotImplementedError
