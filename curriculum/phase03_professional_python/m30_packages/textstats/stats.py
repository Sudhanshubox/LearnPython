"""Statistics built on top of the tokens module.

Import words and sentences from the sibling module with a RELATIVE import.
"""


def word_count(text):
    """Number of words."""
    raise NotImplementedError


def top_words(text, n=3):
    """The n most common words as (word, count) pairs, most common first,
    ties broken alphabetically."""
    raise NotImplementedError


def avg_sentence_length(text):
    """Average number of words per sentence, rounded to 2 decimals (0.0 for no sentences)."""
    raise NotImplementedError
