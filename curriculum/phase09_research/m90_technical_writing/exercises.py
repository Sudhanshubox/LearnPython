"""m90 exercises: tools that check your writing, tables and figures."""

import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLAIM_WORDS = ("outperform", "improve", "better", "best", "state-of-the-art", "state of the art",
               "significant", "faster", "superior", "surpass")


# 1. "MEAN ± STD" of values multiplied by scale, both with `decimals` decimals (sample std;
#    0.0 for a single value). format_mean_std([0.961, 0.969, 0.959], 1, 100) == "96.3 ± 0.5"
def format_mean_std(values, decimals=1, scale=1.0):
    raise NotImplementedError


# 2. A Markdown results table (README section 4). results: {method: {metric: [per-seed values]}}
#    (methods in insertion order). Header "| Method | M1 | M2 |", then a separator line with one
#    "|---" per column (Method included) followed by "|", then one row per method with
#    format_mean_std cells. In each column, the method with the best MEAN (highest if
#    higher_is_better[metric], else lowest) has its cell wrapped in ** **.
def results_table(results, metrics, higher_is_better, decimals=1, scale=1.0):
    raise NotImplementedError


def split_sentences(text):
    """Given: split text into sentences (after . ! or ?, before a capital letter, [ or ()."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(])", text.strip()) if s.strip()]


# 3. The sentences (from split_sentences) that make a claim (contain one of CLAIM_WORDS,
#    case-insensitive, also as part of a longer word like "outperforms") but have no evidence:
#    no digit, no citation like [3], and no "Table N" / "Figure N" / "Fig. N" reference
#    (case-insensitive).
def unsupported_claims(text):
    raise NotImplementedError


# 4. Syllables, approximately: the number of groups of consecutive vowels (aeiouy) in the
#    lowercased word, minus one for a silent final "e" (a word ending in "e" but not "le" or
#    "ee", if that leaves at least one group); at least 1.
#    flesch_reading_ease(text): 206.835 - 1.015 * words/sentences - 84.6 * syllables/words,
#    with words = runs of letters, sentences = max(1, len(split_sentences(text))); 0.0 if
#    there are no words.
def count_syllables(word):
    raise NotImplementedError


def flesch_reading_ease(text):
    raise NotImplementedError


# 5. Problems with a matplotlib figure (README section 5), checking each axes i in fig.axes:
#    "axes i: missing x label", "axes i: missing y label", and, when it has more than one
#    line, "axes i: several lines but no legend" if there's no legend.
def figure_problems(fig):
    raise NotImplementedError


# 6. A learning-curve figure. curves: {name: list of runs}, each run a list of values per step.
#    For each name plot the mean over runs against steps 1..T with label f"{name} (n={runs})",
#    and shade mean ± sample std with fill_between (alpha 0.2; no shading needed for one run).
#    Label the axes with xlabel / ylabel, add a legend, and return the figure.
def learning_curve_figure(curves, xlabel="Training step", ylabel="Validation loss"):
    raise NotImplementedError


# 7. Check a Markdown draft's structure: headings are lines starting with 1-6 "#" and a space.
#    Return (missing, out_of_order): the required section names (case-insensitive match with a
#    heading's text) that don't appear, in required order; and whether the ones that do
#    appear are in a different order than required.
def missing_sections(markdown, required):
    raise NotImplementedError
