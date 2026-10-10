"""m86 exercises: tools for reading papers critically."""

import math
import re
import statistics
from dataclasses import dataclass, field

STOPWORDS = {"a", "an", "the", "on", "of", "for", "to", "in", "is", "are", "and", "with", "via", "by", "from"}


# 1. Find a new-style arXiv ID (4 digits, a dot, 4 or 5 digits, optionally "v" + version)
#    anywhere in text: URLs like "https://arxiv.org/abs/1706.03762v7", "arXiv:2307.03381",
#    or a bare "1512.03385". Return (id, version or None), e.g. ("1706.03762", 7), or None if
#    there's no ID. The ID must not be part of a longer number (like "12.34567.8").
def parse_arxiv_id(text):
    raise NotImplementedError


@dataclass
class PaperNote:
    """Given: a structured note about one paper (README section 2)."""
    title: str
    authors: list
    year: int
    arxiv_id: str = ""
    passes: int = 1                 # how many of the three passes you've done
    problem: str = ""
    method: str = ""
    claims: list = field(default_factory=list)
    evidence: list = field(default_factory=list)       # evidence[i] supports claims[i]
    limitations: list = field(default_factory=list)
    questions: list = field(default_factory=list)


# 2. Check a note and return a list of problems (empty if fine), in this order:
#    "missing title" (blank), "missing authors", "passes must be 1, 2 or 3";
#    if passes >= 2: "missing problem", "missing method" (blank strings), "missing claims",
#      and "every claim needs evidence" if there are fewer evidence items than claims;
#    if passes == 3: "missing limitations", "missing questions".
def validate_note(note):
    raise NotImplementedError


# 3. The BibTeX key: the first author's last name (lowercased, letters only) + year + the first
#    word of the title (lowercased, letters only) that isn't in STOPWORDS.
#    "Attention Is All You Need", Ashish Vaswani, 2017 -> "vaswani2017attention".
#    to_bibtex(note): exactly
#      @article{KEY,
#        title = {{TITLE}},
#        author = {A and B and C},
#        year = {YEAR},
#        eprint = {ID},              <- these two lines only if arxiv_id is set
#        archivePrefix = {arXiv}
#      }
#    (each field line indented by two spaces, lines joined with ",\n", then "\n}").
def bibtex_key(note):
    raise NotImplementedError


def to_bibtex(note):
    raise NotImplementedError


def note_markdown(note):
    """Given: render a note as Markdown for your reading log."""
    lines = [f"# {note.title}", "", f"{', '.join(note.authors)} ({note.year})"]
    if note.arxiv_id:
        lines.append(f"arXiv: {note.arxiv_id}")
    lines += ["", f"Passes completed: {note.passes}", "", "## Problem", note.problem or "-", "", "## Method", note.method or "-", ""]
    lines.append("## Claims and evidence")
    for i, claim in enumerate(note.claims):
        ev = note.evidence[i] if i < len(note.evidence) else "(no evidence noted)"
        lines.append(f"- {claim} (evidence: {ev})")
    for title, items in (("Limitations", note.limitations), ("Questions", note.questions)):
        lines += ["", f"## {title}"] + [f"- {x}" for x in items]
    return "\n".join(lines) + "\n"


# 4. Reading-list prioritization (README section 4). Papers are dicts with "title", "year",
#    optional "abstract" and "citations".
#    relevance(paper, interests): the fraction of the (lowercased) interest words that appear
#      among the words of title + abstract (words = runs of letters a-z after lowercasing,
#      minus STOPWORDS). 0.0 if interests is empty.
#    priority = 3 * relevance + log10(1 + citations) / sqrt(1 + age), age = max(0, current_year - year).
#    reading_list: the top_n papers by priority, highest first (ties: by title).
def relevance(paper, interests):
    raise NotImplementedError


def priority(paper, interests, current_year):
    raise NotImplementedError


def reading_list(papers, interests, current_year, top_n=5):
    raise NotImplementedError


# 5. Recompute a results table (README section 5). rows: dicts with "name", "scores" (a list)
#    and "reported_average". Return [(name, recomputed mean rounded to 3 decimals,
#    reported_average)] for the rows where they differ by more than tolerance.
def table_inconsistencies(rows, tolerance=0.05):
    raise NotImplementedError


# 6. Is a reported improvement bigger than seed noise? Given the per-seed scores of a baseline
#    and of a method, return {"difference": mean(method) - mean(baseline),
#    "seed_std": the larger of the two sample standard deviations (statistics.stdev),
#    "convincing": difference > 2 * seed_std}.
def improvement_vs_noise(baseline_runs, method_runs):
    raise NotImplementedError
