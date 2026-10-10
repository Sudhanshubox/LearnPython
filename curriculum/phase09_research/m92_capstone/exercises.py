"""m92 exercises: plan and assess your capstone.

Fill in proposal.toml too: test_my_proposal checks it with your validate_proposal.
"""

import datetime as dt
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
PROJECT_TYPES = {"reproduction", "extension", "original", "application"}
PLACEHOLDERS = ("todo", "tbd", "xxx", "...")

REQUIRED_TEXT = [("project", "title"), ("project", "one_sentence"), ("question", "research_question"),
                 ("question", "hypothesis"), ("question", "why_it_matters"), ("method", "approach"),
                 ("evaluation", "success_criterion")]
REQUIRED_LISTS = [("method", "baselines"), ("evaluation", "metrics"), ("evaluation", "datasets")]


def load_proposal(path=None):
    """Given: read proposal.toml (or another path) into a dict."""
    path = Path(path) if path else HERE / "proposal.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


# 1. True if text is empty after stripping, or contains (case-insensitively) any of PLACEHOLDERS.
def is_placeholder(text):
    raise NotImplementedError


# 2. Check a proposal dict (as loaded from proposal.toml) and return a list of problems, in this
#    order (an empty list means it's ready):
#    - for each (section, key) in REQUIRED_TEXT whose value is missing, not a string, or a
#      placeholder: "SECTION.KEY: write it"
#    - for each in REQUIRED_LISTS that is missing, not a list, empty, or contains a placeholder
#      (check str(item)): "SECTION.KEY: list at least one"
#    - project.type not in PROJECT_TYPES: "project.type: one of " + ", ".join(sorted(PROJECT_TYPES))
#    - project.one_sentence longer than 40 words: "project.one_sentence: at most 40 words"
#    - evaluation.seeds not an int >= 3: "evaluation.seeds: at least 3"
#    - for key in ("gpu_hours", "api_budget_usd"): compute.KEY not a number >= 0:
#      "compute.KEY: a number >= 0"
#    - milestones (a list of tables with "week" and "deliverable"): fewer than 4 ->
#      "milestones: at least 4"; otherwise, weeks not strictly increasing integers starting at
#      >= 1 -> "milestones: weeks must be increasing integers from 1"; otherwise the last week
#      > 16 -> "milestones: keep the plan within 16 weeks".
#      Separately, any placeholder deliverable -> "milestones: every milestone needs a concrete deliverable"
#    - fewer than 2 risks, or any risk/mitigation that's a placeholder:
#      "risks: at least 2, each with a mitigation"
def validate_proposal(p):
    raise NotImplementedError


# 3. Due dates: for each milestone, (start date + week weeks - 1 day, deliverable), where start
#    is an ISO date string like "2026-11-02". Return a list of (datetime.date, str).
def schedule(start, milestones):
    raise NotImplementedError


# 4. Estimated cost in dollars: gpu_hours * gpu_price_per_hour + api_budget_usd, rounded to 2 decimals.
def compute_cost(gpu_hours, api_budget_usd, gpu_price_per_hour=1.5):
    raise NotImplementedError


# 5. The README's self-assessment rubric: each criterion rated 0-4 contributes
#    weight * rating / 4. Raise ValueError if a criterion is missing or outside 0-4.
#    Return the total rounded to 1 decimal (out of 100).
RUBRIC = {"question": 15, "baselines": 15, "rigor": 20, "results": 20, "writing": 15, "code": 15}


def rubric_score(self_assessment):
    raise NotImplementedError
