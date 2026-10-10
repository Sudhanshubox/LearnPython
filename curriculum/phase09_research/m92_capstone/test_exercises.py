import copy
import datetime as dt

import pytest

from exercises import (
    PROJECT_TYPES,
    compute_cost,
    is_placeholder,
    load_proposal,
    rubric_score,
    schedule,
    validate_proposal,
)

GOOD = {
    "project": {"title": "Reversed digits for multiplication",
                "one_sentence": "Test whether reversed answers help a small GPT learn multiplication.",
                "type": "extension"},
    "question": {"research_question": "Do reversed answers raise exact-match accuracy on 2x2-digit multiplication?",
                 "hypothesis": "Reversed answers reach 20 points higher accuracy at 5,000 steps.",
                 "why_it_matters": "It tests whether the format trick generalizes beyond addition."},
    "method": {"approach": "Train minigpt on both formats with equal budgets.", "baselines": ["plain format"], "ablations": []},
    "evaluation": {"metrics": ["exact match"], "datasets": ["generated problems"],
                   "success_criterion": "Paired 95% CI excluding 0 over 5 seeds.", "seeds": 5},
    "compute": {"gpu_hours": 0, "api_budget_usd": 0.0},
    "milestones": [{"week": w, "deliverable": d} for w, d in
                   [(1, "data pipeline"), (2, "pilot"), (4, "main runs"), (6, "report")]],
    "risks": [{"risk": "no learning", "mitigation": "smaller problems"},
              {"risk": "high variance", "mitigation": "more seeds"}],
}


def changed(path, value):
    p = copy.deepcopy(GOOD)
    target = p
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return p


def test_is_placeholder():
    assert is_placeholder("") and is_placeholder("   ") and is_placeholder("TODO: fill in")
    assert is_placeholder("we will decide (TBD)") and is_placeholder("...")
    assert not is_placeholder("A specific question about LoRA rank")


def test_validate_good_proposal():
    assert validate_proposal(GOOD) == []


def test_validate_text_and_lists():
    assert validate_proposal(changed(("question", "hypothesis"), "TODO")) == ["question.hypothesis: write it"]
    assert validate_proposal(changed(("method", "approach"), 42)) == ["method.approach: write it"]
    assert validate_proposal(changed(("evaluation", "metrics"), [])) == ["evaluation.metrics: list at least one"]
    assert validate_proposal(changed(("method", "baselines"), ["TODO"])) == ["method.baselines: list at least one"]
    assert validate_proposal(changed(("project", "type"), "survey")) == ["project.type: one of " + ", ".join(sorted(PROJECT_TYPES))]
    assert validate_proposal(changed(("project", "one_sentence"), "word " * 41)) == ["project.one_sentence: at most 40 words"]
    p = copy.deepcopy(GOOD)
    del p["question"]
    assert validate_proposal(p)[:3] == ["question.research_question: write it", "question.hypothesis: write it",
                                         "question.why_it_matters: write it"]


def test_validate_numbers():
    assert validate_proposal(changed(("evaluation", "seeds"), 1)) == ["evaluation.seeds: at least 3"]
    assert validate_proposal(changed(("evaluation", "seeds"), 3.5)) == ["evaluation.seeds: at least 3"]
    assert validate_proposal(changed(("compute", "gpu_hours"), -1)) == ["compute.gpu_hours: a number >= 0"]
    p = copy.deepcopy(GOOD)
    del p["compute"]
    assert validate_proposal(p) == ["compute.gpu_hours: a number >= 0", "compute.api_budget_usd: a number >= 0"]


def test_validate_milestones_and_risks():
    ms = GOOD["milestones"]
    assert validate_proposal(changed(("milestones",), ms[:3])) == ["milestones: at least 4"]
    swapped = [ms[0], ms[2], ms[1], ms[3]]
    assert validate_proposal(changed(("milestones",), swapped)) == ["milestones: weeks must be increasing integers from 1"]
    long = ms[:3] + [{"week": 20, "deliverable": "report"}]
    assert validate_proposal(changed(("milestones",), long)) == ["milestones: keep the plan within 16 weeks"]
    vague = ms[:3] + [{"week": 6, "deliverable": "TODO"}]
    assert validate_proposal(changed(("milestones",), vague)) == ["milestones: every milestone needs a concrete deliverable"]
    assert validate_proposal(changed(("risks",), GOOD["risks"][:1])) == ["risks: at least 2, each with a mitigation"]
    no_plan = [GOOD["risks"][0], {"risk": "data loss", "mitigation": ""}]
    assert validate_proposal(changed(("risks",), no_plan)) == ["risks: at least 2, each with a mitigation"]


def test_template_is_not_ready(tmp_path):
    template = load_proposal()
    if not validate_proposal(template):
        pytest.skip("your proposal is already complete")
    assert len(validate_proposal(template)) > 5


def test_schedule():
    due = schedule("2026-11-02", GOOD["milestones"])
    assert due[0] == (dt.date(2026, 11, 8), "data pipeline")
    assert due[-1] == (dt.date(2026, 12, 13), "report")


def test_compute_cost():
    assert compute_cost(10, 25) == 40.0
    assert compute_cost(3.333, 0, gpu_price_per_hour=0.9) == 3.0


def test_rubric_score():
    assert rubric_score({k: 4 for k in ["question", "baselines", "rigor", "results", "writing", "code"]}) == 100.0
    mixed = {"question": 3, "baselines": 2, "rigor": 4, "results": 3, "writing": 2, "code": 1}
    assert rubric_score(mixed) == pytest.approx(15 * 0.75 + 15 * 0.5 + 20 + 20 * 0.75 + 15 * 0.5 + 15 * 0.25, abs=0.05)
    with pytest.raises(ValueError):
        rubric_score({"question": 5})
    with pytest.raises(ValueError):
        rubric_score({**mixed, "code": -1})


def test_my_proposal():
    """Passes once you've filled in proposal.toml (README section 2)."""
    problems = validate_proposal(load_proposal())
    assert problems == [], "Finish proposal.toml:\n" + "\n".join(problems)
