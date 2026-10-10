import re

import numpy as np
import pytest

from code_checks import LOOP_NODES, contains
from exercises import (
    JUDGE_MODEL,
    JUDGE_TEMPLATE,
    bootstrap_ci,
    cohens_kappa,
    exact_match,
    judge_prompt,
    llm_judge,
    normalize_answer,
    paired_bootstrap,
    pairwise_judge,
    parse_judge,
    run_eval,
    token_f1,
)
from fakeclaude import FakeClaude, text_reply


def test_normalize_answer():
    assert normalize_answer("  The Eiffel   Tower! ") == "eiffel tower"
    assert normalize_answer("An apple, a day.") == "apple day"
    assert normalize_answer("theory") == "theory", "only whole words are articles"


def test_exact_match_and_f1():
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match("Paris", "London") == 0.0
    assert token_f1("the cat sat on the mat", "a cat sat") == pytest.approx(2 * (2 / 4) * (2 / 2) / (2 / 4 + 1))
    assert token_f1("big big dog", "big dog") == pytest.approx(0.8)
    assert token_f1("", "") == 1.0 and token_f1("x", "") == 0.0 and token_f1("cat", "dog") == 0.0


def test_judge_prompt_and_parse():
    p = judge_prompt("Q?", "Ref.", "Ans.", "5: correct")
    assert p == JUDGE_TEMPLATE.format(question="Q?", reference="Ref.", answer="Ans.", rubric="5: correct")
    assert parse_judge("<reasoning> Good. </reasoning>\n<score>4</score>") == (4, "Good.")
    assert parse_judge("<score>2</score> on reflection <score> 5 </score>") == (5, "")
    assert parse_judge("<reasoning>hmm</reasoning><score>9</score>") == (None, "hmm")
    assert parse_judge("I'd give it a four.") == (None, "")


def test_llm_judge():
    fake = FakeClaude().add_text("<reasoning>Correct.</reasoning><score>5</score>")
    assert llm_judge(fake.client(), "What is 2+2?", "4", "Four", "5: correct") == 5
    body = fake.bodies[-1]
    assert body["model"] == JUDGE_MODEL and body["output_config"] == {"effort": "medium"}
    assert body["messages"][0]["content"] == judge_prompt("What is 2+2?", "4", "Four", "5: correct")
    fake.add_text("Looks fine to me!")
    assert llm_judge(fake.client(), "q", "r", "a", "rubric") is None


def first_wins_judge(body):
    return text_reply("<winner>1</winner>")


def fair_judge(body):
    prompt = body["messages"][0]["content"]
    first = re.search(r"<response_1>(.*?)</response_1>", prompt, re.S).group(1)
    second = re.search(r"<response_2>(.*?)</response_2>", prompt, re.S).group(1)
    return text_reply("<winner>1</winner>" if "correct" in first and "correct" not in second else "<winner>2</winner>")


def test_pairwise_judge():
    good, bad = "A correct, clear answer.", "A vague answer."
    fair = FakeClaude(responder=fair_judge)
    assert pairwise_judge(fair.client(), "Q", good, bad) == "A"
    assert pairwise_judge(fair.client(), "Q", bad, good) == "B"
    prompts = [b["messages"][0]["content"] for b in fair.bodies[-2:]]
    assert "<response_1>A vague answer.</response_1>" in prompts[0]
    assert "<response_1>A correct, clear answer.</response_1>" in prompts[1], "swap the order the second time"
    biased = FakeClaude(responder=first_wins_judge)
    assert pairwise_judge(biased.client(), "Q", good, bad) == "tie", "position bias must not produce a winner"
    broken = FakeClaude(responder=lambda b: text_reply("Both are fine."))
    assert pairwise_judge(broken.client(), "Q", good, bad) == "tie"


def test_bootstrap_ci():
    scores = np.random.default_rng(1).binomial(1, 0.8, size=100).astype(float)
    mean, low, high = bootstrap_ci(scores)
    assert mean == pytest.approx(scores.mean())
    assert low < mean < high and 0.1 < high - low < 0.25
    m2, l2, h2 = bootstrap_ci(np.tile(scores, 10))
    assert (h2 - l2) < 0.5 * (high - low), "10x the data -> a much narrower interval"
    assert bootstrap_ci(scores, seed=3) == bootstrap_ci(scores, seed=3)
    assert not contains(bootstrap_ci, LOOP_NODES)


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    difficulty = rng.normal(size=200)
    a = (difficulty + rng.normal(scale=0.3, size=200) > 0).astype(float)
    b = (difficulty + 0.3 + rng.normal(scale=0.3, size=200) > 0).astype(float)
    diff, low, high, significant = paired_bootstrap(a, b)
    assert diff == pytest.approx(b.mean() - a.mean()) and significant and low > 0
    same = paired_bootstrap(a, a.copy())
    assert same[0] == 0 and not same[3]
    noisy = paired_bootstrap(a[:20], b[:20][::-1])
    assert isinstance(noisy[3], bool)


def test_cohens_kappa():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert cohens_kappa(["p", "p", "f", "f"], ["p", "f", "p", "f"]) == pytest.approx(0.0)
    judge = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    human = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    assert cohens_kappa(judge, human) == pytest.approx((0.8 - 0.82) / (1 - 0.82)), "high raw agreement can be chance"
    assert cohens_kappa([1, 1], [1, 1]) == 1.0


def test_run_eval():
    dataset = [{"question": "Capital of France?", "answer": "Paris"},
               {"question": "Capital of Japan?", "answer": "Tokyo"},
               {"question": "Largest planet?", "answer": "Jupiter"}]
    answers = {"Capital of France?": "Paris.", "Capital of Japan?": "Kyoto", "Largest planet?": "the planet Jupiter"}
    rows, summary = run_eval(answers.get, dataset, {"em": exact_match, "f1": token_f1})
    assert [r["prediction"] for r in rows] == ["Paris.", "Kyoto", "the planet Jupiter"]
    assert rows[0] == {"question": "Capital of France?", "prediction": "Paris.", "em": 1.0, "f1": 1.0}
    assert summary["em"] == pytest.approx(1 / 3)
    assert summary["f1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
