import random

import pytest
import torch
import torch.nn.functional as F

from exercises import (
    STOI,
    VOCAB,
    accuracy_by_carries,
    count_carries,
    curve_area,
    encode_batch,
    evaluate,
    format_example,
    make_pairs,
    predict,
    train_adder,
    write_report,
)
from minigpt import GPT, GPTConfig


def test_format_example():
    assert format_example(7, 120, 3, False) == "007+120=0127\n"
    assert format_example(7, 120, 3, True) == "007+120=7210\n"
    assert format_example(999, 999, 3, False) == "999+999=1998\n"
    assert format_example(5, 6, 1, True) == "5+6=11\n"


def test_make_pairs():
    pairs = make_pairs(5, 3, seed=4)
    rng = random.Random(4)
    assert pairs == [(rng.randrange(1000), rng.randrange(1000)) for _ in range(5)]


def test_encode_batch():
    x, y = encode_batch([(7, 120), (999, 1)], 3, reverse=False)
    assert x.dtype == torch.int64 and x.shape == (2, 12) and y.shape == (2, 12)
    ids = [STOI[c] for c in "007+120=0127\n"]
    assert x[0].tolist() == ids[:-1]
    assert y[0].tolist() == [-100] * 7 + ids[8:]


class Oracle(torch.nn.Module):
    """A fake 'model' that always predicts the right next character."""

    def __init__(self, reverse, wrong_last=False):
        super().__init__()
        self.reverse, self.wrong_last = reverse, wrong_last
        self.config = GPTConfig(vocab_size=len(VOCAB), block_size=16)

    def forward(self, idx, targets=None):
        logits = torch.zeros(idx.shape[0], idx.shape[1], len(VOCAB))
        for i, row in enumerate(idx.tolist()):
            text = "".join(VOCAB[t] for t in row)
            a, b = int(text[0:3]), int(text[4:7])
            full = format_example(a, b, 3, self.reverse)
            nxt = full[len(text)] if len(text) < len(full) else "\n"
            if self.wrong_last and len(text) == 11:
                nxt = str((int(nxt) + 1) % 10)
            logits[i, -1, STOI[nxt]] = 1.0
        return logits, None


def test_predict_and_evaluate():
    pairs = [(7, 120), (999, 999), (500, 500)]
    for reverse in (False, True):
        oracle = Oracle(reverse)
        oracle.train()
        assert predict(oracle, pairs, 3, reverse) == [127, 1998, 1000]
        assert oracle.training, "restore the previous mode"
        assert evaluate(oracle, pairs, 3, reverse) == (1.0, 1.0)
    wrong = Oracle(False, wrong_last=True)
    exact, digit = evaluate(wrong, pairs, 3, False)
    assert exact == 0.0 and digit == pytest.approx(0.75)


def test_count_carries():
    assert count_carries(99, 1) == 2
    assert count_carries(128, 367) == 1
    assert count_carries(999, 1) == 3
    assert count_carries(0, 0) == 0
    assert count_carries(555, 555) == 3


def test_curve_area():
    assert curve_area([(100, 0.0, 0.5), (200, 0.5, 0.8), (300, 1.0, 1.0)]) == pytest.approx(0.5)


def test_accuracy_by_carries():
    pairs = [(1, 2), (9, 1), (99, 1), (10, 20)]
    acc = accuracy_by_carries(Oracle(True), pairs, 3, True)
    assert acc == {0: 1.0, 1: 1.0, 2: 1.0}
    assert list(acc) == sorted(acc)


@pytest.fixture(scope="module")
def experiment():
    train, test = make_pairs(2000, 3, 0), make_pairs(300, 3, 1)
    results = {}
    for name, reverse in (("plain", False), ("reversed", True)):
        model, curve = train_adder(train, test, 3, reverse, steps=1200, eval_every=300)
        results[name] = {"curve": curve, "by_carries": accuracy_by_carries(model, test, 3, reverse)}
    return results


@pytest.mark.timeout(120)
def test_reversal_helps(experiment):
    plain, rev = experiment["plain"]["curve"], experiment["reversed"]["curve"]
    assert [s for s, _, _ in rev] == [300, 600, 900, 1200]
    assert rev[-1][1] > 0.85, "reversed answers should be learned well within 1200 steps"
    assert curve_area(rev) > curve_area(plain), "reversed answers are learned faster"
    assert rev[-1][1] > plain[-1][1]
    by = experiment["plain"]["by_carries"]
    assert by[0] >= by[max(by)], "carries make the plain format harder"


def test_report(tmp_path, experiment):
    path = tmp_path / "report.md"
    write_report(experiment, path)
    text = path.read_text(encoding="utf-8")
    for heading in ["# Teaching addition to a small transformer", "## Question", "## Setup", "## Results",
                    "## Analysis", "## Limitations"]:
        assert heading in text
    for name in ("plain", "reversed"):
        assert f"{experiment[name]['curve'][-1][1]:.3f}" in text
    assert "carr" in text.lower()
    assert len(text.split()) > 120, "write real sentences about your findings"
