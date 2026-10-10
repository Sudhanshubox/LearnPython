import math

import pytest

from exercises import (
    PaperNote,
    bibtex_key,
    improvement_vs_noise,
    note_markdown,
    parse_arxiv_id,
    priority,
    reading_list,
    relevance,
    table_inconsistencies,
    to_bibtex,
    validate_note,
)

ATTENTION = PaperNote(title="Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"], year=2017,
                      arxiv_id="1706.03762")


def test_parse_arxiv_id():
    assert parse_arxiv_id("https://arxiv.org/abs/1706.03762v7") == ("1706.03762", 7)
    assert parse_arxiv_id("see arXiv:2307.03381 for details") == ("2307.03381", None)
    assert parse_arxiv_id("1512.03385") == ("1512.03385", None)
    assert parse_arxiv_id("https://arxiv.org/pdf/2106.09685v2.pdf") == ("2106.09685", 2)
    assert parse_arxiv_id("no id here, version 3.14") is None
    assert parse_arxiv_id("12.34567.8") is None


def test_validate_note():
    assert validate_note(ATTENTION) == []
    assert validate_note(PaperNote(title=" ", authors=[], year=2020, passes=0)) == \
        ["missing title", "missing authors", "passes must be 1, 2 or 3"]
    second = PaperNote(title="T", authors=["A"], year=2020, passes=2, problem="p",
                       claims=["c1", "c2"], evidence=["e1"])
    assert validate_note(second) == ["missing method", "every claim needs evidence"]
    third = PaperNote(title="T", authors=["A"], year=2020, passes=3, problem="p", method="m",
                      claims=["c"], evidence=["e"], questions=["q"])
    assert validate_note(third) == ["missing limitations"]


def test_bibtex():
    assert bibtex_key(ATTENTION) == "vaswani2017attention"
    lora = PaperNote(title="LoRA: Low-Rank Adaptation of Large Language Models", authors=["Edward J. Hu"], year=2021)
    assert bibtex_key(lora) == "hu2021lora"
    on = PaperNote(title="On the Difficulty of Training RNNs", authors=["Razvan Pascanu"], year=2013)
    assert bibtex_key(on) == "pascanu2013difficulty"
    assert to_bibtex(ATTENTION) == (
        "@article{vaswani2017attention,\n"
        "  title = {{Attention Is All You Need}},\n"
        "  author = {Ashish Vaswani and Noam Shazeer},\n"
        "  year = {2017},\n"
        "  eprint = {1706.03762},\n"
        "  archivePrefix = {arXiv}\n"
        "}")
    assert "eprint" not in to_bibtex(lora)


def test_note_markdown_is_given():
    md = note_markdown(PaperNote(title="T", authors=["A"], year=2020, claims=["c"], evidence=["Fig 2"]))
    assert "- c (evidence: Fig 2)" in md


PAPERS = [
    {"title": "Deep Residual Learning for Image Recognition", "year": 2015, "citations": 200000,
     "abstract": "residual networks ease the training of deep networks"},
    {"title": "Teaching Arithmetic to Small Transformers", "year": 2023, "citations": 150,
     "abstract": "small transformers learn addition; reversed digits and data format matter"},
    {"title": "Attention Is All You Need", "year": 2017, "citations": 150000,
     "abstract": "the transformer architecture based solely on attention"},
    {"title": "A Survey of Gardening", "year": 2024, "citations": 3, "abstract": "plants and soil"},
]


def test_relevance_and_priority():
    assert relevance(PAPERS[1], ["Transformers", "addition", "digits"]) == pytest.approx(1.0)
    assert relevance(PAPERS[0], ["transformers", "addition"]) == 0.0
    assert relevance(PAPERS[0], []) == 0.0
    p = priority(PAPERS[2], ["attention"], 2026)
    assert p == pytest.approx(3 * 1.0 + math.log10(150001) / math.sqrt(10))


def test_reading_list():
    top = reading_list(PAPERS, ["transformers", "addition", "digits"], 2026, top_n=2)
    assert [p["title"] for p in top] == ["Teaching Arithmetic to Small Transformers", "Attention Is All You Need"]
    assert len(reading_list(PAPERS, [], 2026)) == 4


def test_table_inconsistencies():
    rows = [{"name": "Ours", "scores": [80.1, 75.3, 90.2], "reported_average": 81.9},
            {"name": "Baseline", "scores": [78.0, 74.0, 88.0], "reported_average": 80.0}]
    assert table_inconsistencies(rows) == [], "81.867 vs 81.9 is within rounding tolerance"
    rows[1]["reported_average"] = 81.0
    assert table_inconsistencies(rows) == [("Baseline", 80.0, 81.0)]


def test_improvement_vs_noise():
    result = improvement_vs_noise([70.1, 72.3, 69.5], [71.0, 73.2, 70.4])
    assert result["difference"] == pytest.approx(0.9)
    assert result["seed_std"] == pytest.approx(1.4742, abs=1e-3)
    assert result["convincing"] is False
    assert improvement_vs_noise([70.0, 70.5, 69.8], [75.1, 75.4, 74.9])["convincing"] is True
