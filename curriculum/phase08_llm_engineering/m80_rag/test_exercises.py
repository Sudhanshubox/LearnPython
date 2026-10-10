import re

import pytest

from exercises import (
    MODEL,
    HybridRetriever,
    answer_question,
    build_rag_prompt,
    chunk_markdown,
    chunk_text_for_index,
    extract_citations,
    fit_to_budget,
    load_course_chunks,
    reciprocal_rank_fusion,
)
from fakeclaude import FakeClaude, text_reply
from retrievers import KeywordRetriever, VectorRetriever

QUERIES = [
    ("How do I catch exceptions with try and except?", "m08"), ("What is a dictionary and how do I look up a key?", "m06"),
    ("binary search on a sorted list", "m16"), ("What does the KV cache store during generation?", "m71"),
    ("git commit and branches", "m32"), ("perplexity of a language model", "m67"), ("LoRA low rank adapters", "m73"),
    ("how does backpropagation compute gradients for a matrix layer", "m57"),
    ("reading a CSV file with pandas", "m46"), ("overfitting and regularization weight decay dropout", "m60"),
    ("what is a closure in python", "m07"), ("how does a heap priority queue work", "m19"),
    ("dynamic programming memoization", "m21"), ("why divide attention scores by square root of d", "m68"),
    ("BPE merges tokenizer", "m66"), ("gradient boosting residuals", "m54"), ("train test split data leakage", "m53"),
    ("generators and yield", "m28"), ("dataclass type hints", "m27"), ("async await event loop", "m33"),
]

DOC = """# Title line

Intro paragraph.

## Section A

First A paragraph.

Second A paragraph.

## Section B
"""


def test_chunk_markdown():
    chunks = chunk_markdown(DOC, "m99", max_chars=1000)
    assert chunks == [
        {"source": "m99", "heading": "Title line", "text": "Intro paragraph."},
        {"source": "m99", "heading": "Section A", "text": "First A paragraph.\n\nSecond A paragraph."},
    ]
    small = chunk_markdown(DOC, "m99", max_chars=25)
    assert [c["text"] for c in small] == ["Intro paragraph.", "First A paragraph.", "Second A paragraph."]
    assert small[2]["heading"] == "Section A"
    long = chunk_markdown("x" * 50, "m1", max_chars=10)
    assert long == [{"source": "m1", "heading": "", "text": "x" * 50}]
    multi = chunk_markdown("```\n# a comment in code\nprint(1)\n```", "m1")
    assert multi[0]["heading"] == "" and "# a comment" in multi[0]["text"], "only single-line # paragraphs are headings"


def test_chunk_text_for_index():
    assert chunk_text_for_index({"source": "m1", "heading": "H", "text": "T"}) == "H\nT"
    assert chunk_text_for_index({"source": "m1", "heading": "", "text": "T"}) == "T"


def test_course_chunks():
    chunks = load_course_chunks()
    assert len(chunks) > 600
    assert max(len(c["text"]) for c in chunks if "\n\n" in c["text"]) <= 1000
    assert {c["source"] for c in chunks} >= {"m01", "m42", "m75"}


def test_rrf():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    assert fused[:2] == ["b", "c"]
    assert set(fused) == {"a", "b", "c", "d"}
    assert reciprocal_rank_fusion([["x"], ["y"]], top_n=1) == ["x"], "ties: smaller item first"
    scores = {"a": 1 / 61, "b": 1 / 62 + 1 / 61}
    assert reciprocal_rank_fusion([["a", "b"], ["b"]]) == sorted(scores, key=lambda i: -scores[i])


@pytest.fixture(scope="module")
def course():
    chunks = load_course_chunks()
    texts = [{"text": chunk_text_for_index(c)} for c in chunks]
    return chunks, KeywordRetriever(texts), VectorRetriever(texts)


def hit(chunks, retriever, k=5):
    return sum(m in [chunks[i]["source"] for i in retriever.search(q, k)] for q, m in QUERIES) / len(QUERIES)


@pytest.mark.timeout(60)
def test_hybrid_beats_each_retriever(course):
    chunks, kw, vec = course
    hybrid = HybridRetriever([kw, vec])
    results = hybrid.search("What does the KV cache store?", k=5)
    assert len(results) == 5 and len(set(results)) == 5
    h_kw, h_vec, h_hyb = hit(chunks, kw), hit(chunks, vec), hit(chunks, hybrid)
    assert h_hyb >= max(h_kw, h_vec), (h_kw, h_vec, h_hyb)
    assert h_hyb >= 0.9


def test_build_rag_prompt():
    chunks = [{"source": "m71", "heading": "The KV cache", "text": "Stores keys & values."},
              {"source": "m68", "heading": "", "text": "Attention <scores>."}]
    p = build_rag_prompt("What's in the <cache>?", chunks)
    assert p.startswith("<documents>")
    assert '<document index="1">' in p and '<document index="2">' in p
    assert "<source>m71: The KV cache</source>" in p and "<source>m68</source>" in p
    assert "Stores keys &amp; values." in p and "Attention &lt;scores&gt;." in p
    assert "[1]" in p and "I don't know based on the course material." in p
    assert p.rstrip().endswith("<question>What's in the &lt;cache&gt;?</question>")


def test_extract_citations():
    assert extract_citations("A [2] and B [1][2], see [7] and [0].", 5) == [2, 1]
    assert extract_citations("No citations.", 3) == []


def test_fit_to_budget():
    chunks = [{"text": "a" * 40}, {"text": "b" * 40}, {"text": "c" * 10}]
    assert fit_to_budget(chunks, 85) == chunks[:2]
    assert fit_to_budget(chunks, 30) == []


def test_answer_question(course):
    chunks, kw, vec = course
    fake = FakeClaude()
    fake.add(text_reply("The KV cache stores keys and values [1][3], see also [9]."))
    result = answer_question(fake.client(), "What does the KV cache store?", chunks, HybridRetriever([kw, vec]), k=4)
    body = fake.bodies[-1]
    assert body["model"] == MODEL and body["output_config"] == {"effort": "low"} and "system" in body
    prompt = body["messages"][0]["content"]
    assert prompt.count("<document index=") == len(result["chunks"]) <= 4
    assert result["answer"].startswith("The KV cache stores")
    assert result["sources"] == [result["chunks"][0]["source"], result["chunks"][2]["source"]]
    assert "m71" in [c["source"] for c in result["chunks"]]
    fake.add_text("I don't know based on the course material.")
    small = answer_question(fake.client(), "KV cache", chunks, kw, k=5, max_context_chars=1)
    assert small["chunks"] == [] and small["sources"] == []
