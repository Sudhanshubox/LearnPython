"""m85 project: "Ask the course", a production RAG assistant. Read the README first."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import numpy as np

from retrievers import KeywordRetriever, VectorRetriever

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
PRICE = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}
HERE = Path(__file__).parent
CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())

SYSTEM = ("You are the teaching assistant for a Python and AI course. Answer learners' questions "
          "accurately and concisely, using only the course documents you are given.")
REFUSAL_MESSAGE = "I can't help with that question. Try asking about the course material in a different way."


# ---- given helpers (from m80) ----
def chunk_markdown(text, source, max_chars=1000):
    chunks, heading, buffer = [], "", []

    def flush():
        if buffer:
            chunks.append({"source": source, "heading": heading, "text": "\n\n".join(buffer)})
            buffer.clear()

    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if para.startswith("#") and "\n" not in para:
            flush()
            heading = para.lstrip("#").strip()
            continue
        if buffer and len("\n\n".join(buffer + [para])) > max_chars:
            flush()
        buffer.append(para)
    flush()
    return chunks


def load_course_chunks(pattern="phase0[1-7]*/*/README.md"):
    chunks = []
    for path in sorted(CURRICULUM.glob(pattern)):
        chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.parent.name.split("_")[0])
    return chunks


def load_eval_set():
    return json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))


def escape_xml(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---- exercises ----


# 1. Hybrid retrieval over the chunks: build a KeywordRetriever and a VectorRetriever on texts
#    {"text": "HEADING\nTEXT"} (just TEXT when there's no heading). search(query, k): fuse the
#    top `depth` results of both with reciprocal rank fusion (score += 1 / (k_rrf + rank),
#    ranks from 1; ties: smaller index first) and return the best k chunk indices.
class HybridRetriever:
    def __init__(self, chunks, depth=20, k_rrf=60):
        raise NotImplementedError

    def search(self, query, k=5):
        raise NotImplementedError


# 2. The grounded prompt: like m80's build_rag_prompt (documents with index, "SOURCE: HEADING"
#    sources and escaped content; instructions to answer only from the documents, cite them as
#    [1] or [2][3], and say "I don't know based on the course material." when the answer isn't
#    there; the escaped question last in <question> tags).
def build_prompt(question, chunks):
    raise NotImplementedError


# 3. Dollars for one call from its usage (PRICE is per million tokens). cache_read_input_tokens
#    and cache_creation_input_tokens may be None.
def message_cost(usage):
    raise NotImplementedError


@dataclass
class Answer:
    text: str
    sources: list         # cited lesson ids, in citation order, without duplicates
    cost_usd: float
    cached: bool = False
    refused: bool = False


class AssistantUnavailable(Exception):
    pass


def normalize_question(question):
    """Given: the cache key for a question (lowercase, single spaces)."""
    return " ".join(question.lower().split())


# 4. The assistant.
#    __init__: keep the arguments; retriever defaults to HybridRetriever(chunks);
#      self.cache = {} (normalized question -> Answer); self.total_cost = 0.0.
#    retrieve(question): the chunks for retriever.search(question, k), kept in order while their
#      total "text" length stays <= max_context_chars (stop at the first that doesn't fit).
#    ask(question):
#      - a cached question returns Answer(same text, same sources, 0.0, cached=True, same refused);
#      - otherwise client.beta.messages.create(model=MODEL, max_tokens=2048, system=SYSTEM,
#        messages=[build_prompt(question, retrieved chunks) as a user message],
#        output_config={"effort": "low"}, betas=[FALLBACK_BETA], fallbacks="default");
#        an anthropic.APIError becomes AssistantUnavailable (raised from it);
#      - add message_cost(usage) to self.total_cost;
#      - stop_reason "refusal" -> Answer(REFUSAL_MESSAGE, [], cost, refused=True) (not cached);
#      - else the reply text, with sources = the lesson ids of the valid cited documents ([n]
#        with 1 <= n <= number of chunks), in order of first citation, without duplicates.
#        Cache it and return it.
class CourseAssistant:
    def __init__(self, client, chunks, retriever=None, k=5, max_context_chars=6000):
        raise NotImplementedError

    def retrieve(self, question):
        raise NotImplementedError

    def ask(self, question):
        raise NotImplementedError


# 5. Grade one answer: client.messages.create(model=MODEL, max_tokens=1024,
#    output_config={"effort": "medium"}, messages=[JUDGE_TEMPLATE filled in, as a user message]).
#    Return the integer in the last <score> tag if it's 1-5, else None.
JUDGE_TEMPLATE = """Grade this answer from a course teaching assistant.

<question>{question}</question>
<reference_answer>{reference}</reference_answer>
<answer_to_grade>{answer}</answer_to_grade>

Give a score from 1 (wrong or unhelpful) to 5 (correct, complete and concise) inside <score></score> tags."""


def judge(client, question, reference, answer):
    raise NotImplementedError


def bootstrap_ci(values, n_boot=2000, seed=0):
    """Given: (mean, 2.5% quantile, 97.5% quantile) of bootstrap resample means (m82)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# 6. Evaluate on items {"question", "expected_module", "reference"}. For each: whether the
#    expected module is among the sources of assistant.retrieve(question); then
#    answer = assistant.ask(question); whether the expected module is in answer.sources;
#    whether it was refused; and judge(judge_client, question, reference, answer.text).
#    Return {"n", "retrieval_hit_rate", "citation_hit_rate",
#            "judge_score": bootstrap_ci(valid scores) (or (None, None, None) if none),
#            "judge_failures": number of None scores, "refusal_rate",
#            "cost_per_question": the increase in assistant.total_cost divided by n,
#            "failures": [{"question", "answer", "score"} for valid scores <= 2]}.
def evaluate(assistant, judge_client, eval_set):
    raise NotImplementedError


# 7. A FastAPI app: GET /health -> {"status": "ok"}; POST /ask with a JSON body
#    {"question": 1-2000 characters} and an X-API-Key header that must be in api_keys (else 401)
#    -> {"answer", "sources", "cached"}; AssistantUnavailable -> 503.
def create_app(assistant, api_keys):
    raise NotImplementedError


# 8. Write the report (README outline) to `path`, with the four section headings, a Markdown
#    table including the retrieval and citation hit rates and the judge score formatted with
#    2 decimals (with its CI), the cost per question, and a line per failure.
def write_report(results, path):
    raise NotImplementedError
