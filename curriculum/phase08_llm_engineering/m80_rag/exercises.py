"""m80 exercises: a RAG pipeline over this course's lessons.

retrievers.py (given) has KeywordRetriever and VectorRetriever.
"""

import re
from pathlib import Path

from retrievers import KeywordRetriever, VectorRetriever

MODEL = "claude-opus-5-5"
CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())


# 1. Structure-aware chunking (README section 2). Split the text into paragraphs on blank
#    lines (strip them; skip empty ones). A paragraph that starts with "#" and is a single
#    line is a heading: it ends the current chunk and becomes the heading for what follows
#    (store it without the leading #'s and spaces). Other paragraphs are added to the current
#    chunk, but if adding one would make "\n\n".join(paragraphs) longer than max_chars, the
#    current chunk ends first (a single long paragraph still becomes its own chunk).
#    Return a list of {"source": source, "heading": heading ("" before any heading),
#    "text": the chunk's paragraphs joined with "\n\n"}.
def chunk_markdown(text, source, max_chars=1000):
    raise NotImplementedError


def load_course_chunks(pattern="phase0[1-7]*/*/README.md", max_chars=1000):
    """Given: chunk every lesson; the source is the module id, e.g. "m08"."""
    chunks = []
    for path in sorted(CURRICULUM.glob(pattern)):
        chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.parent.name.split("_")[0], max_chars)
    return chunks


# 2. The text to index for a chunk: "HEADING\nTEXT" if it has a heading, else just the text.
def chunk_text_for_index(chunk):
    raise NotImplementedError


# 3. Reciprocal rank fusion (README section 3). rankings: lists of items, best first.
#    Score each item by the sum of 1 / (k + rank) over the rankings it appears in (rank
#    starts at 1). Return the items sorted by score, highest first (ties: smaller item first),
#    cut to top_n if given.
def reciprocal_rank_fusion(rankings, k=60, top_n=None):
    raise NotImplementedError


# 4. A hybrid retriever: search(query, k) gets the top `depth` results from every retriever
#    and fuses them with reciprocal_rank_fusion(..., k_rrf, top_n=k).
class HybridRetriever:
    def __init__(self, retrievers, depth=20, k_rrf=60):
        raise NotImplementedError

    def search(self, query, k=5):
        raise NotImplementedError


def escape_xml(text):
    """Given: escape &, < and > for use inside XML-style tags."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# 5. The grounded prompt (README section 4): a <documents> block with one
#    <document index="i"> per chunk (i from 1), each with <source>SOURCE: HEADING</source>
#    (just SOURCE if there's no heading) and <document_content>TEXT</document_content>,
#    everything escaped; then instructions to answer only from the documents, to cite them as
#    [1], [2][3]..., and to say "I don't know based on the course material." if the answer
#    isn't there; then <question>...</question> LAST.
def build_rag_prompt(question, chunks):
    raise NotImplementedError


# 6. The cited document numbers in an answer, in order of first appearance, without
#    duplicates, keeping only numbers from 1 to n_docs.
def extract_citations(answer, n_docs):
    raise NotImplementedError


# 7. Keep chunks in order while the total length of their "text" stays <= max_chars;
#    stop at the first chunk that doesn't fit.
def fit_to_budget(chunks, max_chars):
    raise NotImplementedError


# 8. The full pipeline: retrieve k chunk indices with retriever.search(question, k), take
#    those chunks, fit_to_budget(..., max_context_chars), call client.messages.create(
#    model=MODEL, max_tokens=2048, a system prompt of your choice, one user message with
#    build_rag_prompt, output_config={"effort": "low"}). Return {"answer": reply text,
#    "sources": the sources of the cited chunks in citation order, "chunks": the chunks used}.
def answer_question(client, question, chunks, retriever, k=5, max_context_chars=6000):
    raise NotImplementedError
