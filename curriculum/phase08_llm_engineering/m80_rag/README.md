# m80 · Retrieval-augmented generation (RAG)

**By the end you can:** build a complete RAG pipeline (structure-aware chunking, keyword and vector retrieval, hybrid search with reciprocal rank fusion, a context budget, a grounded prompt with numbered citations, and citation checking) and measure which retriever actually finds the answers.

**Why it matters for AI:** RAG is the most common LLM application architecture in industry: support bots over help centres, assistants over company wikis, legal and medical search, coding assistants over your repo. It lets a model answer from documents it was never trained on, cite them, and stay current without retraining. Most RAG failures are retrieval failures, which is why this module measures retrieval separately.

---

## 1. The pipeline

```
indexing (once):    documents ─▶ chunks ─▶ keyword index + vector index
answering (per question):
    question ─▶ retrieve top chunks (hybrid) ─▶ fit to a context budget
             ─▶ prompt: <documents> + instructions + question ─▶ Claude
             ─▶ answer with [1][2] citations ─▶ check citations ─▶ show answer + sources
```

## 2. Chunking

Retrieval works on **chunks**, not whole documents. Too big, and a chunk mixes many topics and wastes context; too small, and it loses the context needed to understand it. Practical rules:

- **Respect structure.** Split at paragraph boundaries, never mid-sentence. Keep paragraphs together until a size limit (here 1,000 characters).
- **Start a new chunk at each heading**, and remember the heading: a paragraph that says "this makes it 4× smaller" is only meaningful with its heading "int8 quantization". Index the heading together with the text ("contextual chunking"; Anthropic's *Contextual Retrieval* article takes this further by having a model write a short context line for each chunk).
- Keep metadata (source file, heading) for citations and filtering.

## 3. Retrieval: keyword, vector and hybrid

`retrievers.py` gives you two retrievers built in m79's style: **keyword** (TF-IDF) and **vector** (LSA embeddings). They fail on different questions: keyword search misses paraphrases; vector search blurs exact terms like `KV cache` or `LoRA`.

**Reciprocal rank fusion** (RRF; Cormack et al., 2009) combines rankings without needing comparable scores:

RRF(item) = Σ over rankings of 1 / (k + rank of item in that ranking),   with k = 60

Items ranked well by *several* retrievers rise to the top. It's simple, robust, and the standard way to do hybrid search. On 20 questions about this course, hit rate@5 is about 0.90 for keyword, 0.85 for vector, and 0.95 for hybrid.

Production systems often add a **reranker** (a model that scores each question–chunk pair more accurately) on the top 20–50 candidates.

## 4. The grounded prompt

Use m77's structure: documents first, each with an index and source; then instructions; the question last:

- Answer **only** from the documents.
- **Cite** documents by index: `[1]`, `[2][3]`.
- If the documents don't contain the answer, **say so** ("I don't know based on the course material"). This one instruction prevents a large share of hallucinations.

Keep a **context budget**: more chunks aren't always better (cost, latency, and models can get distracted by irrelevant text). Add chunks in rank order until the budget is reached.

## 5. Checking the answer

Parse the citation markers, ignore any that point to documents that don't exist (`[7]` when there were 5), de-duplicate, and map them to sources to show the user. An answer with no valid citations is a warning sign worth logging. m82 shows how to evaluate faithfulness (whether the answer is actually supported by the cited text).

## 6. When RAG isn't the answer

RAG retrieves *facts*. It doesn't teach new *skills* or styles (that's fine-tuning, m73), and with today's 1M-token context windows, a small enough corpus can simply be put in the prompt in full, with prompt caching (m83) making repeated questions cheap. Choose by corpus size, update frequency and cost.

---

## Problem-solving habit #80: debug the pipeline stage by stage

When a RAG answer is wrong, find which stage failed before changing anything: was the right chunk retrieved at all (check retrieval)? Was it cut by the budget? Was it in the prompt but ignored or misread (check generation)? Each stage has its own fix: chunking, retrieval, budget, or prompt. Logging the retrieved chunks with every answer makes this a two-minute job.

## Common mistakes

- Fixed-size character chunks that cut sentences and separate text from its heading.
- Evaluating only the final answers, so retrieval problems look like model problems.
- Stuffing 30 chunks into the prompt "to be safe".
- No "I don't know" instruction.
- Trusting citation numbers without checking they exist.

## Go deeper (optional)

1. Read *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020) and Anthropic's *Introducing Contextual Retrieval* (2024).
2. Add contextual chunk headers generated by Claude (one call per chunk, with prompt caching on the document) and measure the change in hit rate.
3. Implement a simple reranker: send the question and the top 20 chunks to Claude and ask for the 5 most relevant indices as structured output (m78). Does hit rate@5 improve?

## Your turn

Open the **Exercises** tab. The corpus is this course's lessons, chunked by section. With a real API key, `answer_question(anthropic.Anthropic(), ...)` gives you an "ask the textbook" assistant.
