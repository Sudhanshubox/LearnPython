# m35 · Project: a search engine package

**Phase 3 finale.** You'll build `searchkit`, a small but real full-text search engine, as a properly structured, typed and tested Python package with a command-line interface. It uses nearly everything from this phase: classes, dataclasses, type hints, packages, generators, files and JSON, testing and performance thinking.

**Why it matters for AI:** this is the **R in RAG** (retrieval-augmented generation). Before an LLM can answer questions about your documents, something has to find the most relevant passages. BM25, which you'll implement here, is still a strong baseline that modern retrieval systems are compared against, and is often combined with vector search in "hybrid" retrieval. In Phase 8 you'll plug a search engine like this one into an LLM.

```bash
$ python -m searchkit index docs --out index.json
Indexed 8 documents.

$ python -m searchkit search index.json "how do neural networks learn"
1. backprop (6.337)  Backpropagation Backpropagation computes the gradient of the loss with respect t
2. neural_networks (5.046)  Neural networks A neural network is a stack of layers. Each layer multiplies its
```

(Notice what it *doesn't* find: "networks" doesn't match "network", because there's no stemming yet. That's one of the stretch goals.)

---

## How search engines work

### 1. Tokenize

Turn text into terms: lowercase, split into words, drop **stopwords** ("the", "is", "of"…) that appear everywhere and carry little meaning.

### 2. Build an inverted index

Instead of scanning every document for every query (slow), build a map from each **term** to the documents containing it and how often:

```
"gradient" → {"backprop": 3, "gradient_descent": 5}
"layer"    → {"neural_networks": 4, "backprop": 1}
```

A query then only touches the documents that contain its terms. Google started with essentially this idea.

### 3. Rank: TF-IDF

A document is more relevant if it mentions a query term **often** (term frequency, tf), especially a term that's **rare** across the collection (inverse document frequency, idf):

```
idf(t) = ln(N / df(t))           N = number of documents, df = documents containing t
tfidf(q, d) = Σ over query terms t of  tf(t, d) × idf(t)
```

"the" appears in every document, so its idf is ln(1) = 0. "backpropagation" appears in one, so it's highly informative.

### 4. Rank better: BM25

TF-IDF has two weaknesses: a term mentioned 50 times isn't really 50× more relevant (**saturation**), and long documents mention everything more often (**length bias**). BM25 fixes both:

```
idf(t)    = ln(1 + (N − df + 0.5) / (df + 0.5))
bm25(q,d) = Σ idf(t) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × dl / avgdl))
```

- `dl` is the document's length in terms and `avgdl` the average length.
- `k1` (usually 1.2–2.0) controls how fast term frequency saturates.
- `b` (usually 0.75) controls how strongly long documents are penalized.

### 5. Evaluate

How do you know your ranking is good? Make a small set of queries with known relevant documents and measure:

- **precision@k**: the fraction of the top-k results that are relevant.
- **MRR** (mean reciprocal rank): the average over queries of 1/(rank of the first relevant result).

This is exactly how retrieval components of RAG systems are evaluated.

---

## The package

```
searchkit/
├── __init__.py     public API: SearchEngine, SearchResult, tokenize, __version__
├── text.py         tokenize() and STOPWORDS
├── index.py        InvertedIndex
├── ranking.py      tf_idf() and bm25()
├── engine.py       SearchEngine: add documents, search, save/load
└── __main__.py     the command-line interface
pyproject.toml
exercises.py        evaluation metrics: precision@k and MRR
docs/               a small sample collection of documents
```

Every file has stubs with docstrings and **type hints**; the docstrings are the specification. Work in this order: `text.py` → `index.py` → `ranking.py` → `engine.py` → `__main__.py` → `exercises.py` → `pyproject.toml`. Run the tests after each file: they're grouped by file, so you can see your progress.

The tests also run **mypy** on the package, so keep every function annotated.

## Stretch goals (optional)

- Stemming or lemmatization, so "networks" also matches "network" (try a simple suffix-stripping rule first, then compare with the Porter stemmer).
- Phrase queries (`"neural network"` in quotes must match adjacent words): store term **positions** in the index.
- Highlight the query terms in the snippet.
- Add documents incrementally without rebuilding, and remove documents.
- Measure: how long does indexing 10,000 documents take? Profile it (m34) and speed it up.
- Write your own tests for the CLI's edge cases and ask the mentor to review them.

## Go deeper (optional, research-level)

1. Read *The Probabilistic Relevance Framework: BM25 and Beyond* (Robertson & Zaragoza, 2009), section 3. Where does the BM25 formula come from?
2. Dense retrieval embeds queries and documents as vectors and searches by similarity (Phase 8). Read the abstract of *Dense Passage Retrieval for Open-Domain Question Answering* (Karpukhin et al., 2020). In which situations does BM25 still beat dense retrieval?
3. The BEIR benchmark (Thakur et al., 2021) compared many retrieval methods across domains. What did it find about BM25's robustness?
