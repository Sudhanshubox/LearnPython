# m79 · Embeddings and vector search

**By the end you can:** explain what text embeddings are and why cosine similarity finds related meaning, build an embedding model from scratch (TF-IDF + SVD, "latent semantic analysis"), implement exact nearest-neighbour search with NumPy, build an approximate **IVF** index and measure its recall/speed trade-off, and evaluate retrieval quality on real queries.

**Why it matters for AI:** retrieval is how LLM applications use knowledge they weren't trained on: your documents, your codebase, today's news. Vector search powers RAG (m80), semantic search, recommendations and deduplication. Knowing how the index works (and that it's approximate) is what lets you debug "the bot couldn't find the answer that's obviously in our docs".

---

## 1. Embeddings

An **embedding** maps a piece of text to a vector of a few hundred numbers, such that texts with similar meaning get vectors pointing in similar directions. You met the idea in m67 (character embeddings) and m69 (token embeddings); text-embedding models produce one vector per sentence or paragraph.

Similarity is measured with **cosine similarity** (m38): the dot product of L2-normalized vectors, from −1 to 1. Normalize once when you store vectors, and similarity becomes a single matrix multiply.

Where do embeddings come from in practice? From dedicated embedding models: hosted APIs (Anthropic recommends Voyage AI; OpenAI, Cohere and Google offer them too) or open models you run yourself (the `sentence-transformers` library). The Claude API itself generates text, not embeddings. Everything in this module works the same with any of them: only the `encode` function changes.

## 2. An embedding model you can build: LSA

Before neural embeddings, **latent semantic analysis** (Deerwester et al., 1990) did the job with linear algebra you already know:

1. Build the TF-IDF matrix (m35): rows = documents, columns = words, entries = how distinctive each word is in each document. `sublinear_tf` (1 + log tf) and removing English stop words help.
2. Take a truncated **SVD** (m39) to compress the ~5,000 word dimensions to ~128 "topic" dimensions.
3. Normalize each row.

Words that appear in similar contexts load on the same components, so a query about "catching errors" can match a paragraph about "exceptions" even with few shared words. Neural embeddings do this far better, but LSA is fast, free, offline, and a good baseline.

**Fit once, encode many:** the vectorizer and SVD are fitted on the corpus; queries are encoded with the *same* fitted transforms (as with `StandardScaler`, m53).

## 3. Exact search

With N normalized vectors in a matrix M, the scores for a query q are `M @ q`. Taking the top k doesn't need a full sort: `np.argpartition` finds the k largest in O(N), then sort just those k. This "flat" index is exact and, thanks to NumPy, fast up to around a million vectors.

## 4. Approximate search: IVF

Beyond that, you need **approximate nearest neighbour (ANN)** search. The **inverted file index (IVF)**, used by FAISS:

1. **Train:** cluster the vectors with k-means (m52) into `n_lists` clusters.
2. **Add:** put each vector in the list of its nearest centroid.
3. **Search:** compare the query with the centroids, and only scan the vectors in the `n_probe` closest lists.

With 16 lists and `n_probe = 1`, each query scans about 1/16 of the data, but it misses true neighbours that sit in other clusters. Raising `n_probe` trades speed for **recall@k** (the fraction of the true top-k found). With `n_probe = n_lists` it's exact again. Other ANN families: HNSW graphs (the most popular today), product quantization (compressing vectors), and LSH.

## 5. Evaluating retrieval

Never assume retrieval works: measure it. Write down real queries with the answer you expect (here: which lesson a question belongs to) and compute the **hit rate@k**: how often the expected source appears in the top k results. You'll find something useful: plain TF-IDF keyword search is a strong baseline, often as good as the dense LSA embeddings on these queries. Each fails on different queries. That's why production systems use **hybrid search** (m80): combine keyword and vector results.

---

## Problem-solving habit #79: measure the approximation

Every approximation (ANN search, quantization in m71, sampling, caching) trades accuracy for speed or cost. Make the trade-off explicit: measure both sides (recall *and* scanned vectors or latency) across settings, and choose a point on the curve deliberately.

## Common mistakes

- Forgetting to normalize, so long documents score higher just because their vectors are longer.
- Fitting the vectorizer on the queries, or re-fitting it at query time.
- Embedding whole 50-page documents as one vector (chunk them first, m80).
- Mixing vectors from two different embedding models in one index.
- Never measuring recall, then blaming the LLM for retrieval misses.

## Go deeper (optional)

1. Install `sentence-transformers` and replace `LSAEmbedder` with `all-MiniLM-L6-v2`. How much does the hit rate improve on the test queries?
2. Read the FAISS paper (*The Faiss library*, Douze et al., 2024) or the HNSW paper (Malkov & Yashunin, 2018). Why do graph indexes beat IVF at high recall?
3. Plot recall@10 against the number of scanned vectors for n_probe = 1, 2, 4, 8, 16 (m48). Where is the "knee" of the curve?

## Your turn

Open the **Exercises** tab. The corpus is the paragraphs of this course's own lessons (Phases 1–7), so searches return your own textbook.
