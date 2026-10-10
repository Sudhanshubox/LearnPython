import numpy as np
import pytest

from code_checks import LOOP_NODES, contains
from exercises import (
    FlatIndex,
    IVFIndex,
    LSAEmbedder,
    hit_rate,
    load_chunks,
    normalize,
    recall_at_k,
    split_paragraphs,
    top_k,
)

QUERIES = [
    ("How do I catch exceptions with try and except?", "m08"),
    ("What is a dictionary and how do I look up a key?", "m06"),
    ("binary search on a sorted list", "m16"),
    ("What does the KV cache store during generation?", "m71"),
    ("git commit and branches", "m32"),
    ("perplexity of a language model", "m67"),
    ("LoRA low rank adapters", "m73"),
    ("how does backpropagation compute gradients for a matrix layer", "m57"),
    ("reading a CSV file with pandas", "m46"),
    ("overfitting and regularization weight decay dropout", "m60"),
]


@pytest.fixture(scope="module")
def corpus():
    chunks = load_chunks()
    embedder = LSAEmbedder(128).fit([text for _, text in chunks])
    return chunks, embedder, embedder.encode([text for _, text in chunks])


def test_split_paragraphs():
    text = "short\n\n" + "a" * 90 + "\n  \n" + "b" * 85 + "\nsame paragraph\n\n\n" + "c" * 10
    assert split_paragraphs(text) == ["a" * 90, "b" * 85 + "\nsame paragraph"]
    assert split_paragraphs("  hello  ", min_chars=1) == ["hello"]


def test_normalize():
    M = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(M, [[0.6, 0.8], [0.0, 0.0]])
    assert np.allclose(normalize(np.array([0.0, 2.0])), [0.0, 1.0])


def test_lsa_embedder(corpus):
    chunks, embedder, E = corpus
    assert len(chunks) > 1000
    assert E.shape == (len(chunks), 128)
    norms = np.linalg.norm(E, axis=1)
    assert np.allclose(norms[norms > 0], 1.0) and (norms > 0).mean() > 0.99, "rows are unit length (or all zero)"
    q = embedder.encode(["exceptions try except finally"])
    assert q.shape == (1, 128)
    again = LSAEmbedder(128).fit([t for _, t in chunks]).encode(["exceptions try except finally"])
    assert np.allclose(q, again), "seeded: the same fit gives the same vectors"


def test_top_k():
    rng = np.random.default_rng(0)
    M = normalize(rng.normal(size=(500, 16)))
    Q = normalize(rng.normal(size=(3, 16)))
    idx, scores = top_k(Q, M, 5)
    full = Q @ M.T
    assert idx.shape == (3, 5) and scores.shape == (3, 5)
    assert np.array_equal(idx, np.argsort(-full, axis=1)[:, :5])
    assert np.allclose(scores, np.sort(full, axis=1)[:, ::-1][:, :5])
    assert top_k(Q[0], M[:3], 10)[0].shape == (1, 3)
    assert not contains(top_k, LOOP_NODES)


def test_flat_index():
    index = FlatIndex()
    index.add(np.array([[1.0, 0.0], [0.0, 2.0]]), ["a", "b"])
    index.add(np.array([[1.0, 1.0]]), ["c"])
    results = index.search(np.array([3.0, 0.1]), k=2)
    assert [r[0] for r in results] == ["a", "c"]
    assert results[0][1] == pytest.approx(3 / np.sqrt(9.01))


def test_retrieval_quality(corpus):
    chunks, embedder, E = corpus
    index = FlatIndex()
    index.add(E, [m for m, _ in chunks])
    sources = [[m for m, _ in index.search(embedder.encode([q])[0], k=5)] for q, _ in QUERIES]
    assert hit_rate(sources, [m for _, m in QUERIES]) >= 0.7


def test_hit_rate_and_recall():
    assert hit_rate([["m01", "m02"], ["m03"]], ["m02", "m04"]) == 0.5
    assert recall_at_k([[1, 2, 3], [4, 5, 9]], [[1, 2, 4], [4, 5, 6]]) == pytest.approx(2 / 3)


@pytest.mark.timeout(60)
def test_ivf_trade_off(corpus):
    chunks, embedder, E = corpus
    flat = FlatIndex()
    flat.add(E, list(range(len(E))))
    queries = embedder.encode([q for q, _ in QUERIES] + [t[:200] for _, t in chunks[::50]])
    exact = [[i for i, _ in flat.search(q, 10)] for q in queries]
    recalls, scanned = {}, {}
    for n_probe in (1, 4, 16):
        ivf = IVFIndex(n_lists=16, n_probe=n_probe).train(E)
        ivf.add(E, list(range(len(E))))
        assert sum(len(l) for l in ivf.lists) == len(E)
        approx, counts = [], []
        for q in queries:
            approx.append([i for i, _ in ivf.search(q, 10)])
            counts.append(ivf.last_scanned)
        recalls[n_probe], scanned[n_probe] = recall_at_k(approx, exact), np.mean(counts)
    assert recalls[16] == pytest.approx(1.0), "probing every list is exact"
    assert recalls[1] < recalls[4] <= recalls[16]
    assert scanned[1] < 0.25 * len(E), "one probe scans only a fraction of the data"
    assert scanned[16] == len(E)
