"""Tests for the searchkit project, grouped by file. Work through them in order."""

import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).parent

# ---------------------------------------------------------------- text.py


def test_tokenize():
    from searchkit.text import tokenize

    assert tokenize("The cat, and THE hat!") == ["cat", "hat"]
    assert tokenize("GPT-4 has 2 heads") == ["gpt", "4", "2", "heads"]
    assert tokenize("") == []
    assert tokenize("the the cat cat") == ["cat", "cat"]


# ---------------------------------------------------------------- index.py


@pytest.fixture
def index():
    from searchkit.index import InvertedIndex

    idx = InvertedIndex()
    idx.add("d1", "the cat sat on the mat with another cat")
    idx.add("d2", "the dog chased the cat")
    idx.add("d3", "dogs and cats are pets")
    return idx


def test_index_postings(index):
    assert index.postings("cat") == {"d1": 2, "d2": 1}
    assert index.postings("unicorn") == {}
    assert index.doc_freq("cat") == 2
    assert index.doc_freq("pets") == 1


def test_index_postings_are_protected(index):
    index.postings("cat")["d9"] = 100
    assert index.postings("cat") == {"d1": 2, "d2": 1}


def test_index_sizes(index):
    assert index.num_docs == 3
    assert index.doc_length("d1") == 5          # cat sat mat another cat ("the", "on", "with" are stopwords)
    assert index.doc_length("d2") == 3          # dog chased cat
    assert index.avg_doc_length == pytest.approx((5 + 3 + 3) / 3)
    assert "d2" in index and "d9" not in index


def test_index_rejects_duplicates(index):
    with pytest.raises(ValueError):
        index.add("d1", "again")


def test_empty_index():
    from searchkit.index import InvertedIndex

    assert InvertedIndex().avg_doc_length == 0.0


# ---------------------------------------------------------------- ranking.py


def test_tf_idf(index):
    from searchkit.ranking import tf_idf

    assert tf_idf(index, ["cat"], "d1") == pytest.approx(2 * math.log(3 / 2))
    assert tf_idf(index, ["cat", "cat", "mat"], "d1") == pytest.approx(2 * math.log(3 / 2) + math.log(3))
    assert tf_idf(index, ["unicorn"], "d1") == 0
    assert tf_idf(index, ["pets"], "d1") == 0


def bm25_reference(tf, df, n, dl, avgdl, k1=1.5, b=0.75):
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def test_bm25(index):
    from searchkit.ranking import bm25

    avgdl = 11 / 3
    assert bm25(index, ["cat"], "d1") == pytest.approx(bm25_reference(2, 2, 3, 5, avgdl))
    expected = bm25_reference(1, 2, 3, 3, avgdl) + bm25_reference(1, 1, 3, 3, avgdl)
    assert bm25(index, ["cat", "dog", "dog"], "d2") == pytest.approx(expected)
    assert bm25(index, ["unicorn"], "d2") == 0
    assert bm25(index, ["cat"], "d1", k1=1.2, b=0.5) == pytest.approx(bm25_reference(2, 2, 3, 5, avgdl, 1.2, 0.5))


def test_bm25_saturates_and_penalizes_length():
    from searchkit.index import InvertedIndex
    from searchkit.ranking import bm25, tf_idf

    idx = InvertedIndex()
    idx.add("few", "ai " * 2 + "x")
    idx.add("many", "ai " * 20 + "x")
    idx.add("other", "y z")
    ratio_tfidf = tf_idf(idx, ["ai"], "many") / tf_idf(idx, ["ai"], "few")
    ratio_bm25 = bm25(idx, ["ai"], "many") / bm25(idx, ["ai"], "few")
    assert ratio_tfidf == pytest.approx(10)
    assert ratio_bm25 < 2, "BM25 should saturate: 10x the mentions is not 10x the score"


# ---------------------------------------------------------------- engine.py


@pytest.fixture
def engine():
    from searchkit import SearchEngine

    e = SearchEngine()
    for path in sorted((HERE / "docs").glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        e.add_document(path.stem, text, text.splitlines()[0])
    return e


def test_engine_basics(engine):
    from searchkit import SearchEngine

    assert len(engine) == 8
    with pytest.raises(ValueError):
        SearchEngine(ranker="magic")


def test_search_ranking(engine):
    results = engine.search("how do neural networks learn")
    assert [r.doc_id for r in results] == ["backprop", "neural_networks"]
    assert results[0].title == "Backpropagation"
    assert results[0].score == round(results[0].score, 3)
    assert results[0].score > results[1].score > 0


def test_search_result_fields(engine):
    from searchkit import SearchResult

    top = engine.search("attention cost sequence length", k=1)[0]
    assert isinstance(top, SearchResult)
    assert top.doc_id == "attention"
    assert top.snippet == "Attention Attention lets every token in a sequence look at every other token and"
    assert "\n" not in top.snippet and len(top.snippet) <= 80


def test_search_k_and_empty_queries(engine):
    assert len(engine.search("gradient", k=1)) == 1
    assert len(engine.search("the data", k=10)) <= 10
    assert engine.search("") == []
    assert engine.search("the and of") == []          # only stopwords
    assert engine.search("quantum blockchain") == []


def test_search_ties_sorted_by_doc_id():
    from searchkit import SearchEngine

    e = SearchEngine()
    e.add_document("b", "same words here")
    e.add_document("a", "same words here")
    e.add_document("c", "different")
    assert [r.doc_id for r in e.search("same")] == ["a", "b"]


def test_tfidf_ranker(engine):
    from searchkit import SearchEngine

    e = SearchEngine(ranker="tfidf")
    e.add_document("x", "gradient gradient gradient")
    e.add_document("y", "gradient")
    e.add_document("z", "nothing here")
    results = e.search("gradient")
    assert [r.doc_id for r in results] == ["x", "y"]
    assert results[0].score == round(3 * math.log(3 / 2), 3)


def test_save_and_load(engine, tmp_path):
    from searchkit import SearchEngine

    path = tmp_path / "index.json"
    engine.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ranker"] == "bm25"
    assert [d["id"] for d in data["documents"]][:2] == ["attention", "backprop"]
    loaded = SearchEngine.load(path)
    assert len(loaded) == 8
    assert loaded.search("decision trees gini") == engine.search("decision trees gini")


def test_public_api():
    import searchkit

    assert searchkit.__version__ == "1.0.0"
    assert sorted(searchkit.__all__) == ["SearchEngine", "SearchResult", "tokenize"]


# ---------------------------------------------------------------- __main__.py


def cli(*args, cwd=HERE):
    return subprocess.run([sys.executable, "-m", "searchkit", *args], capture_output=True, text=True, cwd=cwd, timeout=60)


def test_cli_index_and_search(tmp_path):
    out = tmp_path / "index.json"
    result = cli("index", "docs", "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Indexed 8 documents."
    result = cli("search", str(out), "how do neural networks learn")
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("1. backprop (")
    assert lines[1].startswith("2. neural_networks (")
    assert cli("search", str(out), "zzz").stdout.strip() == "No results."
    assert len(cli("search", str(out), "gradient", "-k", "1").stdout.strip().splitlines()) == 1


def test_cli_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = cli("index", str(empty), "--out", str(tmp_path / "i.json"))
    assert result.returncode == 1
    assert "no .txt files" in result.stdout + result.stderr


# ---------------------------------------------------------------- exercises.py


def test_metrics():
    from exercises import mean_reciprocal_rank, precision_at_k, reciprocal_rank

    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert precision_at_k(["a"], {"a"}, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0
    assert mean_reciprocal_rank([["a"], ["x", "b"]], [{"a"}, {"b"}]) == pytest.approx(0.75)
    assert mean_reciprocal_rank([], []) == 0.0


def test_evaluate_your_engine(engine):
    from exercises import mean_reciprocal_rank

    queries = {
        "chain rule gradient weights": {"backprop"},
        "learning rate step size": {"gradient_descent"},
        "byte pair merges vocabulary": {"tokenization"},
        "validation loss memorizes": {"overfitting"},
        "cosine similarity vectors": {"embeddings"},
        "gini impurity random forests": {"decision_trees"},
    }
    results = [[r.doc_id for r in engine.search(q)] for q in queries]
    assert mean_reciprocal_rank(results, list(queries.values())) == 1.0


# ---------------------------------------------------------------- quality


def test_pyproject():
    data = tomllib.loads((HERE / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert (project["name"], project["version"], project["requires-python"]) == ("searchkit", "1.0.0", ">=3.11")
    assert project.get("description")
    assert project["dependencies"] == []
    assert project["scripts"]["searchkit"] == "searchkit.__main__:main"


@pytest.mark.timeout(120)
def test_mypy():
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--disallow-untyped-defs", "--no-incremental", "searchkit", "exercises.py"],
        capture_output=True, text=True, cwd=HERE,
    )
    assert result.returncode == 0, "mypy found problems:\n" + result.stdout
