import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from code_checks import LOOP_NODES, contains
from exercises import (
    MLPLanguageModel,
    NeuralBigram,
    NGramModel,
    bigram_counts,
    bigram_nll,
    bigram_probs,
    build_vocab,
    evaluate_nll,
    make_context_dataset,
    nearest_neighbors,
    perplexity,
    split_ids,
    train_lm,
)

CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())


def load(pattern):
    text = "".join(p.read_text(encoding="utf-8") for p in sorted(CURRICULUM.glob(pattern)))
    stoi, itos = build_vocab(text)
    ids = [stoi[c] for c in text]
    return stoi, itos, split_ids(ids)


@pytest.fixture(scope="module")
def small():
    return load("phase01*/*/README.md")


@pytest.fixture(scope="module")
def big():
    return load("phase0[1-3]*/*/README.md")


def test_build_vocab_and_split():
    stoi, itos = build_vocab("hello")
    assert itos == ["e", "h", "l", "o"] and stoi == {"e": 0, "h": 1, "l": 2, "o": 3}
    assert split_ids(list(range(10))) == (list(range(9)), [9])
    assert split_ids(list(range(10)), 0.3) == (list(range(7)), [7, 8, 9])


def test_bigram_counts_and_probs():
    c = bigram_counts([0, 1, 1, 2, 0, 1], 3)
    assert c.shape == (3, 3) and c.dtype.is_floating_point
    assert torch.equal(c, torch.tensor([[0.0, 2, 0], [0, 1, 1], [1, 0, 0]]))
    assert not contains(bigram_counts, LOOP_NODES)
    p = bigram_probs(c, alpha=1.0)
    assert torch.allclose(p.sum(dim=1), torch.ones(3))
    assert p[0, 1].item() == pytest.approx(3 / 5)
    assert bigram_probs(c, alpha=0.0)[0, 1].item() == 1.0


def test_bigram_nll_and_perplexity():
    probs = torch.full((4, 4), 0.25)
    assert bigram_nll(probs, [0, 1, 2, 3, 0]) == pytest.approx(math.log(4))
    assert perplexity(math.log(4)) == pytest.approx(4.0), "a uniform model's perplexity is V"
    p = bigram_probs(bigram_counts([0, 1, 0, 1], 2), 0.0)
    assert bigram_nll(p, [0, 1, 0]) == pytest.approx(0.0)


def test_ngram_model_small():
    m = NGramModel(2, V=3, alpha=1.0).fit([0, 1, 1, 2, 0, 1])
    assert m.prob([2, 0], 1) == pytest.approx((2 + 1) / (2 + 3))
    assert m.prob([5, 1], 2) == pytest.approx((1 + 1) / (2 + 3))
    uni = NGramModel(1, V=3, alpha=0.0).fit([0, 1, 1, 2])
    assert uni.prob([], 1) == pytest.approx(0.5)
    tri = NGramModel(3, V=3, alpha=1.0).fit([0, 1, 2])
    assert tri.prob([2, 2], 0) == pytest.approx(1 / 3), "unseen contexts are uniform"
    assert m.nll([0, 1, 1]) == pytest.approx(-(math.log(3 / 5) + math.log(2 / 5)) / 2)


@pytest.mark.timeout(60)
def test_curse_of_sparsity(big):
    stoi, itos, (train, val) = big
    V = len(itos)
    results = {n: NGramModel(n, V, alpha=0.01).fit(train) for n in (1, 2, 4, 6)}
    tr = {n: perplexity(m.nll(train[:20000])) for n, m in results.items()}
    va = {n: perplexity(m.nll(val[:20000])) for n, m in results.items()}
    assert tr[1] > tr[2] > tr[4] > tr[6], "longer contexts always fit the training data better"
    assert va[4] < va[2] < va[1]
    assert va[6] > va[4], "but too-long contexts are too sparse to generalize"
    probs = bigram_probs(bigram_counts(train, V), 0.01)
    assert perplexity(bigram_nll(probs, val)) == pytest.approx(perplexity(results[2].nll(val)), rel=1e-4)


@pytest.mark.timeout(60)
def test_neural_bigram_matches_counts(small):
    stoi, itos, (train, val) = small
    V = len(itos)
    torch.manual_seed(0)
    model = NeuralBigram(V)
    assert isinstance(model.logits, nn.Embedding) and not model.logits.weight.any()
    X, Y = torch.tensor(train[:-1]), torch.tensor(train[1:])
    assert evaluate_nll(model, X, Y) == pytest.approx(math.log(V), rel=1e-5)
    losses = train_lm(model, X, Y, steps=300, batch_size=1024, lr=0.3)
    assert len(losses) == 300 and all(isinstance(v, float) for v in losses)
    assert model.training
    count_nll = bigram_nll(bigram_probs(bigram_counts(train, V), 0.1), train)
    assert evaluate_nll(model, X, Y) == pytest.approx(count_nll, abs=0.1), "gradient descent rediscovers the counts"


def test_make_context_dataset():
    X, Y = make_context_dataset([5, 6, 7, 8, 9], 3)
    assert X.dtype == torch.int64 and torch.equal(X, torch.tensor([[5, 6, 7], [6, 7, 8]]))
    assert torch.equal(Y, torch.tensor([8, 9]))
    assert not contains(make_context_dataset, LOOP_NODES)


def test_mlp_lm_shapes():
    torch.manual_seed(0)
    m = MLPLanguageModel(V=20, block_size=4, embed_dim=3, hidden=7)
    assert m.embed.weight.shape == (20, 3) and m.hidden.in_features == 12 and m.out.out_features == 20
    x = torch.randint(0, 20, (5, 4))
    out = m(x)
    assert out.shape == (5, 20)
    assert torch.allclose(out, m.out(torch.tanh(m.hidden(m.embed(x).reshape(5, -1)))))


@pytest.mark.timeout(120)
def test_mlp_beats_bigram(big):
    stoi, itos, (train, val) = big
    V = len(itos)
    X, Y = make_context_dataset(train, 8)
    Xv, Yv = make_context_dataset(val, 8)
    torch.manual_seed(0)
    model = MLPLanguageModel(V, 8)
    losses = train_lm(model, X, Y, steps=2000)
    assert sum(losses[-100:]) / 100 < 0.6 * losses[0]
    mlp_ppl = perplexity(evaluate_nll(model, Xv, Yv))
    bigram_ppl = perplexity(bigram_nll(bigram_probs(bigram_counts(train, V), 1.0), val))
    assert mlp_ppl < 0.85 * bigram_ppl
    neighbors = nearest_neighbors(model.embed.weight.detach(), stoi["a"], k=5)
    assert len(neighbors) == 5 and stoi["a"] not in neighbors


def test_nearest_neighbors():
    E = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [2.0, 0.1]])
    assert nearest_neighbors(E, 0, k=2) == [4, 1]
    assert nearest_neighbors(E, 2, k=1) == [1]
    assert E[0, 0] == 1.0
