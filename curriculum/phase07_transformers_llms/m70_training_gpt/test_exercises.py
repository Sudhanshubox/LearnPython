import math

import pytest
import torch

from code_checks import LOOP_NODES, contains
from exercises import (
    CURRICULUM,
    configure_optimizer,
    encode_dataset,
    estimate_loss,
    generate,
    get_batch,
    load_corpus,
    lr_at,
    train_gpt,
)
from minigpt import GPT, GPTConfig


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def dataset():
    return encode_dataset(load_corpus("phase0[1-3]*/*/README.md"))


def test_load_corpus():
    text = load_corpus("phase01*/m01*/README.md")
    assert text == (CURRICULUM / "phase01_foundations").glob("m01*/README.md").__next__().read_text(encoding="utf-8")
    two = load_corpus("phase01*/m0[12]*/README.md")
    paths = sorted(CURRICULUM.glob("phase01*/m0[12]*/README.md"))
    assert two == "\n\n".join(p.read_text(encoding="utf-8") for p in paths)


def test_encode_dataset():
    stoi, itos, train, val = encode_dataset("hello world", val_fraction=0.2)
    assert itos == sorted(set("hello world")) and stoi["h"] == itos.index("h")
    assert train.dtype == torch.int64 and len(train) == 8 and len(val) == 3
    assert "".join(itos[i] for i in torch.cat([train, val]).tolist()) == "hello world"


def test_get_batch():
    data = torch.arange(100)
    x, y = get_batch(data, 8, 5, gen(0))
    assert x.shape == (5, 8) and y.shape == (5, 8)
    assert torch.equal(y, x + 1)
    starts = torch.randint(0, 92, (5,), generator=gen(0))
    assert torch.equal(x[:, 0], starts)
    assert not contains(get_batch, LOOP_NODES)
    x, y = get_batch(torch.arange(10), 9, 50, gen(1))
    assert y.max() == 9, "windows must stay inside the data"


def test_estimate_loss():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, d_model=16, dropout=0.5))
    data = torch.randint(0, 20, (500,), generator=gen(0))
    model.train()
    a = estimate_loss(model, data, 8, eval_iters=5, generator=gen(1))
    assert model.training, "restore train mode"
    b = estimate_loss(model, data, 8, eval_iters=5, generator=gen(1))
    assert a == b, "evaluate in eval mode (dropout off)"
    assert isinstance(a, float) and a == pytest.approx(math.log(20), abs=0.15)
    model.eval()
    estimate_loss(model, data, 8, eval_iters=2, generator=gen(1))
    assert not model.training


def test_configure_optimizer():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=20, block_size=8, n_layer=1, n_head=2, d_model=16))
    opt = configure_optimizer(model, 1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))
    assert decay["betas"] == (0.9, 0.95) and decay["lr"] == 1e-3


def test_lr_at():
    assert lr_at(0, 100, 10, 1.0, 0.1) == pytest.approx(0.1)
    assert lr_at(9, 100, 10, 1.0, 0.1) == pytest.approx(1.0)
    assert lr_at(55, 100, 10, 1.0, 0.1) == pytest.approx(0.55)
    assert lr_at(100, 100, 10, 1.0, 0.1) == pytest.approx(0.1)
    assert lr_at(500, 100, 10, 1.0, 0.1) == pytest.approx(0.1)


@pytest.fixture(scope="module")
def trained(dataset):
    stoi, itos, train, val = dataset
    config = GPTConfig(vocab_size=len(itos), block_size=64, n_layer=2, n_head=4, d_model=64)
    return train_gpt(config, train, val, max_steps=600, eval_interval=300)


@pytest.mark.timeout(120)
def test_training_beats_bigram(dataset, trained):
    stoi, itos, train, val = dataset
    model, history = trained
    assert len(history["train_loss"]) == 600
    assert [s for s, _ in history["val_loss"]] == [300, 600]
    V = len(itos)
    counts = torch.zeros(V, V)
    counts.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1), accumulate=True)
    probs = (counts + 0.1) / (counts + 0.1).sum(1, keepdim=True)
    bigram = -torch.log(probs[val[:-1], val[1:]]).mean().item()
    final = history["val_loss"][-1][1]
    assert history["train_loss"][0] == pytest.approx(math.log(V), abs=0.2)
    assert final < bigram - 0.25, f"GPT {final:.3f} should clearly beat the bigram {bigram:.3f}"


@pytest.mark.timeout(60)
def test_generate(dataset, trained):
    stoi, itos, _, _ = dataset
    model, _ = trained
    model.train()
    prompt = torch.tensor([[stoi[c] for c in "The "]])
    out = generate(model, prompt, 30, temperature=0)
    assert not model.training
    assert out.shape == (1, 34) and torch.equal(out[:, :4], prompt)
    assert torch.equal(out, generate(model, prompt, 30, temperature=0))
    a = generate(model, prompt, 100, 1.0, top_k=5, generator=gen(3))
    b = generate(model, prompt, 100, 1.0, top_k=5, generator=gen(3))
    assert torch.equal(a, b)
    long = generate(model, prompt, 80, 1.0, generator=gen(4))
    assert long.shape == (1, 84), "crop the context to block_size"
    with torch.no_grad():
        logits = model(prompt)[0][0, -1]
    top1 = generate(model, prompt, 1, 1.0, top_k=1, generator=gen(5))[0, -1]
    assert top1 == logits.argmax(), "top_k=1 is greedy"
