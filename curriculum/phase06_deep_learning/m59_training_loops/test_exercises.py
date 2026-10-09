import math
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from exercises import (
    ArrayDataset,
    EarlyStopping,
    evaluate,
    fit,
    load_checkpoint,
    make_loaders,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    warmup_cosine_lr,
)


@pytest.fixture(scope="module")
def digits():
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.25, random_state=0, stratify=y)


def make_model(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 10))


def test_set_seed():
    set_seed(5)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(5)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == a


def test_array_dataset():
    X, y = np.arange(12, dtype=np.float64).reshape(4, 3), np.array([1, 0, 1, 2])
    ds = ArrayDataset(X, y)
    assert len(ds) == 4
    x2, y2 = ds[2]
    assert x2.dtype == torch.float32 and y2.dtype == torch.int64
    assert torch.equal(x2, torch.tensor([6.0, 7.0, 8.0])) and y2.item() == 1


def test_make_loaders(digits):
    X_train, X_val, y_train, y_val = digits
    tl, vl = make_loaders(X_train, y_train, X_val, y_val, batch_size=100, seed=1)
    xb, yb = next(iter(tl))
    assert xb.shape == (100, 64) and yb.shape == (100,)
    assert sum(len(b[0]) for b in tl) == len(X_train)
    assert torch.equal(next(iter(vl))[1], torch.as_tensor(y_val[:100])), "don't shuffle validation data"
    first = next(iter(tl))[1]
    assert not torch.equal(first, torch.as_tensor(y_train[:100])), "shuffle the training data"
    tl2, _ = make_loaders(X_train, y_train, X_val, y_val, batch_size=100, seed=1)
    assert torch.equal(next(iter(tl2))[1], yb), "the shuffle must be reproducible from the seed"


def test_train_one_epoch(digits):
    X_train, X_val, y_train, y_val = digits
    tl, vl = make_loaders(X_train, y_train, X_val, y_val, batch_size=64)
    model = make_model()
    model.eval()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_one_epoch(model, tl, opt, nn.CrossEntropyLoss())
    assert model.training, "call model.train()"
    assert isinstance(loss, float) and 0.5 < loss < math.log(10) + 0.3
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    second = train_one_epoch(model, tl, opt, nn.CrossEntropyLoss())
    assert second < loss


def test_train_one_epoch_weights_by_batch_size():
    X = torch.zeros(5, 1)
    y = torch.tensor([0, 0, 0, 0, 0])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=4)
    model = nn.Linear(1, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    def loss_fn(logits, yb):
        v = torch.tensor(float(len(yb)), requires_grad=True)   # loss = batch size
        return v * 1.0
    assert train_one_epoch(model, loader, opt, loss_fn) == pytest.approx((4 * 4 + 1 * 1) / 5)


def test_evaluate(digits):
    X_train, X_val, y_train, y_val = digits
    _, vl = make_loaders(X_train, y_train, X_val, y_val, batch_size=64)
    model = make_model(dropout=0.5)
    model.train()
    loss1, acc1 = evaluate(model, vl, nn.CrossEntropyLoss())
    loss2, acc2 = evaluate(model, vl, nn.CrossEntropyLoss())
    assert loss1 == loss2 and acc1 == acc2, "call model.eval() so dropout is off"
    assert isinstance(loss1, float) and isinstance(acc1, float)
    with torch.no_grad():
        model.eval()
        logits = model(torch.as_tensor(X_val, dtype=torch.float32))
    y = torch.as_tensor(y_val)
    assert acc1 == pytest.approx((logits.argmax(1) == y).float().mean().item())
    assert loss1 == pytest.approx(nn.functional.cross_entropy(logits, y).item(), rel=1e-5)


def test_evaluate_no_grad(digits):
    X_train, X_val, y_train, y_val = digits
    _, vl = make_loaders(X_train, y_train, X_val, y_val)
    seen = []
    def loss_fn(logits, yb):
        seen.append(logits.requires_grad)
        return nn.functional.cross_entropy(logits, yb)
    evaluate(make_model(), vl, loss_fn)
    assert not any(seen), "evaluate without recording gradients"


def test_warmup_cosine_lr():
    lrs = [warmup_cosine_lr(s, 100, 10, 1.0) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(1.0)
    assert lrs[55] == pytest.approx(0.5)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
    assert warmup_cosine_lr(100, 100, 10, 1.0) == pytest.approx(0.0)
    assert warmup_cosine_lr(100, 100, 10, 1.0, min_lr=0.1) == pytest.approx(0.1)
    assert warmup_cosine_lr(0, 10, 0, 3e-4) == pytest.approx(3e-4)


def test_checkpoint_roundtrip(tmp_path):
    model = make_model(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 64)).sum().backward()
    opt.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, epoch=7, best_metric=0.25)
    other = make_model(1)
    other_opt = torch.optim.AdamW(other.parameters(), lr=1e-3)
    assert load_checkpoint(path, other, other_opt) == (7, 0.25)
    for a, b in zip(model.parameters(), other.parameters()):
        assert torch.equal(a, b)
    assert other_opt.state_dict()["state"][0]["exp_avg"].shape == (64, 64), "restore the optimizer too"


def test_early_stopping():
    es = EarlyStopping(patience=2, min_delta=0.01)
    assert es.best == math.inf
    assert es.step(1.0) is False and es.improved and es.best == 1.0
    assert es.step(0.995) is False and not es.improved and es.best == 1.0
    assert es.step(0.5) is False and es.improved
    assert es.step(0.6) is False
    assert es.step(0.7) is True


@pytest.mark.timeout(120)
def test_fit(digits, tmp_path):
    X_train, X_val, y_train, y_val = digits
    tl, vl = make_loaders(X_train, y_train, X_val, y_val, batch_size=64, seed=0)
    model = make_model(0)
    path = tmp_path / "best.pt"
    history = fit(model, tl, vl, epochs=15, lr=1e-2, checkpoint_path=path, warmup_steps=20)
    assert set(history) == {"train_loss", "val_loss", "val_acc", "lr"}
    n = len(history["val_loss"])
    assert 1 <= n <= 15 and all(len(v) == n for v in history.values())
    assert max(history["val_acc"]) > 0.95
    if n == 15:
        assert history["lr"][-1] == pytest.approx(0.0, abs=1e-9), "cosine decays to 0 at the end"
    assert history["lr"][0] < 1e-2
    best = min(history["val_loss"])
    loss, _ = evaluate(model, vl, nn.CrossEntropyLoss())
    assert loss == pytest.approx(best, rel=1e-5), "reload the best checkpoint at the end"
    assert torch.load(path)["best_metric"] == pytest.approx(best)


@pytest.mark.timeout(120)
def test_fit_stops_early_and_is_reproducible(digits, tmp_path):
    X_train, X_val, y_train, y_val = digits
    tl, vl = make_loaders(X_train, y_train, X_val, y_val)
    history = fit(make_model(0), tl, vl, epochs=10, lr=0.0, checkpoint_path=tmp_path / "a.pt", patience=2)
    assert len(history["val_loss"]) == 3, "lr=0 never improves, so stop after 1 + patience epochs"
    runs = []
    for name in ["b.pt", "c.pt"]:
        tl, vl = make_loaders(X_train, y_train, X_val, y_val, seed=3)
        runs.append(fit(make_model(1), tl, vl, epochs=3, lr=1e-3, checkpoint_path=tmp_path / name))
    assert runs[0] == runs[1]
