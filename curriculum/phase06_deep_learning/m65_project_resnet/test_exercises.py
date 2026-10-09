import statistics

import pytest
import torch
import torch.nn as nn

from exercises import (
    BasicBlock,
    DeepNet,
    layer_grad_norms,
    load_data,
    run_experiment,
    summarize,
    train,
    write_report,
    zero_init_residual,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_load_data(data):
    X, y, Xv, yv = data
    assert X.shape == (1347, 1, 8, 8) and Xv.shape == (450, 1, 8, 8)
    assert X.dtype == torch.float32 and y.dtype == torch.int64
    assert float(X.max()) == 1.0 and float(X.min()) == 0.0
    assert abs(torch.bincount(y).float() / len(y) - torch.bincount(yv).float() / len(yv)).max() < 0.01


def test_basic_block():
    torch.manual_seed(0)
    block = BasicBlock(4, residual=True)
    plain = BasicBlock(4, residual=False)
    plain.load_state_dict(block.state_dict())
    assert [type(m) for m in (block.conv1, block.bn1, block.conv2, block.bn2)] == [nn.Conv2d, nn.BatchNorm2d] * 2
    assert block.conv1.bias is None and block.conv1.kernel_size == (3, 3)
    x = torch.randn(2, 4, 5, 5)
    inner = block.bn2(block.conv2(torch.relu(block.bn1(block.conv1(x)))))
    assert torch.allclose(block(x), torch.relu(inner + x), atol=1e-6)
    assert torch.allclose(plain(x), torch.relu(inner), atol=1e-6)
    assert block(x).shape == x.shape


def test_deep_net():
    torch.manual_seed(0)
    net = DeepNet(3, residual=True)
    assert len(net.blocks) == 3 and all(isinstance(b, BasicBlock) and b.residual for b in net.blocks)
    assert not any(b.residual for b in DeepNet(2, residual=False).blocks)
    assert sum(isinstance(m, nn.Conv2d) for m in net.modules()) == 7
    x = torch.randn(4, 1, 8, 8)
    assert net(x).shape == (4, 10)
    assert torch.allclose(net(x), net.head(net.blocks(net.stem(x)).mean(dim=(2, 3))))


@pytest.mark.timeout(60)
def test_train(data):
    torch.manual_seed(0)
    model = DeepNet(2, residual=True)
    h = train(model, data, epochs=3)
    assert set(h) == {"train_loss", "val_acc"} and len(h["train_loss"]) == 3
    assert all(isinstance(v, float) for v in h["train_loss"] + h["val_acc"])
    assert h["train_loss"][-1] < h["train_loss"][0]
    assert h["val_acc"][-1] > 0.8
    torch.manual_seed(0)
    assert train(DeepNet(2, residual=True), data, epochs=3) == h, "same seed, same run"


def test_summarize():
    rows = [{"n_blocks": 2, "residual": False, "seed": s, "final_train_loss": l, "final_val_acc": a}
            for s, l, a in [(0, 0.1, 0.9), (1, 0.3, 0.8)]]
    rows.append({"n_blocks": 12, "residual": True, "seed": 0, "final_train_loss": 0.2, "final_val_acc": 0.95})
    s = summarize(rows)
    assert set(s) == {(2, False), (12, True)}
    assert s[(2, False)]["train_loss_mean"] == pytest.approx(0.2)
    assert s[(2, False)]["train_loss_std"] == pytest.approx(statistics.stdev([0.1, 0.3]))
    assert s[(2, False)]["val_acc_mean"] == pytest.approx(0.85) and s[(2, False)]["n"] == 2
    assert s[(12, True)]["train_loss_std"] == 0.0


@pytest.fixture(scope="module")
def experiment(data):
    return run_experiment(data, depths=(2, 12), seeds=(0,), epochs=8)


@pytest.mark.timeout(120)
def test_degradation_reproduced(experiment):
    assert [(r["n_blocks"], r["residual"], r["seed"]) for r in experiment] == \
        [(2, False, 0), (2, True, 0), (12, False, 0), (12, True, 0)]
    s = summarize(experiment)
    plain_shallow = s[(2, False)]["train_loss_mean"]
    plain_deep = s[(12, False)]["train_loss_mean"]
    res_deep = s[(12, True)]["train_loss_mean"]
    assert plain_deep > 3 * plain_shallow, "the deep plain net should train WORSE than the shallow one"
    assert res_deep < 0.5 * plain_deep, "residual connections should fix it"
    assert s[(12, True)]["val_acc_mean"] > 0.9


def test_layer_grad_norms(data):
    xb, yb = data[0][:256], data[1][:256]
    torch.manual_seed(0)
    small = DeepNet(1, residual=True)
    norms = layer_grad_norms(small, xb, yb)
    assert len(norms) == 3 and all(isinstance(v, float) for v in norms)
    assert norms[0] == pytest.approx(small.stem[0].weight.grad.norm().item())
    spreads = {}
    for residual in (False, True):
        torch.manual_seed(0)
        g = layer_grad_norms(DeepNet(24, residual), xb, yb)
        assert len(g) == 49
        spreads[residual] = max(g) / min(g)
    assert spreads[False] > 100, "deep plain BN nets have exploding gradients toward the input"
    assert spreads[True] < 30


def test_zero_init_residual():
    torch.manual_seed(0)
    net = zero_init_residual(DeepNet(4, residual=True))
    assert all((b.bn2.weight == 0).all() for b in net.blocks)
    assert all((b.bn1.weight == 1).all() for b in net.blocks)
    net.eval()
    h = net.stem(torch.randn(3, 1, 8, 8))
    assert torch.allclose(net.blocks(h), h), "every block starts as the identity"
    plain = zero_init_residual(DeepNet(2, residual=False))
    assert all((b.bn2.weight == 1).all() for b in plain.blocks)


def test_report(tmp_path, experiment):
    s = summarize(experiment)
    path = tmp_path / "report.md"
    write_report(s, path)
    text = path.read_text(encoding="utf-8")
    for heading in ["# Residual vs plain networks", "## Setup", "## Results", "## Analysis", "## Ablations", "## Limitations"]:
        assert heading in text
    table_rows = [line for line in text.splitlines() if line.startswith("|") and "---" not in line]
    assert len(table_rows) >= 5, "a header row plus one row per configuration"
    for key, stats in s.items():
        assert f"{stats['train_loss_mean']:.3f}" in text
    assert "25" in text, "include the number of conv layers"
    assert len(text.split()) > 120, "write real sentences about your findings"
