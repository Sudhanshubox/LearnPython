import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    LoRALinear,
    apply_lora,
    build_sft_example,
    compare_lora_and_full,
    evaluate_lm,
    get_batch,
    load_corpora,
    merge_lora,
    sft_loss,
    train_lm,
    trainable_parameters,
)
from minigpt import GPT, GPTConfig


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def small_gpt(V=30):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=V, block_size=16, n_layer=2, n_head=2, d_model=16))


def test_lora_linear():
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4, alpha=8)
    assert lora.base is base and not any(p.requires_grad for p in base.parameters())
    assert lora.A.shape == (4, 32) and lora.B.shape == (16, 4) and lora.scaling == 2.0
    assert not lora.B.any(), "B starts at zero"
    assert lora.A.std().item() == pytest.approx(1 / math.sqrt(32), rel=0.3)
    x = torch.randn(5, 32, generator=gen(1))
    assert torch.allclose(lora(x), base(x)), "at init, LoRA changes nothing"
    with torch.no_grad():
        lora.B.normal_(generator=gen(2))
    expected = base(x) + 2.0 * (x @ (lora.B @ lora.A).T)
    assert torch.allclose(lora(x), expected, atol=1e-5)
    assert {n for n, p in lora.named_parameters() if p.requires_grad} == {"A", "B"}


def test_lora_merge():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(8, 6), r=2, alpha=4)
    with torch.no_grad():
        lora.B.normal_(generator=gen(3))
    merged = lora.merged()
    assert type(merged) is nn.Linear
    x = torch.randn(4, 8, generator=gen(4))
    assert torch.allclose(merged(x), lora(x), atol=1e-5)
    no_bias = LoRALinear(nn.Linear(8, 6, bias=False))
    assert no_bias.merged().bias is None


def test_apply_lora_and_merge_model():
    model = small_gpt()
    total = sum(p.numel() for p in model.parameters())
    assert apply_lora(model, r=4, alpha=8) is model
    block = model.blocks[0]
    assert all(isinstance(m, LoRALinear) for m in (block.attn.qkv, block.attn.proj, block.mlp.fc, block.mlp.proj))
    per_block = 4 * (16 + 48) + 4 * (16 + 16) + 4 * (16 + 64) + 4 * (64 + 16)
    assert trainable_parameters(model) == 2 * per_block
    assert not model.tok_emb.weight.requires_grad
    only_qkv = apply_lora(small_gpt(), r=4, targets=("qkv",))
    assert isinstance(only_qkv.blocks[1].attn.qkv, LoRALinear)
    assert type(only_qkv.blocks[1].attn.proj) is nn.Linear and type(only_qkv.blocks[1].mlp.fc) is nn.Linear
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.B.normal_(0, 0.1, generator=gen(5))
    idx = torch.randint(0, 30, (2, 10), generator=gen(6))
    before = model(idx)[0]
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(model(idx)[0], before, atol=1e-5)
    assert sum(p.numel() for p in model.parameters()) == total


def test_build_sft_example():
    stoi = {c: i + 1 for i, c in enumerate("QA: \nhiyo")}
    x, y = build_sft_example("Q: hi\nA: ", "yo\n", stoi, block_size=16)
    ids = [stoi[c] for c in "Q: hi\nA: yo\n"]
    assert x.dtype == torch.int64 and x.shape == (16,) and y.shape == (16,)
    assert x.tolist() == ids[:-1] + [0] * 5
    assert y.tolist() == [-100] * 8 + [stoi["y"], stoi["o"], stoi["\n"]] + [-100] * 5
    with pytest.raises(ValueError):
        build_sft_example("Q: hi\nA: ", "yo\n", stoi, block_size=5)


def test_sft_loss():
    logits = torch.randn(2, 4, 5, generator=gen(0))
    y = torch.tensor([[-100, 1, 2, -100], [-100, -100, 3, -100]])
    keep = y != -100
    expected = F.cross_entropy(logits[keep], y[keep])
    assert sft_loss(logits, y).item() == pytest.approx(expected.item(), rel=1e-6)


def test_get_batch_and_train_lm():
    data = torch.arange(100) % 7
    x, y = get_batch(data, 8, 4, gen(0))
    assert x.shape == (4, 8) and torch.equal(y, data[torch.randint(0, 92, (4,), generator=gen(0))[:, None] + torch.arange(8) + 1])
    model = apply_lora(small_gpt(V=7), r=2)
    frozen = model.tok_emb.weight.clone()
    before = evaluate_lm(model, data, batches=2, batch_size=8)
    assert model.training
    train_lm(model, data, steps=60, lr=3e-2, batch_size=8)
    assert torch.equal(model.tok_emb.weight, frozen), "frozen weights must not change"
    assert evaluate_lm(model, data, batches=2, batch_size=8) < before - 0.1


@pytest.fixture(scope="module")
def experiment():
    stoi, itos, (eng_train, eng_val), (code_train, code_val) = load_corpora()
    torch.manual_seed(0)
    base = GPT(GPTConfig(vocab_size=len(itos), block_size=64, n_layer=2, n_head=4, d_model=64))
    train_lm(base, eng_train, steps=500, lr=6e-3)
    snapshot = copy.deepcopy(base.state_dict())
    results = compare_lora_and_full(base, eng_val, code_train, code_val)
    return base, snapshot, results


@pytest.mark.timeout(120)
def test_lora_learns_less_and_forgets_less(experiment):
    base, snapshot, r = experiment
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in snapshot.items()), "don't modify base"
    assert set(r) == {"base", "lora", "full"}
    assert r["lora"]["trainable"] < 0.2 * r["full"]["trainable"]
    assert r["full"]["trainable"] == r["base"]["trainable"]
    assert r["lora"]["code"] < r["base"]["code"] - 0.25, "LoRA learns the new domain"
    assert r["full"]["code"] < r["lora"]["code"], "...but less than full fine-tuning"
    forget_lora = r["lora"]["english"] - r["base"]["english"]
    forget_full = r["full"]["english"] - r["base"]["english"]
    assert forget_full > forget_lora, "...and forgets less"
