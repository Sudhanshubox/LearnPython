import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import (
    CHARS,
    RewardModel,
    bradley_terry_loss,
    dpo_loss,
    encode,
    hidden_states,
    kl_penalized_reward,
    make_tasks,
    pad_batch,
    preference_probability,
    sequence_logprob,
    sft_train,
    train_dpo,
)
from minigpt import GPT, GPTConfig

STOI = {c: i for i, c in enumerate(CHARS)}


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=len(CHARS), block_size=16, n_layer=2, n_head=2, d_model=32))


def test_bradley_terry_loss():
    rc, rr = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0])
    expected = -(math.log(1 / (1 + math.exp(-2))) + math.log(0.5)) / 2
    assert bradley_terry_loss(rc, rr).item() == pytest.approx(expected)
    assert bradley_terry_loss(rc + 5, rr + 5).item() == pytest.approx(expected), "only differences matter"


def test_hidden_states():
    model = tiny()
    idx = torch.randint(0, len(CHARS), (2, 7))
    h = hidden_states(model, idx)
    assert h.shape == (2, 7, 32)
    assert torch.allclose(model.lm_head(h), model(idx)[0], atol=1e-5)


def test_pad_batch():
    idx, lengths = pad_batch([[1, 2, 3], [4]], pad_id=0)
    assert idx.dtype == torch.int64 and idx.tolist() == [[1, 2, 3], [4, 0, 0]] and lengths == [3, 1]


def test_reward_model_reads_last_real_token():
    rm = RewardModel(tiny())
    assert isinstance(rm.head, nn.Linear) and rm.head.out_features == 1
    seqs = [encode("Q: abc\nA: yes!\n", STOI), encode("Q: abc\nA: no.\n", STOI)]
    idx, lengths = pad_batch(seqs)
    r = rm(idx, lengths)
    assert r.shape == (2,)
    alone = rm(torch.tensor([seqs[1]]), [len(seqs[1])])
    assert torch.allclose(r[1], alone[0], atol=1e-5), "padding must not affect the reward"


@pytest.mark.timeout(60)
def test_reward_model_learns_preferences():
    tasks, test = make_tasks(300), make_tasks(50, seed=1)
    rm = RewardModel(tiny())
    opt = torch.optim.AdamW(rm.parameters(), lr=3e-3)
    for step in range(40):
        batch = tasks[(step * 32) % 280:(step * 32) % 280 + 32]
        idx_c, len_c = pad_batch([encode(p + c, STOI) for p, c, _ in batch])
        idx_r, len_r = pad_batch([encode(p + r, STOI) for p, _, r in batch])
        loss = bradley_terry_loss(rm(idx_c, len_c), rm(idx_r, len_r))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        idx_c, len_c = pad_batch([encode(p + c, STOI) for p, c, _ in test])
        idx_r, len_r = pad_batch([encode(p + r, STOI) for p, _, r in test])
        assert (rm(idx_c, len_c) > rm(idx_r, len_r)).float().mean() == 1.0


def test_sequence_logprob():
    model = tiny()
    prompts, responses = ["Q: abc\nA: ", "Q: xy\nA: "], ["yes!\n", "no.\n"]
    lp = sequence_logprob(model, prompts, responses, STOI)
    assert lp.shape == (2,) and lp.requires_grad
    for i in range(2):
        ids = torch.tensor([encode(prompts[i] + responses[i], STOI)])
        logp = torch.log_softmax(model(ids[:, :-1])[0], dim=-1)[0]
        n = len(prompts[i])
        expected = sum(logp[t, ids[0, t + 1]] for t in range(n - 1, ids.shape[1] - 1))
        assert lp[i].item() == pytest.approx(expected.item(), rel=1e-5)


def test_kl_penalized_reward():
    out = kl_penalized_reward(torch.tensor([1.0, 1.0]), torch.tensor([-2.0, -5.0]), torch.tensor([-3.0, -5.0]), 0.5)
    assert torch.allclose(out, torch.tensor([0.5, 1.0]))


def test_dpo_loss():
    pc, pr = torch.tensor([-1.0, -2.0], requires_grad=True), torch.tensor([-3.0, -1.0], requires_grad=True)
    rc, rr = torch.tensor([-2.0, -2.0]), torch.tensor([-2.0, -2.0])
    loss, cr, rj, acc = dpo_loss(pc, pr, rc, rr, beta=0.5)
    margins = 0.5 * ((pc - rc) - (pr - rr))
    assert loss.item() == pytest.approx(-F.logsigmoid(margins).mean().item())
    assert torch.allclose(cr, torch.tensor([0.5, 0.0])) and torch.allclose(rj, torch.tensor([-0.5, 0.5]))
    assert not cr.requires_grad and acc == 0.5
    loss.backward()
    assert pc.grad is not None
    same, *_ = dpo_loss(rc, rr, rc, rr)
    assert same.item() == pytest.approx(math.log(2)), "policy == reference gives log 2"


@pytest.fixture(scope="module")
def sft_model():
    return sft_train(tiny(), make_tasks(400), STOI, steps=200)


@pytest.mark.timeout(60)
def test_sft_is_indifferent(sft_model):
    p = preference_probability(sft_model, make_tasks(100, seed=1), STOI)
    assert isinstance(p, float) and 0.25 < p < 0.75


def drift(policy, reference, tasks):
    prompts = [p for p, _, _ in tasks]
    with torch.no_grad():
        total = 0.0
        for k in (1, 2):
            answers = [t[k] for t in tasks]
            total += (sequence_logprob(policy, prompts, answers, STOI) - sequence_logprob(reference, prompts, answers, STOI)).abs().mean().item()
    return total


@pytest.mark.timeout(60)
def test_dpo(sft_model):
    tasks, test = make_tasks(400), make_tasks(100, seed=1)
    results = {}
    for beta in (0.1, 1.0):
        policy, reference = copy.deepcopy(sft_model), copy.deepcopy(sft_model)
        losses = train_dpo(policy, reference, tasks, STOI, beta=beta, steps=60)
        assert len(losses) == 60 and losses[0] == pytest.approx(math.log(2), abs=1e-4)
        assert all(not p.requires_grad for p in reference.parameters())
        assert drift(reference, sft_model, test) == 0.0, "the reference must not change"
        results[beta] = (preference_probability(policy, test, STOI), drift(policy, reference, test))
    assert results[0.1][0] > 0.95 and results[1.0][0] > 0.95, "DPO teaches the preference"
    assert results[1.0][1] < results[0.1][1], "a larger beta keeps the policy closer to the reference"
