"""m74 exercises: reward models and DPO on a toy preference task. The model is in minigpt.py."""

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt import GPT, GPTConfig

CHARS = "\nQAabcdefghijklmnopqrstuvwxyz!.:? "


def make_tasks(n, seed=0):
    """Given: prompts like "Q: abc\\nA: ", each with a preferred and a dispreferred answer."""
    rng = random.Random(seed)
    tasks = []
    for _ in range(n):
        word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(3))
        tasks.append((f"Q: {word}\nA: ", "yes!\n", "no.\n"))
    return tasks


def encode(text, stoi):
    """Given: the list of ids of the characters of text."""
    return [stoi[c] for c in text]


# 1. The Bradley-Terry loss: -mean(log sigmoid(r_chosen - r_rejected)). Use F.logsigmoid.
def bradley_terry_loss(r_chosen, r_rejected):
    raise NotImplementedError


# 2. The final hidden states of a minigpt GPT (B, T, d): token + position embeddings, every
#    block, then ln_f. (The same as the model's forward pass without lm_head.)
def hidden_states(model, idx):
    raise NotImplementedError


# 3. A reward model: self.base (a GPT) and self.head = nn.Linear(d_model, 1).
#    forward(idx, lengths) returns a (B,) tensor: the head applied to the hidden state at
#    the LAST REAL position of each sequence (index lengths[b] - 1). No loops.
class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, lengths):
        raise NotImplementedError


# 4. Right-pad lists of ids with pad_id to the longest length. Return (int64 tensor (B, T),
#    list of the original lengths).
def pad_batch(sequences, pad_id=0):
    raise NotImplementedError


# 5. log pi(response | prompt) for each pair (README section 5): encode prompt + response,
#    pad_batch them, run the model on idx[:, :-1], take log_softmax, gather the log-probs of
#    the actual next tokens idx[:, 1:], and sum ONLY over the positions that predict response
#    tokens (for example i, positions len(prompt_i) - 1 up to length_i - 2). Return (B,).
#    Gradients must flow (don't use no_grad here).
def sequence_logprob(model, prompts, responses, stoi):
    raise NotImplementedError


# 6. The per-sample KL-penalized reward used in RLHF: reward - beta * (logp_policy - logp_ref).
def kl_penalized_reward(reward, logp_policy, logp_ref, beta):
    raise NotImplementedError


# 7. The DPO loss (README section 4). Inputs are (B,) sequence log-probs.
#    Return (loss, chosen_rewards, rejected_rewards, accuracy) where the rewards are the
#    DETACHED implicit rewards beta * (policy - reference) and accuracy is the float fraction
#    with chosen_reward > rejected_reward.
def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    raise NotImplementedError


# 8. SFT on the tasks: each step samples batch_size tasks with rng = random.Random(seed)
#    (rng.sample), picks the chosen or rejected answer with probability 0.5 each
#    (rng.random() < 0.5 -> chosen), and minimizes -mean(sequence_logprob) with
#    AdamW(lr, weight_decay=0). Return the model.
def sft_train(model, tasks, stoi, steps=300, lr=3e-3, batch_size=32, seed=0):
    raise NotImplementedError


# 9. DPO training: freeze the reference (eval mode, requires_grad False) and compute its
#    log-probs without gradients; each step samples batch_size tasks with
#    random.Random(seed).sample, computes dpo_loss with beta, and takes an AdamW(lr,
#    weight_decay=0) step on the policy. Return the list of losses (floats).
def train_dpo(policy, reference, tasks, stoi, beta=0.1, steps=100, lr=1e-3, batch_size=32, seed=0):
    raise NotImplementedError


# 10. The model's average preference for the chosen answer, in eval mode without gradients:
#     mean over tasks of sigmoid(log pi(chosen) - log pi(rejected)), as a float.
def preference_probability(model, tasks, stoi):
    raise NotImplementedError
