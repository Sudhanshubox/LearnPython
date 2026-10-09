"""m73 exercises: LoRA, SFT data, and LoRA vs full fine-tuning. The model is in minigpt.py."""

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt import GPT, GPTConfig

CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())


def load_corpora(val_fraction=0.1):
    """Given: English (lessons) and Python (test files) from Phases 1-4, character-level.

    Returns (stoi, itos, (english_train, english_val), (code_train, code_val)).
    """
    english = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(CURRICULUM.glob("phase0[1-4]*/*/README.md")))
    code = "\n\n".join(p.read_text(encoding="utf-8") for p in sorted(CURRICULUM.glob("phase0[1-4]*/*/test_exercises.py")))
    itos = sorted(set(english) | set(code))
    stoi = {c: i for i, c in enumerate(itos)}

    def split(text):
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_fraction))
        return data[:n], data[n:]

    return stoi, itos, split(english), split(code)


# 1. LoRA around a frozen nn.Linear (README section 4).
#    __init__: keep base as self.base and freeze its parameters; self.r = r,
#    self.scaling = alpha / r; self.A = Parameter(torch.randn(r, in) / sqrt(in)),
#    self.B = Parameter(zeros(out, r)).
#    forward: base(x) + (x @ A.T @ B.T) * scaling.
#    merged(): a NEW nn.Linear (same bias setting) with weight W + scaling * B @ A and the
#    base's bias.
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def merged(self):
        raise NotImplementedError


# 2. Freeze every parameter of the model, then in each block wrap every nn.Linear child of
#    block.attn and block.mlp whose attribute name is in targets with LoRALinear(child, r, alpha).
#    Return the model.
def apply_lora(model, r=8, alpha=16, targets=("qkv", "proj", "fc")):
    raise NotImplementedError


# 3. Replace every LoRALinear anywhere in the model with its merged() nn.Linear. Return the model.
def merge_lora(model):
    raise NotImplementedError


# 4. The number of trainable parameters.
def trainable_parameters(model):
    raise NotImplementedError


# 5. One SFT example (README section 2): ids = the characters of prompt + response;
#    x = ids[:-1], y = ids[1:] but with -100 for every target that is part of the prompt;
#    then pad x with pad_id and y with -100 up to block_size.
#    Raise ValueError if len(ids) - 1 > block_size. Return (x, y) as int64 tensors.
def build_sft_example(prompt, response, stoi, block_size, pad_id=0):
    raise NotImplementedError


# 6. Mean cross-entropy over the non-ignored positions: logits (B, T, V), y (B, T).
def sft_loss(logits, y):
    raise NotImplementedError


# 7. Random windows: starts from torch.randint(0, len(data) - block_size, (batch_size,),
#    generator=generator); return (x, y) with y shifted by one.
def get_batch(data, block_size, batch_size, generator):
    raise NotImplementedError


# 8. Train on a token stream: AdamW over the TRAINABLE parameters only (lr, weight_decay=0),
#    CosineAnnealingLR(T_max=steps) stepped each step, batches of length
#    model.config.block_size from a generator seeded with seed, train mode. Return the model.
def train_lm(model, data, steps, lr, batch_size=32, seed=0):
    raise NotImplementedError


# 9. Mean loss over `batches` batches of batch_size from a generator seeded with 99, in eval
#    mode without gradients; restore train mode. Return a float.
def evaluate_lm(model, data, batches=10, batch_size=64):
    raise NotImplementedError


# 10. The experiment (README section 5). results["base"] = {"english": evaluate_lm(base,
#     english_val), "code": evaluate_lm(base, code_val), "trainable": trainable_parameters(base)}.
#     Then for "lora" and "full": deep-copy base (copy.deepcopy), apply_lora for "lora" only,
#     train_lm on code_train for `steps` steps with lora_lr or full_lr and seed=1, and record
#     the same three numbers. Don't modify base. Return the results dict.
def compare_lora_and_full(base, english_val, code_train, code_val, steps=300, lora_lr=1e-2, full_lr=3e-3):
    raise NotImplementedError
