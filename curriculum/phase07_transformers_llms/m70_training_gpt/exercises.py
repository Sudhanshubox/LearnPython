"""m70 exercises: pretraining a small GPT on this course's lessons.

The model itself is in minigpt.py (given).
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from minigpt import GPT, GPTConfig

CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())


# 1. The text of every lesson matching the glob pattern (relative to CURRICULUM), in sorted
#    path order, read as UTF-8 and joined with "\n\n".
def load_corpus(pattern="phase0[1-6]*/*/README.md"):
    raise NotImplementedError


# 2. Character-level encoding: itos = sorted unique characters, stoi = {char: id}, data = a
#    1-D int64 tensor of the whole text's ids, split by position into the first
#    int(len * (1 - val_fraction)) ids for training and the rest for validation.
#    Return (stoi, itos, train_data, val_data).
def encode_dataset(text, val_fraction=0.1):
    raise NotImplementedError


# 3. A batch of batch_size random windows (README section 2): start positions from
#    torch.randint(0, len(data) - block_size, (batch_size,), generator=generator).
#    Return (x, y), both (batch_size, block_size), y shifted one ahead of x. No Python loops.
def get_batch(data, block_size, batch_size, generator=None):
    raise NotImplementedError


# 4. The mean validation loss over eval_iters batches from get_batch(data, block_size,
#    batch_size, generator), in eval mode without gradients. Restore the model's previous
#    train/eval mode afterwards. Return a float.
def estimate_loss(model, data, block_size, batch_size=32, eval_iters=20, generator=None):
    raise NotImplementedError


# 5. AdamW with two parameter groups: trainable parameters with 2+ dims get weight_decay,
#    the rest 0.0; lr and betas as given.
def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    raise NotImplementedError


# 6. The learning rate at step (0-based): linear warmup to max_lr over warmup_steps
#    (max_lr * (step + 1) / warmup_steps), then cosine decay from max_lr to min_lr reaching
#    min_lr at max_steps, and min_lr for any step >= max_steps.
def lr_at(step, max_steps, warmup_steps, max_lr, min_lr):
    raise NotImplementedError


# 7. The training loop. torch.manual_seed(seed) then GPT(config); configure_optimizer(model,
#    max_lr); batches from a torch.Generator seeded with seed; evaluation batches from a
#    separate generator seeded with seed + 1. Each step: set every param group's lr to
#    lr_at(step, max_steps, warmup_steps, max_lr, max_lr / 10), forward, zero_grad, backward,
#    clip the gradient norm to grad_clip, step. Record every step's training loss, and after
#    step + 1 is a multiple of eval_interval (and after the last step), record
#    (step + 1, estimate_loss(model, val_data, config.block_size, generator=eval_generator)).
#    Return (model, {"train_loss": [...], "val_loss": [(step, loss), ...]}).
def train_gpt(config, train_data, val_data, max_steps=500, batch_size=32, max_lr=6e-3,
              warmup_steps=50, eval_interval=100, grad_clip=1.0, seed=0):
    raise NotImplementedError


# 8. Generate max_new_tokens tokens after idx (B, T) (README section 5), in eval mode without
#    gradients. Crop the context to the last block_size tokens. temperature == 0 means greedy
#    argmax; otherwise divide the logits by temperature, keep only the top_k largest if top_k
#    is given, and sample with torch.multinomial(probs, 1, generator=generator).
#    Return the extended (B, T + max_new_tokens) tensor.
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
    raise NotImplementedError
