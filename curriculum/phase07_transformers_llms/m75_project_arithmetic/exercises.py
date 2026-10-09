"""m75 project: teaching addition to a small transformer. Read the README first.

The model is in minigpt.py.
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F

from minigpt import GPT, GPTConfig

VOCAB = "0123456789+=\n"
STOI = {c: i for i, c in enumerate(VOCAB)}


# 1. One training string: zero-padded operands, the answer zero-padded to n_digits + 1
#    digits (reversed if `reverse`), and a final newline.
#    format_example(7, 120, 3, False) == "007+120=0127\n"
#    format_example(7, 120, 3, True)  == "007+120=7210\n"
def format_example(a, b, n_digits, reverse):
    raise NotImplementedError


# 2. n random problems: rng = random.Random(seed); each pair is
#    (rng.randrange(10 ** n_digits), rng.randrange(10 ** n_digits)).
def make_pairs(n, n_digits, seed):
    raise NotImplementedError


# 3. Encode a list of pairs into (x, y) int64 tensors: ids of format_example(...),
#    x = ids[:, :-1], y = ids[:, 1:] with the first 2 * n_digits + 1 targets set to -100 so
#    only the answer (and the newline) is learned. No padding needed (all strings have the
#    same length).
def encode_batch(pairs, n_digits, reverse):
    raise NotImplementedError


# 4. Greedy batched generation, in eval mode without gradients (restore the previous mode):
#    start from the prompt "aaa+bbb=" (the first 2 * n_digits + 2 characters) for every pair,
#    append the argmax token n_digits + 1 times, decode those characters, un-reverse them if
#    `reverse`, and return the list of predicted sums as ints (-1 if not all digits).
def predict(model, pairs, n_digits, reverse):
    raise NotImplementedError


# 5. (exact, digit) accuracy as floats: exact = fraction of pairs whose predicted sum is
#    right; digit = fraction of correct digits when the true and predicted sums are written
#    as zero-padded (n_digits + 1)-digit strings in normal order (a prediction of -1 gets
#    no digits right).
def evaluate(model, pairs, n_digits, reverse):
    raise NotImplementedError


# 6. Train one model (README "Setup"): torch.manual_seed(seed); GPT(GPTConfig(
#    vocab_size=len(VOCAB), block_size=16, n_layer=2, n_head=4, d_model=64));
#    AdamW(lr, weight_decay=0.0); rng = random.Random(seed); each step's batch is
#    [train_pairs[rng.randrange(len(train_pairs))] for _ in range(batch_size)]; cross-entropy
#    with ignore_index=-100. After every eval_every steps, append
#    (step + 1, exact, digit) from evaluate(model, eval_pairs, ...) to the curve.
#    Return (model, curve).
def train_adder(train_pairs, eval_pairs, n_digits, reverse, steps=1200, eval_every=300,
                lr=1e-3, batch_size=64, seed=0):
    raise NotImplementedError


# 7. The mean exact accuracy over the points of a curve.
def curve_area(curve):
    raise NotImplementedError


# 8. The number of carries when adding a and b on paper:
#    count_carries(99, 1) == 2, count_carries(128, 367) == 1, count_carries(0, 0) == 0.
def count_carries(a, b):
    raise NotImplementedError


# 9. {number of carries: exact accuracy of the model on the pairs with that many carries},
#    with keys in increasing order.
def accuracy_by_carries(model, pairs, n_digits, reverse):
    raise NotImplementedError


# 10. Write the report (README outline) to `path`. `results` looks like
#     {"plain": {"curve": [...], "by_carries": {...}}, "reversed": {...}}.
#     Include the five section headings, a Markdown table row for each format with its final
#     exact accuracy formatted with 3 decimals, the carries analysis, and your own sentences.
def write_report(results, path):
    raise NotImplementedError
