"""m65 project: reproduce the ResNet degradation result. Read the README first."""

import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# 1. Load the digits, scale pixels to [0, 1] (divide by 16), split with
#    train_test_split(..., test_size=0.25, random_state=0, stratify=y), and return
#    (X_train, y_train, X_val, y_val) with X as float32 tensors of shape (N, 1, 8, 8) and
#    y as int64 tensors.
def load_data():
    raise NotImplementedError


# 2. A block with two 3x3 convolutions (padding 1, bias=False, `channels` in and out), each
#    followed by BatchNorm2d: self.conv1, self.bn1, self.conv2, self.bn2.
#    forward: out = relu(bn1(conv1(x))); out = bn2(conv2(out)); add x if self.residual;
#    then apply a final relu.
class BasicBlock(nn.Module):
    def __init__(self, channels, residual=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 3. self.stem = nn.Sequential(Conv2d(1, channels, 3, padding=1, bias=False),
#    BatchNorm2d(channels), ReLU()); self.blocks = nn.Sequential of n_blocks BasicBlocks;
#    self.head = nn.Linear(channels, n_classes).
#    forward: stem -> blocks -> mean over the two spatial dims (global average pooling) -> head.
class DeepNet(nn.Module):
    def __init__(self, n_blocks, residual, channels=16, n_classes=10):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 4. Train the model (README "Your setup"): SGD(lr, momentum=0.9, weight_decay=1e-4) and
#    CosineAnnealingLR(optimizer, T_max=epochs * number_of_batches) stepped after every batch.
#    Each epoch: shuffle with torch.randperm(n, generator=g) where g is a torch.Generator
#    seeded with `seed`, batches of batch_size, train mode. After each epoch, evaluate the
#    validation accuracy in eval mode without gradients.
#    Return {"train_loss": [mean training loss per epoch, weighted by batch size],
#            "val_acc": [validation accuracy per epoch]} as lists of floats.
def train(model, data, epochs=8, lr=0.05, batch_size=64, seed=0):
    raise NotImplementedError


# 5. For every n_blocks in depths, every residual in (False, True), and every seed in seeds
#    (in that nesting order), call torch.manual_seed(seed), build DeepNet(n_blocks, residual),
#    train it with train(model, data, epochs=epochs, seed=seed), and append a row:
#    {"n_blocks": ..., "residual": ..., "seed": ..., "final_train_loss": ..., "final_val_acc": ...}.
#    Return the list of rows.
def run_experiment(data, depths=(2, 12), seeds=(0, 1, 2), epochs=8):
    raise NotImplementedError


# 6. Aggregate rows into {(n_blocks, residual): {"train_loss_mean", "train_loss_std",
#    "val_acc_mean", "n"}}. Use statistics.mean and statistics.stdev (0.0 when n == 1).
def summarize(rows):
    raise NotImplementedError


# 7. Gradient norms at initialization: put the model in train mode, zero its gradients,
#    backpropagate cross-entropy on (xb, yb) once, and return the gradient norm of the weight
#    of every nn.Conv2d in model.modules() order (first layer first), as floats.
def layer_grad_norms(model, xb, yb):
    raise NotImplementedError


# 8. Ablation: in every RESIDUAL BasicBlock of the model, set bn2.weight to zero (so each
#    block initially computes relu(x)). Leave plain blocks alone. Return the model.
def zero_init_residual(model):
    raise NotImplementedError


# 9. Write the report (README outline) as Markdown to `path`, with all five section headings,
#    a Markdown table of the summary with one row per configuration (include the number of
#    conv layers, 2 * n_blocks + 1, and the train loss mean formatted with 3 decimals), and
#    your own sentences in each section.
def write_report(summary, path):
    raise NotImplementedError
