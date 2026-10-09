"""m58 exercises: tensors, autograd and nn.Module.

Use torch operations only: no NumPy inside your functions and no Python loops over elements.
"""

import math

import torch
import torch.nn as nn


# 1. The best available device as a string: "cuda" if available, else "mps" if
#    torch.backends.mps.is_available(), else "cpu".
def get_device():
    raise NotImplementedError


# 2. Convert a NumPy array of features to a float32 tensor and an array of labels to an
#    int64 tensor. The returned tensors must NOT share memory with the NumPy inputs.
def to_tensors(X_np, y_np):
    raise NotImplementedError


# 3. Squared Euclidean distances between every row of A (n, d) and every row of B (m, d):
#    an (n, m) tensor. Use broadcasting or the expansion |a|^2 - 2 a.b + |b|^2. No loops.
def pairwise_sq_distances(A, B):
    raise NotImplementedError


# 4. Scaled dot-product scores for batches: Q is (batch, T, d), K is (batch, S, d).
#    Return Q @ K^T / sqrt(d), shape (batch, T, S). (A preview of attention, Phase 7.)
def attention_scores(Q, K):
    raise NotImplementedError


# 5. The gradient of a scalar function f at the tensor x, using autograd.
#    Don't modify x (its requires_grad and grad must be unchanged afterwards): work on
#    x.detach().clone() with requires_grad enabled. Return a tensor shaped like x.
def grad_of(f, x):
    raise NotImplementedError


# 6. Fit y ~ X @ w + b by gradient descent on the mean squared error, using autograd for
#    the gradients. Start from w = zeros(d), b = zeros(()) (both with requires_grad=True).
#    Each step: loss, backward, update under torch.no_grad(), zero the grads.
#    Return (w, b) detached.
def fit_linear_autograd(X, y, steps=200, lr=0.1):
    raise NotImplementedError


# 7. A linear layer from scratch: weight is an nn.Parameter of shape (out_features,
#    in_features) and bias an nn.Parameter of shape (out_features,), both initialized
#    uniformly in [-1/sqrt(in_features), 1/sqrt(in_features)] (use tensor.uniform_ under
#    torch.no_grad(), or nn.init.uniform_). forward(x) returns x @ weight.T + bias.
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 8. An MLP with registered layers: sizes [d, h1, ..., k] gives nn.Linear layers stored in
#    an nn.ModuleList called self.layers, with ReLU between them (none after the last).
class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 9. The number of TRAINABLE parameters (individual numbers, not tensors) in a model.
def count_parameters(model):
    raise NotImplementedError


# 10. Freeze every parameter of the model except those whose name (from named_parameters)
#     starts with one of the given prefixes. Return the model.
def freeze_except(model, prefixes):
    raise NotImplementedError


# 11. The straight-through estimator: forward rounds to the nearest integer, backward
#     passes the incoming gradient through unchanged.
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


# 12. A clipped ReLU whose backward you write yourself: forward is min(max(x, 0), cap),
#     backward passes the gradient only where 0 < x < cap. Save what you need with
#     ctx.save_for_backward; store the plain number cap on ctx directly. backward must
#     return one gradient per forward input: (grad for x, None for cap).
class ClippedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cap):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
