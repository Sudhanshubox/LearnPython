"""m63 exercises: tools for debugging training, and a bug hunt."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. The expected cross-entropy at initialization for n_classes classes (uniform predictions).
def expected_initial_loss(n_classes):
    raise NotImplementedError


# 2. Train the model on ONE batch (xb, yb) for `steps` steps with Adam(lr) and cross-entropy,
#    in train mode. Return the list of loss values (floats), one per step.
def overfit_single_batch(model, xb, yb, steps=200, lr=1e-2):
    raise NotImplementedError


# 3. Run forward + cross-entropy + backward once on (xb, yb) (zero the old gradients first),
#    then return {parameter name: gradient norm as a float} for every parameter that
#    requires grad. Use None for a parameter whose .grad is None after backward.
def grad_norms_by_layer(model, xb, yb):
    raise NotImplementedError


# 4. The update-to-data ratio of every parameter with a gradient:
#    {name: lr * grad.norm() / param.norm()} as floats (README section 3).
def update_to_data_ratios(model, lr):
    raise NotImplementedError


# 5. For every nn.ReLU module in the model (in model.named_modules() order), the fraction of
#    its output units that are zero for EVERY example in the batch x. Return a list of floats.
#    Use forward hooks (remove them afterwards) and no gradients. For an output of shape
#    (N, ...), a "unit" is one position in (...).
def dead_relu_fractions(model, x):
    raise NotImplementedError


# 6. Names of the parameters that contain a non-finite value (nan or inf) in either the
#    parameter itself or its .grad, in named_parameters() order.
def find_nonfinite(model):
    raise NotImplementedError


# 7. Register a forward hook on every submodule (every module in model.modules() except the
#    model itself) that raises RuntimeError(f"non-finite output in {name}") when the module's
#    output tensor contains nan or inf, where name is the module's name from named_modules().
#    Return the list of hook handles (so the caller can remove them).
def add_nan_checks(model):
    raise NotImplementedError


# 8. BUG HUNT. This function trains a classifier, runs without errors... and has FIVE silent
#    bugs (see README section 6). Fix them all without changing the signature.
#    Intended behaviour: each epoch, shuffle with torch.randperm(len(X), generator=g), loop
#    over mini-batches in TRAIN mode, compute cross-entropy on the raw logits, take an
#    optimizer step with fresh gradients, and record each epoch's mean batch loss as a float.
def train_classifier(model, optimizer, X, y, epochs=10, batch_size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        X_shuffled = X[perm]
        y_shuffled = y[torch.randperm(len(y), generator=g)]
        losses = []
        for start in range(0, len(X), batch_size):
            xb = X_shuffled[start:start + batch_size]
            yb = y_shuffled[start:start + batch_size]
            logits = model(xb)
            loss = F.cross_entropy(torch.softmax(logits, dim=1), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss)
        history.append(sum(losses) / len(losses))
    return history
