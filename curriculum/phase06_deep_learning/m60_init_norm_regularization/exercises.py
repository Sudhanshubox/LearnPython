"""m60 exercises: initialization, normalization and regularization, from scratch."""

import torch
import torch.nn as nn


def deep_mlp(depth, width, activation=nn.ReLU):
    """Given: depth blocks of (Linear(width, width), activation()) in an nn.Sequential."""
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), activation()]
    return nn.Sequential(*layers)


# 1. Re-initialize every nn.Linear in the model (use model.modules()) and return the model:
#    "normal1": weights N(0, 1); "small": N(0, 0.01**2); "xavier": nn.init.xavier_normal_;
#    "kaiming": nn.init.kaiming_normal_ with nonlinearity="relu".
#    Biases are set to zero in every scheme. Raise ValueError for an unknown scheme.
def init_weights(model, scheme):
    raise NotImplementedError


# 2. Run the model on x (without recording gradients) and return a list with the std of the
#    OUTPUT of every nn.ReLU and nn.Tanh module, in forward order. Use forward hooks, and
#    remove them afterwards (even if the forward pass raises).
def activation_stds(model, x):
    raise NotImplementedError


# 3. Batch normalization for x of shape (N, D) (README section 2). In training: normalize with
#    the batch mean and biased variance, and update running_mean / running_var IN PLACE
#    (using the unbiased variance for running_var) without recording gradients for that
#    update. In evaluation: normalize with the running statistics and change nothing.
def batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    raise NotImplementedError


# 4. A BatchNorm1d module: parameters "weight" (ones) and "bias" (zeros), buffers
#    "running_mean" (zeros) and "running_var" (ones), using batchnorm_forward with
#    self.training.
class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 5. Layer normalization over the last dimension of x (any number of leading dims).
def layernorm(x, gamma, beta, eps=1e-5):
    raise NotImplementedError


# 6. Inverted dropout. In training (and p > 0): keep each element where
#    torch.rand(x.shape, generator=generator) >= p and scale the kept ones by 1 / (1 - p).
#    Otherwise return x unchanged.
def dropout(x, p, training, generator=None):
    raise NotImplementedError


# 7. Two AdamW parameter groups: {"params": [...], "weight_decay": weight_decay} with every
#    trainable parameter of 2+ dimensions, then {"params": [...], "weight_decay": 0.0} with
#    every other trainable parameter. Keep the order of model.parameters().
def weight_decay_param_groups(model, weight_decay):
    raise NotImplementedError


# 8. Mean cross-entropy against label-smoothed targets (README section 6). Use log_softmax.
def label_smoothing_cross_entropy(logits, y, eps):
    raise NotImplementedError
