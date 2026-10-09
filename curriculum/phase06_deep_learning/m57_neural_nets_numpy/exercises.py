"""m57 exercises: a multi-layer network with hand-written backprop, using only NumPy.

Shapes: X is (n, d), labels y are ints in [0, k) with shape (n,).
"""

import numpy as np


# 1. He initialization: weights of shape (fan_in, fan_out) drawn with
#    rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).
def he_init(fan_in, fan_out, rng):
    raise NotImplementedError


# 2. Linear layer. forward returns (X @ W + b, cache) where cache holds what backward needs.
#    backward(dout, cache) returns (dX, dW, db). No loops.
def linear_forward(X, W, b):
    raise NotImplementedError


def linear_backward(dout, cache):
    raise NotImplementedError


# 3. ReLU. forward returns (max(0, z), cache); backward(dout, cache) returns dz.
def relu_forward(z):
    raise NotImplementedError


def relu_backward(dout, cache):
    raise NotImplementedError


# 4. Mean softmax cross-entropy over the batch, and its gradient with respect to the logits.
#    Return (loss as a float, dlogits with the same shape as logits). Be numerically stable:
#    logits of 1000 must not produce inf or nan.
def softmax_cross_entropy(logits, y):
    raise NotImplementedError


# 5. An MLP for classification. sizes = [d, h1, ..., k]: linear layers with ReLU between
#    them and no ReLU after the last one.
#    __init__: self.params is a dict {"W1", "b1", "W2", "b2", ...} with W_i = he_init(...)
#    drawn in order W1, W2, ... from ONE np.random.default_rng(seed), and b_i = zeros.
#    forward(X): return the logits, saving caches on self for backward.
#    backward(dlogits): return a dict of gradients with the same keys as self.params.
#    loss_and_grads(X, y, weight_decay=0.0): forward + softmax_cross_entropy + backward,
#      adding 0.5 * weight_decay * sum of squared weights (W's only, not b's) to the loss
#      and weight_decay * W to each W gradient. Return (loss, grads).
#    predict(X): the predicted class for each row.
class MLP:
    def __init__(self, sizes, seed=0):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dlogits):
        raise NotImplementedError

    def loss_and_grads(self, X, y, weight_decay=0.0):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


# 6. Gradient check: numerical gradients of model.loss_and_grads(X, y, weight_decay)[0]
#    with respect to EVERY entry of every parameter, using central differences with step h.
#    Change the parameter arrays in place and restore them afterwards.
#    Return a dict with the same keys and shapes as model.params. (Loops are fine here.)
def numerical_grads(model, X, y, weight_decay=0.0, h=1e-5):
    raise NotImplementedError


# 7. Yield arrays of indices covering range(n) in batches of batch_size (the last one may
#    be smaller), in the order of one rng.permutation(n).
def iterate_minibatches(n, batch_size, rng):
    raise NotImplementedError


# 8. SGD with momentum, IN PLACE: for every key, velocity[key] = momentum * velocity[key]
#    - lr * grads[key], then params[key] += velocity[key]. Create velocity[key] as zeros
#    the first time a key is missing.
def sgd_momentum_step(params, grads, velocity, lr, momentum=0.9):
    raise NotImplementedError


# 9. Train with mini-batches: rng = np.random.default_rng(seed); each epoch, loop over
#    iterate_minibatches(len(X), batch_size, rng), and update with sgd_momentum_step
#    (one velocity dict for the whole run).
#    Return a history dict: "train_loss" = list of the mean mini-batch loss per epoch, and
#    "val_acc" = list of the accuracy on (X_val, y_val) after each epoch.
def train(model, X, y, X_val, y_val, epochs=20, lr=0.05, batch_size=64, momentum=0.9,
          weight_decay=0.0, seed=0):
    raise NotImplementedError
