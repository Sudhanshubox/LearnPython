"""m40 exercises: derivatives, gradients and the chain rule.

All arrays are float64 NumPy arrays.
"""

import numpy as np


# 1. Numerical gradient of a scalar function f at a point x (any shape) using central
#    differences with step h, one coordinate at a time. Return an array shaped like x.
#    Don't modify x itself (work on a copy). Loops are fine here.
def numerical_gradient(f, x, h=1e-5):
    raise NotImplementedError


# 2. The chain rule as a higher-order function: given f_prime, g and g_prime, return a
#    function computing the derivative of f(g(x)).
#    chain_rule(np.cos, lambda x: x**2, lambda x: 2*x)(x) == cos(x**2) * 2x
def chain_rule(f_prime, g, g_prime):
    raise NotImplementedError


# 3. Activation functions and their derivatives (elementwise, no loops).
#    sigmoid_grad uses sigmoid(x) * (1 - sigmoid(x)); tanh_grad uses 1 - tanh(x)**2;
#    relu_grad is 1.0 where x > 0 and 0.0 elsewhere.
def sigmoid(x):
    raise NotImplementedError


def sigmoid_grad(x):
    raise NotImplementedError


def tanh_grad(x):
    raise NotImplementedError


def relu_grad(x):
    raise NotImplementedError


# 4. f(x) = 0.5 * x @ A @ x + b @ x for a symmetric matrix A. Return (f(x), gradient).
def quadratic(A, b, x):
    raise NotImplementedError


# 5. Mean squared error of a linear model y_hat = X @ w + b, and its gradients.
#    Return (loss, dw, db). No loops.
def mse_loss_and_grads(w, b, X, y):
    raise NotImplementedError


# 6. Binary cross-entropy of a logistic model p = sigmoid(X @ w + b), and its gradients.
#    Clip p to [1e-12, 1 - 1e-12] inside the logs. Return (loss, dw, db). No loops.
def bce_loss_and_grads(w, b, X, y):
    raise NotImplementedError


# 7. Numerical Jacobian of F: R^n -> R^m at x (1-D), shape (m, n), central differences.
def numerical_jacobian(F, x, h=1e-5):
    raise NotImplementedError


# 8. The Jacobian of softmax at z (1-D): diag(p) - outer(p, p), where p = softmax(z).
def softmax_jacobian(z):
    raise NotImplementedError


# 9. Relative error between two gradients: ||a - b|| / (||a|| + ||b||), and 0.0 if both are zero.
#    Then gradient_check(f, grad_f, x) returns the relative error between grad_f(x) and
#    numerical_gradient(f, x).
def relative_error(a, b):
    raise NotImplementedError


def gradient_check(f, grad_f, x):
    raise NotImplementedError


# 10. Backpropagation through one layer: y = relu(W @ x + b), loss = 0.5 * ||y - t||^2.
#     W: (m, n), x: (n,), b: (m,), t: (m,). Return (loss, dW, db, dx).
def layer_backward(W, b, x, t):
    raise NotImplementedError
