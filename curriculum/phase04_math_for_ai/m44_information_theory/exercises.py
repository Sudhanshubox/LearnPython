"""m44 exercises: information theory. Natural logs (nats) unless stated otherwise.

Treat 0 * log(0) as 0. No Python loops: vectorize.
"""

import numpy as np


# 1. Entropy of a probability distribution p (1-D), in bits (log base 2).
def entropy_bits(p):
    raise NotImplementedError


# 2. Cross-entropy H(p, q) = -sum p log q, and KL(p || q) = sum p log(p / q), in nats.
#    Terms where p == 0 contribute 0. Clip q to >= 1e-12 inside logs.
def cross_entropy(p, q):
    raise NotImplementedError


def kl_divergence(p, q):
    raise NotImplementedError


# 3. Numerically stable log-sum-exp along `axis` (keepdims=False), and log_softmax.
#    logsumexp(z) = m + log(sum(exp(z - m))) with m = max(z).
def logsumexp(z, axis=-1):
    raise NotImplementedError


def log_softmax(z, axis=-1):
    raise NotImplementedError


# 4. Cross-entropy loss computed directly from LOGITS, the way frameworks do it.
#    logits: (n, k), labels: (n,) ints. Return the mean of -log_softmax(logits)[i, label_i].
#    Must not overflow for logits like 1000.
def cross_entropy_from_logits(logits, labels):
    raise NotImplementedError


# 5. ...and its gradient with respect to the logits: (softmax(logits) - onehot(labels)) / n.
def cross_entropy_grad(logits, labels):
    raise NotImplementedError


# 6. Perplexity from the probabilities a language model gave to each actual next token:
#    exp(mean(-log p)). token_probs is 1-D.
def perplexity(token_probs):
    raise NotImplementedError


# 7. Knowledge distillation loss: KL(teacher || student) between softened distributions,
#    softmax(teacher_logits / T) and softmax(student_logits / T), averaged over rows,
#    then multiplied by T**2 (as in Hinton et al.). Both inputs are (n, k).
def distillation_loss(teacher_logits, student_logits, T=2.0):
    raise NotImplementedError


# 8. Mutual information (in nats) of two discrete variables from their joint
#    distribution table P (rows: values of X, columns: values of Y, sums to 1):
#    I = sum P(x, y) log(P(x, y) / (P(x) P(y))) over cells with P(x, y) > 0.
def mutual_information(P):
    raise NotImplementedError
