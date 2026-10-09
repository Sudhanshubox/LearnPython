"""m45 project: a recommender system with matrix factorization.

R is a (n_users, n_items) float array; masks are boolean arrays of the same shape
(True = this rating is observed / in this split). Only ever learn from entries where the
TRAIN mask is True. Vectorize everything (loops are fine only for epochs in factorize).
"""

import numpy as np


def make_ratings(n_users=200, n_items=150, rank=3, density=0.3, noise=0.1, seed=0):
    """Synthetic ratings: 3 + user bias + item bias + low-rank tastes + noise.
    Returns (R, mask): the full matrix and which entries are observed. (Already written.)"""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_users, rank))
    V = rng.normal(0, 1, (n_items, rank))
    user_bias = rng.normal(0, 0.5, n_users)
    item_bias = rng.normal(0, 0.5, n_items)
    R = 3.0 + user_bias[:, None] + item_bias[None, :] + U @ V.T + rng.normal(0, noise, (n_users, n_items))
    mask = rng.random((n_users, n_items)) < density
    return R, mask


# 1. Split the observed entries: each observed entry goes to TEST with probability
#    test_fraction (decided by np.random.default_rng(seed).random(mask.shape) < test_fraction),
#    otherwise to TRAIN. Return (train_mask, test_mask): disjoint, and together equal to mask.
def split_mask(mask, test_fraction=0.2, seed=0):
    raise NotImplementedError


# 2. Root mean squared error between R and pred over the entries where mask is True.
def rmse(R, pred, mask):
    raise NotImplementedError


# 3. Baseline predictions for EVERY cell (n_users, n_items):
#      mu  = mean of the training ratings
#      b_u = mean over each user's training ratings of (r - mu)           (0 if none)
#      b_i = mean over each item's training ratings of (r - mu - b_u)     (0 if none)
#      prediction = mu + b_u[user] + b_i[item]
def baseline_predict(R, train_mask):
    raise NotImplementedError


# 4. SVD approach: fill every cell NOT in train_mask with the baseline prediction, then
#    return the rank-k approximation of that filled matrix (np.linalg.svd).
def svd_predict(R, train_mask, k):
    raise NotImplementedError


# 5. The matrix-factorization objective and its gradients (see the README):
#      E = where(mask, U @ V.T - T, 0);  n = number of True entries in mask
#      loss = sum(E**2) / n + reg * (sum(U**2) + sum(V**2))
#      dU = 2/n * E @ V + 2*reg*U;   dV = 2/n * E.T @ U + 2*reg*V
#    Return (loss, dU, dV).
def mf_loss_and_grads(T, mask, U, V, reg):
    raise NotImplementedError


# 6. Train the factorization on the residuals T = R - baseline_predict(R, train_mask).
#    Initialize with rng = np.random.default_rng(seed): U = rng.normal(0, 0.1, (n_users, k)),
#    then V = rng.normal(0, 0.1, (n_items, k)). Run `epochs` Adam steps (from m41, with
#    beta1=0.9, beta2=0.999, eps=1e-8) on U and V using mf_loss_and_grads.
#    Return (U, V, base, losses): base is the baseline matrix and losses[t] is the loss
#    before step t. Predictions are then base + U @ V.T.
def factorize(R, train_mask, k, lr=0.01, reg=1e-4, epochs=500, seed=0):
    raise NotImplementedError


# 7. The n items with the highest predicted rating for `user`, EXCLUDING items the user
#    already rated in train_mask, best first (ties: lower item index first).
#    Returns a list of item indices.
def recommend(pred, train_mask, user, n=5):
    raise NotImplementedError


# 8. The n items most similar to `item` by cosine similarity of their rows in V
#    (excluding the item itself), most similar first. Returns a list of item indices.
def similar_items(V, item, n=5):
    raise NotImplementedError
