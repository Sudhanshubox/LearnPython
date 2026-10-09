# m51 · Decision trees and random forests from scratch

**By the end you can:** measure impurity with Gini and entropy, find the best split efficiently, grow a decision tree recursively with stopping rules, compute feature importances, and combine many randomized trees into a random forest that generalizes better than any single tree.

**Why it matters for AI:** tree ensembles (random forests and gradient boosting, m54) are still the strongest out-of-the-box models for tabular data, the kind most businesses have. In m18 you *ran* a decision tree; now you'll *learn* one from data. Random forests also teach one of ML's deepest ideas: averaging many noisy, diverse models reduces variance. The same idea is behind ensembles, dropout and model averaging in deep learning.

---

## 1. Impurity: how mixed is a node?

For a node containing labels with class proportions p₁…pₖ:

- **Gini impurity:** 1 − Σ pᵢ². It's 0 for a pure node and largest when classes are evenly mixed (0.5 for two balanced classes).
- **Entropy:** −Σ pᵢ log₂ pᵢ (m44). 0 for pure, 1 bit for a balanced binary node.

Both work well in practice; Gini is slightly cheaper (no logarithm).

## 2. Choosing a split

A split asks "is feature j ≤ threshold t?" and sends samples left or right. Its quality is the **impurity decrease (gain)**:

gain = impurity(parent) − (n_left/n) · impurity(left) − (n_right/n) · impurity(right)

Search all features, and for each feature all **candidate thresholds**: the midpoints between consecutive *distinct* sorted values.

**Doing it fast.** Recomputing the class counts of both sides for every threshold is O(n²) per feature. Instead, sort the feature once, and sweep left to right: the left side's class counts are a **cumulative sum** of one-hot labels (m37's vectorization!), and the right side's are total − left. That gives every threshold's gain in one vectorized O(n·k) pass after an O(n log n) sort.

## 3. Growing the tree

```
grow(X, y, depth):
    if depth == max_depth or len(y) < min_samples_split or the node is pure:
        return a leaf predicting the majority class
    find the best split; if no split has positive gain, return a leaf
    return a node(feature, threshold, left=grow(left part), right=grow(right part))
```

This is m18's recursion again: each subtree is built from its own subset of the data.

## 4. Overfitting and stopping

A tree grown until every leaf is pure memorizes the training data: 100% training accuracy, much worse test accuracy. Limits like `max_depth`, `min_samples_split` and `min_samples_leaf` (or pruning afterwards) trade a little training accuracy for better generalization. This is the **bias–variance trade-off**: deep trees have low bias and high variance.

## 5. Feature importance

Add up, for each feature, the impurity decrease of every split that used it, weighted by the fraction of samples reaching that node. Normalize to sum to 1. It's quick and useful, but biased towards features with many distinct values, so cross-check with permutation importance (shuffle one feature in the test set and measure how much the score drops).

## 6. Random forests

A single deep tree has high variance: change the data a little and the tree changes a lot. **Averaging** many trees reduces variance, but only if they make *different* mistakes. Random forests make trees diverse in two ways:

1. **Bagging:** each tree trains on a **bootstrap sample** (m43): n rows drawn with replacement.
2. **Random features:** at each split, only a random subset of features is considered (typically √d).

Prediction is a **majority vote**. Forests rarely overfit as you add trees; more trees just cost more time.

---

## Problem-solving habit #47: compare against simpler models

Before celebrating a complex model, compare it with a depth-1 tree ("decision stump"), a shallow tree, and logistic regression. If the fancy model barely beats them, the extra complexity might not be worth it, and the simple model is easier to explain and debug.

## Go deeper (optional, research-level)

1. Read section 1–3 of Leo Breiman's *Random Forests* (2001). How does he relate a forest's error to the strength of individual trees and the correlation between them?
2. Implement permutation importance and compare it with your impurity-based importance on a dataset where you add a random noise column with many distinct values. Which method is fooled?
3. Fit your tree on perfectly balanced XOR data (the four corners of a square, equally many of each). Why can't it make even one split? Why does scikit-learn's tree still manage? Greedy algorithms (m22) can miss splits that only pay off one level later.
4. Why do trees need no feature scaling, while logistic regression does (m50)? What else are trees invariant to?

## Your turn

Open the **Exercises** tab. Implement everything with NumPy; the tests compare your tree with scikit-learn's.
