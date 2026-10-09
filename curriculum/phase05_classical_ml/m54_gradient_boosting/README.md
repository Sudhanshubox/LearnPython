# m54 · Gradient boosting from scratch

**By the end you can:** build regression trees, combine them into a gradient-boosted ensemble for regression and classification, explain *why* fitting residuals is gradient descent, use the learning rate and number of trees to control overfitting, and pick the number of trees with early stopping.

**Why it matters for AI:** gradient-boosted trees (XGBoost, LightGBM, CatBoost, scikit-learn's `HistGradientBoosting`) win the majority of competitions on tabular data and power fraud detection, credit scoring, ranking and ads systems. The core idea, "add a small model that fixes the current errors, then repeat", is also a beautiful bridge between Phase 4's gradient descent and Phase 6's neural networks.

---

## 1. Regression trees

Like m51's classification trees, but each leaf predicts the **mean** target of its samples, and splits minimize the **sum of squared errors** (SSE) instead of Gini:

SSE(node) = Σ (yᵢ − ȳ)²,   gain = SSE(parent) − SSE(left) − SSE(right)

The same sorting + cumulative-sum trick finds the best threshold fast: with running sums S = Σy and Q = Σy² on the left side, SSE(left) = Q − S²/n_left (and similarly on the right).

## 2. Boosting: fix the remaining errors

```
F₀(x) = mean(y)                        # start with a constant prediction
for m = 1..M:
    residuals rᵢ = yᵢ − F_{m−1}(xᵢ)    # what the current model gets wrong
    fit a small tree hₘ to the residuals
    Fₘ(x) = F_{m−1}(x) + η · hₘ(x)     # take a small step towards fixing them
```

Each tree is **weak** (depth 2–6), but together they form a strong model. The **learning rate** η (0.01–0.3) shrinks each step: smaller η needs more trees but generalizes better.

## 3. Why it's called *gradient* boosting

For the squared loss L = ½(y − F)², the negative gradient with respect to the prediction F is exactly the residual y − F. So fitting a tree to the residuals means fitting it to the **negative gradient**, and adding it is a gradient-descent step, taken in "function space" instead of parameter space.

That view generalizes to any differentiable loss. For **binary classification** with log-loss, the model F is a log-odds score, p = σ(F), and the negative gradient is y − p (m40's beautiful result again). So:

```
F₀ = log(p̄ / (1 − p̄))          # log-odds of the positive rate
residuals = y − σ(F)            # fit trees to these
```

(XGBoost and LightGBM also use second derivatives, a Newton step, for better leaf values. We'll keep the plain gradient version.)

## 4. Controlling overfitting

Boosting keeps reducing the *training* loss as trees are added, but the *validation* loss eventually rises. Levers:

- **learning rate** (smaller is safer) and **number of trees**,
- **tree depth** and **min samples per leaf**,
- **subsampling** rows or features per tree (stochastic gradient boosting).

**Early stopping:** track the validation loss after every tree (`staged_predict`) and keep the number of trees where it was lowest.

## 5. Bagging vs boosting

| | Random forest (m51) | Gradient boosting |
|---|---|---|
| trees | deep, independent, in parallel | shallow, sequential, each fixes the last |
| reduces mainly | variance | bias |
| overfits with more trees? | rarely | yes, without early stopping |
| tuning effort | low | moderate, usually higher accuracy |

---

## Problem-solving habit #50: try the strong baseline first

On tabular data, before designing anything fancy, run a well-tuned gradient boosting model (`HistGradientBoostingClassifier` takes one line). It's the bar any new idea must clear, including deep learning, which often doesn't on tables.

## Go deeper (optional, research-level)

1. Read *Greedy Function Approximation: A Gradient Boosting Machine* (Friedman, 2001), sections 1–4. Where does the "function space gradient descent" view come from?
2. Read the XGBoost paper (Chen & Guestrin, 2016), section 2. Derive their optimal leaf weight −G/(H + λ) using a second-order Taylor expansion of the loss.
3. Plot training and validation MSE against the number of trees for learning rates 0.5, 0.1 and 0.02. How does early stopping's best iteration change?

## Your turn

Open the **Exercises** tab. Implement with NumPy; the tests compare your booster with scikit-learn's.
