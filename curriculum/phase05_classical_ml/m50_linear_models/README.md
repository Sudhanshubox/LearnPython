# m50 · Linear and logistic regression from scratch

**By the end you can:** implement linear regression, ridge regression, logistic regression and softmax (multinomial) regression as reusable estimator classes with scikit-learn's `fit` / `predict` interface, explain regularization, and get results that match scikit-learn on real datasets.

**Why it matters for AI:** these models are the workhorses of tabular ML and strong baselines for everything else. They are also exactly **a neural network's last layer**: softmax regression on top of learned features is how every classifier, including an LLM predicting the next token, produces its output. Building them from scratch ties together Phase 4's linear algebra, gradients, optimization and information theory.

---

## 1. The estimator interface

scikit-learn models all look the same, which is why they're interchangeable (m25's "design the interface"):

```python
model = SomeModel(hyperparameters...)    # settings only, no data
model.fit(X_train, y_train)              # learn; stores learned values in attributes ending in _
model.predict(X_test)                    # use
model.score(X_test, y_test)              # a default metric (R² or accuracy)
```

Learned attributes end with an underscore (`coef_`, `intercept_`) so they're easy to tell apart from hyperparameters. `fit` returns `self`, so you can write `Model().fit(X, y).predict(X2)`.

## 2. Linear regression

ŷ = Xw + b, minimizing mean squared error. Closed form via least squares (m38, m41): add a column of ones for the intercept and use `np.linalg.lstsq`.

## 3. Ridge regression: L2 regularization

With many features (or correlated ones), least squares can overfit: huge weights that cancel each other out. **Ridge** adds a penalty on the weights' size:

minimize ‖Xw + b − y‖² + α‖w‖²

Closed form, with the intercept not penalized (center X and y first):

w = (XᶜᵀXᶜ + αI)⁻¹ Xᶜᵀyᶜ,   b = ȳ − x̄ · w

Larger α → smaller weights → simpler model (more bias, less variance). α is chosen on validation data. (L1 regularization, **lasso**, pushes many weights to exactly zero: automatic feature selection.)

## 4. Logistic regression

For binary labels, predict a probability with the sigmoid: p = σ(Xw + b). Minimize binary cross-entropy (the NLL, m43). There's no closed form, so use gradient descent with the gradient from m40:

∂L/∂w = Xᵀ(p − y)/n + λw    ∂L/∂b = mean(p − y)

The loss is **convex** (m41), so gradient descent reaches the global optimum. **Standardize features first** (mean 0, std 1); otherwise features on large scales dominate and gradient descent crawls.

## 5. Softmax regression (multinomial logistic regression)

For k classes, learn a weight matrix W (d × k) and bias b (k), compute logits Z = XW + b, and probabilities P = softmax(Z). The cross-entropy gradient (m44) has the same beautiful form:

∂L/∂W = Xᵀ(P − Y)/n + λW    ∂L/∂b = mean over rows of (P − Y)

where Y is the one-hot label matrix (m36). This is the final layer of every neural classifier.

## 6. Standardization: fit on train only

```python
scaler = StandardScaler().fit(X_train)       # learn mean and std from TRAINING data
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)          # apply the SAME transformation
```

Fitting the scaler on all data before splitting is leakage (m49). You'll write your own `StandardScaler` here and use scikit-learn's pipelines in m53.

## 7. Interpreting linear models

With standardized features, the size of a weight shows how strongly a feature pushes the prediction, and its sign shows the direction. That's a big advantage over black-box models, but correlated features can share or swap weight, so interpret with care.

---

## Problem-solving habit #46: match a trusted implementation

When you implement an algorithm yourself, compare it against a trusted library on the same data. If your answers differ, either you have a bug or you've misunderstood a detail (like which terms are regularized). Both are worth finding. The tests here do exactly that with scikit-learn.

## Go deeper (optional, research-level)

1. Derive the ridge closed form by setting the gradient of the penalized loss to zero. Why does adding αI always make the matrix invertible?
2. Plot the ridge coefficients as α goes from 1e-3 to 1e3 (a "regularization path"). Then do the same with scikit-learn's `Lasso`. What's different?
3. Logistic regression on perfectly separable data has no finite optimum without regularization: the weights grow forever. Why? Try it with λ = 0 and watch ‖w‖ during training.

## Your turn

Open the **Exercises** tab. Use only NumPy inside your classes; the tests compare them with scikit-learn's.
