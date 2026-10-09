# m41 · Optimization

**By the end you can:** implement gradient descent, SGD with mini-batches, momentum and Adam from scratch, explain how the learning rate controls stability, use learning-rate schedules with warmup, and know when a problem has a closed-form solution instead.

**Why it matters for AI:** "training" a model *means* running an optimizer. Choosing the optimizer, learning rate and schedule is a huge part of getting models to work, and these are the exact algorithms (Adam, cosine decay with warmup) used to train today's LLMs. When a training run diverges or stalls, understanding what the optimizer is doing is how you fix it.

---

## 1. Gradient descent

To minimize a loss L(θ), repeatedly step against the gradient (m40):

θ ← θ − η ∇L(θ)

η is the **learning rate**: too small and progress crawls; too large and you overshoot, bounce around, or **diverge** to infinity.

**How large is too large?** For f(x) = ½ λ x², the gradient is λx and one step gives x ← (1 − ηλ) x. That shrinks only if |1 − ηλ| < 1, so η < 2/λ. In many dimensions the steepest direction (the largest eigenvalue of the Hessian, m39!) sets the limit: η < 2/λ_max. Meanwhile, flat directions (small λ) converge slowly. A big ratio λ_max/λ_min (**ill-conditioning**) makes plain gradient descent zig-zag.

## 2. Stochastic and mini-batch gradient descent

The true gradient averages over the whole dataset, which is expensive for millions of examples. **SGD** estimates it from a small random **mini-batch** (32–4096 examples) at each step:

```
for epoch in range(epochs):
    shuffle the data
    for each batch:                 # m28's data loader!
        θ -= lr * gradient on the batch
```

The estimate is noisy but cheap, and the noise even helps escape poor regions.

## 3. Momentum

Keep a running "velocity" that accumulates past gradients, like a ball rolling downhill:

v ← β v + ∇L(θ)    θ ← θ − η v    (β ≈ 0.9)

Momentum speeds up consistent directions and dampens zig-zags across narrow valleys.

## 4. Adam

Adam keeps two running averages per parameter: of the gradient (m, like momentum) and of the squared gradient (v, a per-parameter scale):

```
m ← β₁ m + (1 − β₁) g
v ← β₂ v + (1 − β₂) g²
m̂ = m / (1 − β₁ᵗ)        v̂ = v / (1 − β₂ᵗ)       # bias correction (t = step number, from 1)
θ ← θ − η m̂ / (√v̂ + ε)
```

Dividing by √v̂ gives every parameter its own effective step size, so steep and flat directions both make progress. The bias correction fixes the fact that m and v start at zero. Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8. **AdamW** (Adam with decoupled weight decay) is the standard optimizer for Transformers.

## 5. Learning-rate schedules

LLMs are trained with **warmup** (increase the learning rate linearly from 0 over the first steps, because early gradients are large and unreliable) followed by **cosine decay** down to a small minimum:

```
step < warmup:   lr = lr_max · step / warmup
afterwards:      progress = (step − warmup) / (total − warmup)
                 lr = lr_min + ½ (lr_max − lr_min)(1 + cos(π · progress))
```

## 6. Convexity

A function is **convex** if every chord lies above the graph (in 1-D: the second derivative is ≥ 0 everywhere). Convex problems, like linear regression with MSE or logistic regression, have no bad local minima: gradient descent finds the global minimum. Neural network losses are **not** convex, yet SGD works remarkably well on them, which is still an active research topic.

## 7. Closed-form solutions

Some problems don't need iteration. Linear least squares has the exact solution from the normal equations (m38): w = (XᵀX)⁻¹Xᵀy. Iterative methods win when the data is huge, the model isn't linear, or there's no closed form, which is almost always in deep learning.

---

## Problem-solving habit #38: sweep the learning rate on a log scale

When training doesn't work, first try learning rates spaced by factors of about 3–10 (1e-1, 3e-2, 1e-2, …) on a small run and watch the loss curves. The best value is usually just below the one where training becomes unstable. It's the single most important hyperparameter.

## Go deeper (optional, research-level)

1. Read *Adam: A Method for Stochastic Optimization* (Kingma & Ba, 2014), Algorithm 1. Then read *Decoupled Weight Decay Regularization* (Loshchilov & Hutter, 2019): what's the difference between L2 regularization and weight decay in Adam?
2. Plot the paths of GD, momentum and Adam on the Rosenbrock function (matplotlib). Why is its curved valley so hard for plain GD?
3. Read about the **edge of stability** (Cohen et al., 2021): during neural network training, the largest Hessian eigenvalue tends to hover right around 2/η. Why is that surprising given section 1?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
