# m40 · Calculus: derivatives, gradients and the chain rule

**By the end you can:** compute derivatives by hand and numerically, take gradients of functions of many variables (vectors and matrices), apply the chain rule through compositions of functions, and verify any gradient with a gradient check.

**Why it matters for AI:** training a neural network means repeatedly asking "if I nudge each of these millions of weights, how does the loss change?", and moving every weight slightly downhill. That question *is* the gradient. **Backpropagation** is the chain rule applied systematically. Frameworks compute gradients automatically, but researchers derive them by hand to design new layers and losses, and every practitioner needs gradient checks to debug custom code.

---

## 1. The derivative

The derivative f′(x) is the slope of f at x: how fast f changes when x changes a tiny bit.

f′(x) ≈ (f(x + h) − f(x − h)) / 2h    (the central difference, m07)

Rules you'll use constantly:

| f(x) | f′(x) |
|---|---|
| xⁿ | n xⁿ⁻¹ |
| eˣ | eˣ |
| ln x | 1/x |
| sin x, cos x | cos x, −sin x |
| σ(x) = 1/(1 + e⁻ˣ) (sigmoid) | σ(x)(1 − σ(x)) |
| tanh x | 1 − tanh² x |
| ReLU = max(0, x) | 1 if x > 0, else 0 |

## 2. The chain rule

For a composition f(g(x)):

(f ∘ g)′(x) = f′(g(x)) · g′(x)

Example: d/dx sin(x²) = cos(x²) · 2x. "Derivative of the outside, evaluated at the inside, times the derivative of the inside." A neural network is a long composition (layer after layer), so the chain rule is how you get the gradient for every layer's weights.

## 3. Partial derivatives and the gradient

For a function of several variables, f(x₁, …, xₙ), the **partial derivative** ∂f/∂xᵢ treats every other variable as a constant. Collect them all into the **gradient**:

∇f = (∂f/∂x₁, …, ∂f/∂xₙ)

The gradient points in the direction of steepest **increase**, so −∇f points steepest downhill. Gradient descent (m41) steps that way.

Useful vector results (for symmetric A):

| f(x) | ∇f |
|---|---|
| aᵀx | a |
| ½ ‖x‖² | x |
| ½ xᵀAx + bᵀx | Ax + b |

## 4. Gradients of losses

**Mean squared error** for a linear model ŷ = Xw + b, L = (1/n) Σ (ŷᵢ − yᵢ)²:

∂L/∂w = (2/n) Xᵀ(ŷ − y)    ∂L/∂b = (2/n) Σ (ŷᵢ − yᵢ)

(m24 computed these with loops; now it's one line.)

**Binary cross-entropy** with p = σ(Xw + b), L = −(1/n) Σ [y log p + (1 − y) log(1 − p)]:

∂L/∂w = (1/n) Xᵀ(p − y)    ∂L/∂b = (1/n) Σ (pᵢ − yᵢ)

The sigmoid's derivative cancels with the log, leaving the beautifully simple "prediction minus target". The same thing happens for softmax with cross-entropy.

## 5. The Jacobian

For a function F from ℝⁿ to ℝᵐ, the **Jacobian** J is the m × n matrix of all partial derivatives, Jᵢⱼ = ∂Fᵢ/∂xⱼ. It generalizes the derivative: near x, F(x + δ) ≈ F(x) + J δ.

The softmax's Jacobian is diag(p) − p pᵀ. The chain rule for vector functions is a **product of Jacobians**, and backpropagation multiplies them from the loss backwards. That ordering ("vector–Jacobian products") is far cheaper than building full Jacobians.

## 6. Backpropagating through a layer

For y = ReLU(Wx + b) and loss L = ½‖y − t‖²:

1. z = Wx + b, y = ReLU(z)    (forward pass: remember z)
2. ∂L/∂y = y − t
3. ∂L/∂z = ∂L/∂y ⊙ [z > 0]    (ReLU passes gradient only where it was active)
4. ∂L/∂W = (∂L/∂z) xᵀ    ∂L/∂b = ∂L/∂z    (and ∂L/∂x = Wᵀ ∂L/∂z for the layer below)

That's backpropagation in four lines, the heart of Phase 6.

## 7. Gradient checking

Never trust a hand-derived gradient until it's checked. Compare it with the numerical gradient using the **relative error**:

rel_err = ‖g_analytic − g_numeric‖ / (‖g_analytic‖ + ‖g_numeric‖)

Below about 1e-7 is excellent; around 1e-4 deserves suspicion; above 1e-2 is almost certainly a bug. Use float64 and a smooth test point (avoid exact ReLU kinks).

---

## Problem-solving habit #37: check every gradient numerically

Whenever you derive or implement a gradient, immediately compare it with a central-difference estimate on a small random input. It takes two minutes and catches the sign errors, missing factors of 2 and transposed matrices that otherwise cost days of mysteriously bad training.

## Go deeper (optional, research-level)

1. Watch 3Blue1Brown's *Essence of Calculus* chapters 1–4 and *Neural networks* chapter 4 ("Backpropagation calculus").
2. Read *Calculus on Computational Graphs: Backpropagation* by Christopher Olah. Why is reverse-mode differentiation (backprop) so much cheaper than forward mode when there's one loss and millions of parameters?
3. Derive the gradient of softmax + cross-entropy with respect to the logits, and show it's p − onehot(y). Then check it numerically with your `numerical_gradient`.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Every analytic gradient you write is checked against numerical gradients.
