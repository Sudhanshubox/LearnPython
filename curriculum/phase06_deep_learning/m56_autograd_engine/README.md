# m56 · Build an autograd engine from scratch

**By the end you can:** explain exactly what happens when you call `loss.backward()` in PyTorch, because you will have built the same machinery yourself: a `Value` class that records every operation into a computation graph, a topological sort, and reverse-mode automatic differentiation. Then you'll build neurons, layers and a multi-layer perceptron on top of it and train it.

**Why it matters for AI:** every deep learning framework (PyTorch, JAX, TensorFlow) is, at its core, this module plus fast tensors and GPUs. Researchers who understand autograd can write custom layers, debug "gradient is None" errors in seconds, and reason about memory use during training. This module is inspired by Andrej Karpathy's *micrograd*; watching his video "The spelled-out intro to neural networks and backpropagation" after you finish is a great review.

---

## 1. The idea: record the computation, then walk it backwards

In m40 you derived gradients by hand. That doesn't scale to a network with millions of parameters. Instead, we make numbers that **remember how they were made**:

```python
a = Value(2.0)
b = Value(-3.0)
c = a * b          # c remembers: "I am a product of a and b"
d = c + 1.0        # d remembers: "I am a sum of c and 1.0"
d.backward()       # fills in a.grad, b.grad, c.grad, d.grad
a.grad             # ∂d/∂a = b = -3.0
```

Each operation creates a new `Value` holding:

- `data`: the number itself (computed in the **forward pass**),
- `grad`: ∂(final output)/∂(this value), filled in by the **backward pass**, starting at 0,
- `_prev`: the values it was computed from (its children in the graph),
- `_backward`: a small function that knows how to push this node's gradient to its children.

## 2. Local derivatives + the chain rule

Every operation only needs to know its **local** derivative. If `out = a * b`, then ∂out/∂a = b and ∂out/∂b = a. By the chain rule (m40), if the final loss L depends on `out`:

∂L/∂a = ∂L/∂out · ∂out/∂a = `out.grad * b.data`

So the multiplication node's `_backward` is:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

The `_backward` closure captures `self`, `other` and `out`, so it can run later, after `out.grad` is known.

| Operation | Local derivatives |
|---|---|
| `a + b` | 1, 1 |
| `a * b` | b, a |
| `a ** n` (n a number) | n · aⁿ⁻¹ |
| `tanh(a)` | 1 − tanh(a)² |
| `relu(a)` | 1 if a > 0 else 0 |
| `exp(a)` | exp(a) |
| `log(a)` | 1/a |

Everything else can be built from these: `a - b` is `a + (-b)`, `-a` is `a * -1`, and `a / b` is `a * b**-1`. Fewer primitives means fewer places for bugs.

## 3. Why `+=` and not `=`: gradient accumulation

If a value is used **twice**, gradients from both uses must add up (the multivariable chain rule sums over all paths):

```python
a = Value(3.0)
b = a + a        # ∂b/∂a = 2
b.backward()
a.grad           # must be 2.0. With "=" instead of "+=" you'd get 1.0
```

This is the single most common bug in hand-written autograd, and it's also why PyTorch **accumulates** gradients across `backward()` calls, so you must zero them before each training step (`optimizer.zero_grad()`).

## 4. The backward pass: topological order

A node's `_backward` may only run once its own `grad` is complete, meaning after every node that *uses* it has pushed its gradient. A **topological order** lists every node after all of its children (m20). Reverse it, and every node comes before its children:

```python
def backward(self):
    order = topological_order(self)    # children before parents
    self.grad = 1.0                    # ∂L/∂L = 1
    for node in reversed(order):
        node._backward()
```

micrograd builds the order with a recursive DFS. That crashes with `RecursionError` on long chains (an RNN unrolled over 5,000 steps, say), because Python's recursion limit is about 1,000. You'll write it **iteratively** with an explicit stack, exactly like m20's iterative DFS: push `(node, children_done)` pairs, and append a node to the order only when you see it the second time.

## 5. Making Values feel like numbers

Python's data model (m26) lets `Value` work naturally with plain numbers:

- `__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__` handle `2 + v`, `2 * v`, `2 - v`, `2 / v`. Python calls them when the left operand is a number that doesn't know about `Value`.
- With `__radd__`, the built-in `sum(values)` works, because it starts from `0 + values[0]`.

## 6. From Values to a neural network

A **neuron** computes a weighted sum of its inputs plus a bias, then applies a nonlinearity:

out = tanh(w₁x₁ + w₂x₂ + … + wₙxₙ + b)

A **layer** is a list of neurons that all see the same inputs. A **multi-layer perceptron (MLP)** is a list of layers, each feeding the next. The final layer is usually **linear** (no tanh) so it can output any number, for example a logit.

Training is the loop you wrote in m41, now with automatic gradients:

```python
for step in range(steps):
    loss = compute_loss(model, X, y)    # forward: builds a fresh graph
    zero_grad(model.parameters())       # forget last step's gradients
    loss.backward()                     # backward: fills every p.grad
    for p in model.parameters():
        p.data -= lr * p.grad           # gradient descent
```

## 7. Cross-entropy on Values

For classification, the model outputs one logit per class. The loss is the negative log of the softmax probability of the correct class (m44):

loss = −log softmax(z)ₜ = logsumexp(z) − zₜ

For numerical stability, subtract the maximum logit first: logsumexp(z) = m + log Σ exp(zᵢ − m). Here m should be a plain **float** (the max of the `.data` values), not a Value: shifting all logits by a constant doesn't change the loss, so no gradient needs to flow through it.

## 8. Checking your gradients

Never trust a hand-written backward pass until it matches **finite differences** (m40): nudge one input by ±h, re-run the forward pass, and compare (f(x+h) − f(x−h)) / 2h with the gradient your engine computed. The tests do this for every operation. PyTorch has the same tool: `torch.autograd.gradcheck`.

---

## Problem-solving habit #56: build the tiny version first

When a system feels like magic (autograd, a database, a compiler, a transformer), build a 100-line version of it. You'll understand the real one far better, and you'll know which parts are essential and which are optimizations. Researchers do this constantly: a toy version is the fastest way to test an idea.

## Common mistakes

- `self.grad = ...` instead of `self.grad += ...` in a `_backward` (breaks reused values).
- Forgetting to wrap plain numbers: `Value(2.0) * 3` must work.
- Forgetting `zero_grad` in the training loop: gradients pile up and training explodes.
- Calling the `_backward` functions in the wrong order (a node before all its users have finished).

## Go deeper (optional, research-level)

1. This engine is **reverse mode** autodiff: one backward pass gives the gradient with respect to *all* inputs. **Forward mode** (dual numbers) gives the derivative of *all* outputs with respect to *one* input. Implement forward mode with a `Dual(value, derivative)` class. When is each mode cheaper? (Hint: count inputs vs outputs. A loss has one output and millions of inputs.)
2. Our engine works on scalars, so a 1,000×1,000 matrix multiply creates a billion nodes. PyTorch's nodes are whole **tensors**. Sketch the `_backward` for `C = A @ B` on matrices. (You'll write it for real in m57.)
3. Read the PyTorch paper (*PyTorch: An Imperative Style, High-Performance Deep Learning Library*, Paszke et al., 2019), section 4 on autograd. Which design choices do you recognize from this module?

## Your turn

Open the **Exercises** tab. Implement the `Value` operations first (the tests check each one with finite differences), then `topological_order` and `backward`, then the network.
