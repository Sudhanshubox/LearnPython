# m57 · Neural networks from scratch in NumPy

**By the end you can:** derive and implement backpropagation for whole matrices (not scalars), build a multi-layer network with ReLU and softmax cross-entropy, verify it with a gradient check, and train it with mini-batch SGD and momentum to over 95% accuracy on handwritten digits.

**Why it matters for AI:** m56's scalar engine creates one graph node per number, which is far too slow for real networks. Real frameworks work on **tensors**: one node per layer, with each backward step a couple of matrix multiplies. After this module, a PyTorch `nn.Linear` holds no mystery: you'll have written its forward and backward yourself. Interviewers at research labs still ask people to do exactly this on a whiteboard.

---

## 1. A layer as matrices

A batch of n examples with d features is a matrix X of shape (n, d). A **linear layer** with h outputs has weights W of shape (d, h) and bias b of shape (h,):

Z = X W + b        shape (n, h); b is broadcast over the rows (m37)

A two-layer network for k classes:

```
X ──linear(W1,b1)──▶ Z1 ──ReLU──▶ A1 ──linear(W2,b2)──▶ logits ──softmax CE──▶ loss
(n,d)               (n,h)         (n,h)                  (n,k)                  scalar
```

## 2. Backward through a linear layer

Suppose we already know G = ∂L/∂Z (shape (n, h), called `dout`). Then:

| Gradient | Formula | Shape check |
|---|---|---|
| ∂L/∂W | Xᵀ G | (d, n) @ (n, h) = (d, h) ✓ same as W |
| ∂L/∂b | G summed over rows | (h,) ✓ same as b |
| ∂L/∂X | G Wᵀ | (n, h) @ (h, d) = (n, d) ✓ same as X |

**The shape trick:** a gradient always has the same shape as the thing it's the gradient of. If you forget the formula, there is usually only one way to multiply the available matrices to get the right shape. Try it: ∂L/∂W must be (d, h), and you have X (n, d) and G (n, h). The only product that works is Xᵀ G.

The bias gradient is a sum because b was **broadcast** to every row; the backward of a broadcast is a sum over the broadcast dimension. (This rule shows up everywhere in deep learning.)

## 3. Backward through ReLU

ReLU(z) = max(0, z) elementwise, so ∂L/∂z = ∂L/∂a where z > 0, and 0 elsewhere:

```python
dz = dout * (z > 0)
```

Each layer's forward pass saves what its backward pass needs (a **cache**): the linear layer saves X and W, ReLU saves z (or the mask). That's exactly why training uses more memory than inference: all these activations must be stored until the backward pass.

## 4. Softmax cross-entropy, fused

With probabilities P = softmax(logits) and one-hot labels Y, the mean cross-entropy over the batch is

L = −(1/n) Σᵢ log P[i, yᵢ]

and its gradient with respect to the logits is wonderfully simple (m44, m50):

∂L/∂logits = (P − Y) / n

Compute softmax stably (subtract the row max) and the loss with **log-softmax** (logits − logsumexp) rather than `log(softmax(...))`, which can produce `log(0) = -inf`. Frameworks fuse softmax and cross-entropy into one function (PyTorch's `F.cross_entropy` takes raw logits) for exactly this reason.

## 5. Initialization: why not zeros, and why √(2/fan_in)

- **All zeros** makes every hidden unit compute the same thing and receive the same gradient, forever. Random init **breaks the symmetry**.
- **Too large** and activations explode layer after layer; **too small** and they shrink to nothing (you'll measure this in m60).
- **He initialization** (He et al., 2015) draws W from N(0, 2/fan_in) for ReLU layers, where fan_in is the number of inputs. The variance of each layer's outputs then stays roughly constant through the network. ReLU zeroes half of its inputs, which is where the 2 comes from.

Biases start at zero.

## 6. Weight decay (L2 regularization)

Add (λ/2) Σ ‖W‖² over the weight matrices to the loss (not the biases, as with ridge in m50). Its gradient is simply λW, added to each weight gradient. It keeps weights small and reduces overfitting.

## 7. Mini-batch SGD with momentum

Computing the gradient on all the data for every step is slow; computing it on one example is noisy. **Mini-batches** of 32–256 examples are the sweet spot, and they run fast as matrix operations.

One **epoch** is one pass over the shuffled training data:

```python
for epoch in range(epochs):
    for idx in iterate_minibatches(n, batch_size, rng):    # a fresh shuffle each epoch
        loss, grads = model.loss_and_grads(X[idx], y[idx])
        sgd_momentum_step(model.params, grads, velocity, lr, momentum)
```

**Momentum** (m41) keeps a running velocity per parameter: v ← μv − lr·g, then p ← p + v. It smooths out noise and speeds up progress along consistent directions. Update parameters **in place** (`p += v`) so the model sees the new values.

## 8. The gradient check

Before training anything, check the analytic gradients against finite differences (m40) on a tiny network and batch, in float64. A relative error below about 1e-6 means your backward pass is right; around 1e-2 means a bug. This one habit saves days of debugging. (ReLU's kink at 0 can cause occasional small mismatches; that's expected.)

---

## Problem-solving habit #57: dimensional analysis

Physicists check units; deep learning engineers check **shapes**. Write the shape of every array next to it in a comment, and assert shapes at function boundaries while developing. Most bugs in model code are shape bugs, and many are silent: broadcasting happily turns a (n,) minus (n, 1) into an (n, n) matrix (m37).

## Common mistakes

- Forgetting the 1/n in the cross-entropy gradient (your learning rate is then secretly n times larger).
- Summing the bias gradient over the wrong axis (`axis=0` sums over the batch, which is what you want).
- Not shuffling between epochs, or using the same shuffle every epoch.
- Updating parameters with `p = p - lr * g` inside a helper: that rebinds a local name and the model never changes (m07). Use `p -= lr * g` on the array itself.
- Evaluating on the training set and calling it accuracy.

## Go deeper (optional, research-level)

1. Derive (P − Y)/n yourself, starting from the softmax Jacobian you built in m40.
2. Replace ReLU with tanh and He init with N(0, 1). Plot the standard deviation of each layer's activations in a 10-layer network at initialization. What happens? Read *Understanding the difficulty of training deep feedforward neural networks* (Glorot & Bengio, 2010).
3. Your network is a **universal approximator**: with enough hidden units, a single hidden layer can approximate any continuous function. Why, then, do we use deep networks instead of one wide layer? (Search for "depth separation" results.)

## Your turn

Open the **Exercises** tab. Build the pieces in order: each later piece uses the earlier ones, and the tests check every gradient against finite differences.
