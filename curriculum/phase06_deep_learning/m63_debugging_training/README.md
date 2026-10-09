# m63 · Debugging neural network training

**By the end you can:** systematically find out why a network isn't learning: check the initial loss, overfit a single batch, inspect per-layer gradient norms and update ratios, detect dead ReLUs, catch NaNs at the exact layer that produces them, and find the classic silent bugs in a training loop.

**Why it matters for AI:** neural network bugs rarely crash. The code runs, the loss goes down a bit, and the model is just quietly worse than it should be. Andrej Karpathy put it this way: "neural net training fails silently". The engineers and researchers who are productive in deep learning aren't the ones who never write bugs; they're the ones with a fast, systematic process for finding them. This module is that process.

---

## 1. Check the loss at initialization

For a k-class classifier with a reasonable initialization, the predicted distribution starts out roughly uniform, so the cross-entropy should start near **ln(k)** (2.30 for 10 classes, 10.8 for a 50,000-token vocabulary). If your first loss is 30, the logits are far too large: the last layer's initialization is off and the model will waste its first steps undoing that. Fixing it is often as easy as scaling down the final layer's weights.

## 2. Overfit a single batch

Take one small batch (say 32 examples) and train on it alone, for a few hundred steps, without regularization. A correct model with enough capacity must drive the loss to **near zero** and reach 100% accuracy on that batch. If it can't:

- the gradients aren't reaching the parameters (a `detach()`, a `no_grad()`, a frozen layer, a typo'd parameter list), or
- the loss is wrong (wrong target alignment, double softmax), or
- the learning rate is far off.

This single test catches most bugs in under a minute. Do it every time you write a new model.

## 3. Look at the gradients

After `loss.backward()`, every parameter has a `.grad`. Print the **norm per layer**:

- `None` means the parameter isn't connected to the loss at all.
- All zeros means something upstream is blocking the gradient (dead ReLUs, a saturated sigmoid, an `argmax` in the graph).
- Norms that shrink sharply toward the input layers indicate vanishing gradients (m62); huge values indicate exploding ones.

Another useful number is the **update-to-data ratio**: how much a step changes each parameter relative to its size, (lr · ‖grad‖) / ‖param‖. As a rule of thumb (Karpathy), around **1e-3** is healthy: much lower and the layer is barely learning, much higher and it's thrashing.

## 4. Dead ReLUs

A ReLU unit that outputs 0 for *every* input in a batch passes no gradient, and if it stays that way, it never recovers. A large fraction of dead units (after a too-high learning rate, say) wastes capacity. Measure it with a forward hook: for each ReLU, the fraction of units that are zero for all examples.

## 5. NaNs and infinities

A single `inf` or `nan` spreads through the whole network in one step. To find where it starts:

- Check the parameters and gradients with `torch.isfinite(t).all()`.
- Register **forward hooks** on every module that raise an error naming the first module whose output isn't finite.
- `torch.autograd.set_detect_anomaly(True)` does a similar check in the backward pass (slowly: debugging only).

Common sources: `log(0)` (use log-softmax or clamp), division by a tiny standard deviation (add ε), exp of a large number, a learning rate that's too high, and float16 overflow (m64).

## 6. The classic silent bugs in a training loop

Each of these "works" (no error) but trains badly or not at all:

1. Forgetting `optimizer.zero_grad()`: gradients accumulate across steps.
2. Applying softmax before `F.cross_entropy`, which already applies log-softmax: the gradients become tiny.
3. Shuffling inputs and labels with **different** permutations: the labels become noise.
4. Training in `eval()` mode (dropout and BatchNorm behave wrongly), or evaluating in `train()` mode.
5. Storing `loss` instead of `loss.item()` in a list, keeping every graph alive (memory grows each step).
6. Data and labels out of order after a `DataLoader` with `shuffle=True` and a separate label list.
7. Normalizing training and test data differently (m53).
8. A learning-rate scheduler stepped per epoch when it was designed per step (or the reverse).

In the exercises you'll get a training function containing several of these and fix it.

## 7. The overall recipe

From Karpathy's *A Recipe for Training Neural Networks* (2019), condensed:

1. Become one with the data (look at examples, label distributions, outliers).
2. Set up the end-to-end skeleton with a dumb baseline, fixed seed, and the initial-loss check.
3. Overfit one batch, then overfit the training set (get a model big enough to fit).
4. Regularize (more data, augmentation, dropout, weight decay, early stopping).
5. Tune (random search over learning rate and a few others).
6. Squeeze out the last bit (ensembles, longer training).

---

## Problem-solving habit #63: bisect

When a model that worked stops working, don't stare at the code: **bisect**. Find a known-good state (a commit, a config) and halve the differences until you find the change that broke it. `git bisect` (m32) automates this over commits. The same applies inside a model: replace half the components with known-good versions and see which half the bug is in.

## Common mistakes in debugging itself

- Changing several things at once, then not knowing which one helped.
- Judging from a single seed (m59): your "fix" may just be noise.
- Debugging at full scale. Shrink everything (data, model, steps) until each experiment takes seconds.

## Go deeper (optional, research-level)

1. Read Karpathy's *A Recipe for Training Neural Networks* in full. Which of its steps would have caught each bug in section 6?
2. Plot the update-to-data ratio of every layer over the first 1,000 steps of training a deeper CNN (m61) at three learning rates. Can you predict from the first 50 steps which learning rate will train best?
3. Read *Visualizing the Loss Landscape of Neural Nets* (Li et al., 2018). How do residual connections change the landscape, and how does that relate to m65?

## Your turn

Open the **Exercises** tab. The last exercise is a bug hunt: `train_classifier` has five silent bugs. Find and fix them all.
