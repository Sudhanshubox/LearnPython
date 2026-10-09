# m60 · Initialization, normalization and regularization

**By the end you can:** show experimentally why initialization matters in deep networks, implement batch normalization and layer normalization from scratch (including running statistics and train/eval behaviour), implement dropout and label smoothing, and set up weight decay the way modern models do.

**Why it matters for AI:** a 20-layer network with the wrong initialization doesn't train at all: its activations explode to 10²⁰ or vanish to 10⁻²⁰. Normalization layers are why today's 100-layer networks train reliably; **LayerNorm** sits in every transformer block (Phase 7). Regularization decides whether a model memorizes or generalizes. These are the details that separate "my model doesn't learn" from a working system.

---

## 1. Initialization: keep the signal alive

Each linear layer multiplies its input's variance by roughly fan_in · Var(W). Stack 20 layers and that factor is raised to the 20th power. You'll measure this with **forward hooks** (functions PyTorch calls after a module's forward pass):

```python
stds = []
handle = layer.register_forward_hook(lambda module, inputs, output: stds.append(output.std().item()))
model(x)
handle.remove()          # always remove hooks when done
```

What you'll find in a 20-layer, 256-wide ReLU network fed unit-variance inputs:

| Init | Std of the final activations |
|---|---|
| N(0, 1) | ~10²⁰ (explodes) |
| N(0, 0.01²) | ~10⁻²⁰ (vanishes) |
| Xavier/Glorot: Var = 2/(fan_in + fan_out) | ~10⁻³ (slowly vanishes: it was designed for tanh) |
| Kaiming/He: Var = 2/fan_in | ~0.5 (stable) |

With **tanh**, N(0, 1) doesn't explode, it **saturates**: almost every unit outputs ±1, where tanh's gradient is about 0, so nothing learns. A stable activation std isn't the whole story; you also need units in their sensitive range.

`torch.nn.init` has all of these (`kaiming_normal_`, `xavier_normal_`, ...). `nn.Linear`'s default init is a scaled-down Kaiming uniform, fine for shallow nets.

## 2. Batch normalization

**BatchNorm** (Ioffe & Szegedy, 2015) normalizes each feature over the batch, then lets the network re-scale it:

μ = mean over the batch,  σ² = variance over the batch (biased: divide by n)
y = γ · (x − μ) / √(σ² + ε) + β

γ (`weight`) and β (`bias`) are learned, starting at 1 and 0. The layer keeps **running averages** for use at evaluation time, when there may be only one example:

running_mean ← (1 − m) · running_mean + m · μ
running_var ← (1 − m) · running_var + m · σ²_unbiased   (PyTorch uses the unbiased n/(n−1) version here)

with momentum m = 0.1. In `eval()` mode, BatchNorm normalizes with the running statistics and doesn't update them. Running averages aren't learned by gradient descent, so they're **buffers**, not parameters: register them with `self.register_buffer(name, tensor)` so they appear in `state_dict()` and move with `.to(device)`, but aren't given to the optimizer.

Why it helps: activations stay well-scaled at every depth regardless of initialization, which allows higher learning rates. The downsides: behaviour depends on the batch (bad for small batches) and differs between train and eval, which is a classic source of bugs.

## 3. Layer normalization

**LayerNorm** (Ba et al., 2016) normalizes each *example* over its features (the last dimension), so it doesn't depend on the batch at all and behaves identically in training and evaluation:

y = γ · (x − mean(x)) / √(var(x) + ε) + β   with mean and var over the last dim

That's why transformers use it: sequences of different lengths and tiny batches are fine. (Many LLMs use **RMSNorm**, which drops the mean subtraction: y = γ · x / √(mean(x²) + ε).)

## 4. Dropout

During training, **dropout** (Srivastava et al., 2014) zeroes each activation with probability p, forcing the network not to rely on any single unit; it works like training an ensemble of sub-networks. **Inverted dropout** scales the survivors by 1/(1 − p) during training so the expected value is unchanged, and does nothing at all at evaluation time.

## 5. Weight decay, done properly

Weight decay (m57, m59) shrinks weights toward zero. Modern practice (GPT-2, most transformers): apply it only to **weight matrices** (parameters with 2+ dimensions), not to biases or normalization parameters (1-D), which have no reason to be pushed to zero. Optimizers accept **parameter groups** for this:

```python
torch.optim.AdamW([
    {"params": matrices, "weight_decay": 0.1},
    {"params": vectors, "weight_decay": 0.0},
], lr=3e-4)
```

## 6. Label smoothing

Training with one-hot targets pushes logits toward infinity: the model becomes overconfident. **Label smoothing** (Szegedy et al., 2016) mixes the target with a uniform distribution: the true class gets 1 − ε + ε/k and every class gets ε/k. The loss is the cross-entropy against this soft target:

loss = −Σⱼ targetⱼ · log softmax(z)ⱼ, averaged over the batch

PyTorch: `F.cross_entropy(logits, y, label_smoothing=0.1)`.

## 7. Other regularizers worth knowing

Data augmentation (often the strongest; m61), early stopping (m59), smaller models, and more data. In the large-model regime, models are often trained for less than one epoch, where the data itself is the regularizer.

---

## Problem-solving habit #60: measure, don't guess

When training misbehaves, instrument the model: hooks on activations and gradients, std and mean per layer, the fraction of dead ReLUs. A table of numbers per layer turns "it doesn't learn" into "layer 14's activations are 10⁻⁹", which points straight to the fix.

## Common mistakes

- Forgetting `model.eval()`, so BatchNorm keeps updating its running statistics on test data and dropout stays on.
- BatchNorm with batch size 1 in training (the variance is 0).
- Storing running statistics as plain tensor attributes (not saved, not moved to the GPU) or as `nn.Parameter`s (the optimizer changes them).
- Weight-decaying biases and normalization gains.

## Go deeper (optional, research-level)

1. Read *How Does Batch Normalization Help Optimization?* (Santurkar et al., 2018). The original paper's explanation, "internal covariate shift", turned out to be mostly wrong. What do they propose instead?
2. Implement RMSNorm and compare it with LayerNorm in a small MLP. Read *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019).
3. Kaiming init assumes ReLU zeroes exactly half of its inputs. Derive the variance-preserving init for Leaky ReLU with slope a (the answer is in He et al., 2015, the PReLU paper).

## Your turn

Open the **Exercises** tab. The tests compare your BatchNorm, LayerNorm and label smoothing with PyTorch's built-in versions.
