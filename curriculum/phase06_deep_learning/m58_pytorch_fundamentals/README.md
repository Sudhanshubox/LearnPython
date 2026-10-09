# m58 · PyTorch fundamentals

**By the end you can:** create and manipulate tensors fluently (shapes, dtypes, devices, views), use autograd to get gradients of anything, write your own `nn.Module` with registered parameters, and even define a custom autograd function with its own backward pass.

**Why it matters for AI:** PyTorch is the language of AI research: nearly every paper's code, every open-source model and most production training runs use it. You already understand what it does internally (m56, m57). This module makes you fast with its API so you can spend your effort on ideas, not syntax.

---

## 1. Tensors: NumPy arrays with superpowers

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
x.shape, x.dtype, x.device          # torch.Size([2, 2]), torch.float32, device(type='cpu')
torch.zeros(3, 4); torch.ones(2, 3); torch.arange(10); torch.linspace(0, 1, 5)
torch.randn(2, 3, generator=torch.Generator().manual_seed(0))
```

Almost everything you know from NumPy (m36, m37) carries over: indexing, slicing, broadcasting, `sum(dim=...)` (PyTorch says `dim`, NumPy says `axis`; both accept either), `@` for matrix multiplication. The two superpowers are **autograd** and **devices** (GPUs).

### dtypes

The default floating dtype is **float32**, not NumPy's float64. Deep learning rarely needs float64, and float32 is twice as fast and half the memory; later you'll go down to bfloat16 (m64). Integer labels must be **int64** (`torch.long`) for loss functions like `cross_entropy`.

```python
a = torch.from_numpy(np_array)      # shares memory with the NumPy array! dtype preserved (float64)
a = torch.as_tensor(np_array, dtype=torch.float32)   # converts (copies) when the dtype differs
t.numpy()                           # back to NumPy (CPU tensors only; shares memory)
t.item()                            # a one-element tensor -> Python number
```

Shared memory means changing one changes the other. That's efficient, and a source of surprising bugs.

### Views vs copies

`view`, `reshape` (usually), `transpose`, `permute`, slicing and `expand` return **views**: new shape metadata over the same memory, with no copy. After `transpose` or `permute`, a tensor is no longer **contiguous** in memory, so `.view()` on it fails; use `.reshape()` (which copies if needed) or call `.contiguous()` first.

```python
x = torch.arange(6).view(2, 3)
x.T.is_contiguous()     # False
x.T.reshape(6)          # fine (copies)
```

### Shapes you'll use constantly

- `unsqueeze(dim)` / `squeeze(dim)`: add or remove a size-1 dimension.
- `x[:, None]` works like NumPy.
- `torch.cat` joins along an existing dim; `torch.stack` creates a new one.
- `x.transpose(-2, -1)` swaps the last two dims: the batched transpose you'll use in attention.
- `torch.matmul` / `@` broadcasts over leading **batch** dims: (B, T, D) @ (B, D, S) → (B, T, S).

## 2. Devices

```python
device = "cuda" if torch.cuda.is_available() else "cpu"   # Apple silicon: "mps"
x = x.to(device)
model = model.to(device)
```

Every tensor in one operation must be on the same device. "Expected all tensors to be on the same device" is the most common error message in deep learning; the fix is always a missing `.to(device)`. Write device-agnostic code and you can develop on a laptop CPU and train on a GPU unchanged.

## 3. Autograd

```python
w = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (w ** 2).sum()
loss.grad_fn            # <SumBackward0>: the graph from m56, built automatically
loss.backward()
w.grad                  # tensor([2., 4.])
```

Exactly your `Value` engine, but on tensors:

- Tensors with `requires_grad=True` are **leaves**; results of operations on them record a `grad_fn`.
- `backward()` on a scalar fills `.grad` on the leaves. Gradients **accumulate** (`+=`, m56), so zero them between steps: `w.grad.zero_()` or `w.grad = None`.
- `with torch.no_grad():` turns recording off: use it for parameter updates and evaluation (faster, less memory).
- `t.detach()` returns a tensor sharing data but cut out of the graph: a **stop-gradient**.
- In-place updates on a leaf that requires grad must happen under `no_grad()`.

## 4. `nn.Module`: the building block

```python
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                  # always first!
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TwoLayerNet(4, 8, 3)
logits = model(torch.randn(5, 4))       # call the module, not .forward(): hooks run in __call__
list(model.named_parameters())           # fc1.weight, fc1.bias, fc2.weight, fc2.bias
model.state_dict()                       # name -> tensor: what you save to disk
```

Assigning an `nn.Module` or an `nn.Parameter` as an attribute **registers** it (PyTorch overrides `__setattr__`, m26), so `parameters()`, `.to(device)` and `state_dict()` find it automatically. A plain tensor attribute is *not* registered: it won't be trained or moved. For a list of layers use `nn.ModuleList`, not a Python list, for the same reason.

`nn.Linear(d_in, d_out)` stores `weight` with shape **(d_out, d_in)** and computes `x @ weight.T + bias`. That transposed layout differs from m57's (d_in, d_out). Its default init draws from U(−1/√d_in, 1/√d_in).

To freeze part of a model (fine-tuning, Phase 7), set `requires_grad = False` on its parameters.

## 5. Custom autograd functions

Sometimes you need a backward pass that isn't the true derivative. The classic case is the **straight-through estimator (STE)**: rounding has zero gradient almost everywhere, which would stop learning in a quantized network, so we *pretend* it's the identity on the way back.

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output              # pass the gradient straight through

y = RoundSTE.apply(x)
```

`ctx.save_for_backward(...)` stores tensors the backward pass needs, like m57's caches. STE is used in quantization-aware training and VQ-VAEs.

---

## Problem-solving habit #58: read the error, then print the shapes

PyTorch errors are precise: "mat1 and mat2 shapes cannot be multiplied (32x64 and 128x10)" tells you exactly which layer is wrong. When something fails, print `x.shape, x.dtype, x.device` of every input at the failing line before changing anything. Nine times out of ten, one of those three is the bug.

## Common mistakes

- Calling `model.forward(x)` instead of `model(x)`.
- Storing layers in a plain Python list (they won't be registered or trained).
- Float64 data from NumPy hitting a float32 model ("expected scalar type Float but found Double").
- Using `.data` to update parameters (it silently bypasses autograd's safety checks); use `torch.no_grad()` instead.
- Forgetting that `torch.from_numpy` shares memory.

## Go deeper (optional, research-level)

1. Read the PyTorch docs page "Autograd mechanics". What does `retain_graph=True` do, and when would you need `create_graph=True`? (Hint: gradients of gradients, used in meta-learning and gradient penalties.)
2. Use `torch.func.grad` and `torch.func.vmap` (the JAX-style functional API) to compute per-example gradients for a batch. Why are per-example gradients needed for differentially private training?
3. Read *Estimating or Propagating Gradients Through Stochastic Neurons* (Bengio et al., 2013), the origin of the straight-through estimator. Why does such a "wrong" gradient still work?

## Your turn

Open the **Exercises** tab. Use PyTorch operations only (no NumPy inside your functions), and no Python loops over tensor elements.
