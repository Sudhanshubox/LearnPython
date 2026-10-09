# m59 · Training loops done right

**By the end you can:** write a complete, professional PyTorch training loop: datasets and data loaders, train and eval modes, an optimizer and a learning-rate schedule with warmup, validation every epoch, checkpointing the best model, early stopping, and runs that are reproducible from a seed.

**Why it matters for AI:** the training loop is the code you'll write more often than any other in deep learning. Every subtle mistake in it (forgetting `model.eval()`, leaking gradients, an unseeded shuffle, saving the last model instead of the best one) silently costs accuracy or makes results impossible to reproduce. Research results are only as trustworthy as the loop that produced them.

---

## 1. Dataset and DataLoader

A **Dataset** answers two questions: how many examples (`__len__`) and what is example *i* (`__getitem__`), the sequence protocol from m26. A **DataLoader** wraps it to produce shuffled mini-batches, stacking individual examples into batch tensors:

```python
from torch.utils.data import Dataset, DataLoader

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

g = torch.Generator().manual_seed(0)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
```

Shuffle the training data (m57), never the validation data (no need, and it makes debugging harder). For big datasets, `num_workers > 0` loads batches in background processes while the GPU trains (m33).

## 2. The canonical loop

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / count
```

Five lines do the real work: forward, loss, `zero_grad`, `backward`, `step`. Two details:

- **Weight the average by batch size.** The last batch is usually smaller; a plain mean of batch losses over-weights it.
- **`loss.item()`**, not `loss`: accumulating the tensor keeps every batch's graph alive, a classic memory leak.

## 3. `train()` vs `eval()`, and `no_grad()`

Some layers behave differently during training: **dropout** randomly zeroes activations only in training, and **batch norm** (m60) uses batch statistics in training but running averages in evaluation. `model.train()` and `model.eval()` flip that switch on every submodule. Evaluating in train mode gives noisy, wrong numbers.

`torch.no_grad()` is a separate thing: it stops autograd from recording, which saves memory and time. Use **both** for evaluation:

```python
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    ...
```

## 4. Optimizers and learning-rate schedules

**AdamW** (m41's Adam plus *decoupled* weight decay; Loshchilov & Hutter, 2019) is the default optimizer for most modern deep learning. The learning rate is still the most important hyperparameter, and it usually shouldn't be constant:

- **Warmup:** start tiny and increase linearly for the first few hundred or thousand steps. Early gradients are large and Adam's statistics aren't reliable yet; warmup prevents an early blow-up. Transformers almost always use it.
- **Cosine decay:** then decrease smoothly to a minimum following half a cosine wave, so the model settles into a good minimum.

For step s with warmup W and total T steps:

- s < W: lr = base_lr · (s + 1) / W
- otherwise: progress p = (s − W) / max(1, T − W), lr = min_lr + (base_lr − min_lr) · ½(1 + cos(πp))

In PyTorch you can write any schedule as a function of the step and pass it to `torch.optim.lr_scheduler.LambdaLR`, or use built-ins like `CosineAnnealingLR`. Call `scheduler.step()` after `optimizer.step()`.

## 5. Checkpoints

A **checkpoint** saves everything needed to resume training, not just the weights:

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_metric": best}, path)

ckpt = torch.load(path)          # weights_only=True by default in recent PyTorch: safe
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])    # Adam's moment estimates live here
```

Save `state_dict()`s, not whole model objects: pickled objects break when your code changes, and loading arbitrary pickles can execute code (m09).

## 6. Early stopping and "keep the best"

Validation loss usually falls, flattens, then rises as the model overfits. **Early stopping** stops training when the validation metric hasn't improved by at least `min_delta` for `patience` epochs. Always **save the best model** and reload it at the end: the last epoch is rarely the best one.

## 7. Reproducibility

```python
import random, numpy as np, torch
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
```

Seed everything that's random: Python, NumPy, PyTorch (which also seeds CUDA), and the DataLoader's shuffle generator. On GPUs, some operations are still nondeterministic; `torch.use_deterministic_algorithms(True)` forces determinism at some speed cost. In research, report the mean and spread over **several seeds** (m43): a single run can mislead.

---

## Problem-solving habit #59: make it run, make it right, make it fast

Get a minimal loop working end to end on a tiny subset first (seconds per epoch). Then add validation, checkpoints, schedules and logging one at a time, checking each. Only then scale up. Debugging a 10-hour run that dies at hour 9 because of a typo in the checkpoint code is the expensive way to learn this.

## Common mistakes

- Forgetting `optimizer.zero_grad()` (gradients accumulate across steps).
- Forgetting `model.eval()` when evaluating, or `model.train()` when going back to training.
- Accumulating `loss` instead of `loss.item()`.
- Selecting hyperparameters or the best epoch on the **test** set (m49): use a validation set.
- Comparing two models trained with different seeds and concluding one is better from a 0.3% difference.

## Go deeper (optional, research-level)

1. Implement a **learning-rate range test** (Smith, *Cyclical Learning Rates for Training Neural Networks*, 2017): increase the LR exponentially over one epoch, plot loss against LR, and pick the LR where loss falls fastest.
2. Read *Decoupled Weight Decay Regularization* (Loshchilov & Hutter). Why is L2 regularization not the same as weight decay for Adam, even though it is for SGD?
3. Train the same model with 5 seeds. How large is the spread in final validation accuracy? How many seeds would you need to detect a 0.5% improvement with a t-test (m43)?

## Your turn

Open the **Exercises** tab. Build the pieces in order; the final `fit` function puts them all together on handwritten digits.
