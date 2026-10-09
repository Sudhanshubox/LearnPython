# m64 · Performance: FLOPs, memory, mixed precision and big batches on small hardware

**By the end you can:** estimate the compute and memory a model needs before you run it, explain float32 vs float16 vs bfloat16, implement dynamic loss scaling, train with autocast, simulate large batches with gradient accumulation, trade compute for memory with activation checkpointing, and benchmark code correctly.

**Why it matters for AI:** compute is the main constraint in modern AI. Whether a training run fits on your GPU, takes a day or a month, or costs ₹10,000 or ₹10 crore comes down to the arithmetic in this module. Research labs decide what's feasible with back-of-the-envelope estimates like these, and every LLM training codebase uses mixed precision, gradient accumulation and checkpointing.

Everything here runs on a CPU, so you can do it on a laptop; the same code runs on a GPU, where the speedups are much larger.

---

## 1. Counting FLOPs

A matrix multiply of (m × k) by (k × n) does m·k·n multiply-adds = **2mkn FLOPs** (floating-point operations). So a linear layer on a batch of B examples costs 2·B·d_in·d_out FLOPs in the forward pass.

The backward pass costs about **twice** the forward (one matmul for the input gradient, one for the weight gradient), so training costs about **3× forward**. For a network with N parameters processing one example (or token), forward ≈ 2N FLOPs and training ≈ **6N FLOPs**. This "6N per token" rule is how you estimate LLM training cost: GPT-3 (175B parameters, 300B tokens) needed about 6 × 175e9 × 300e9 ≈ 3.15e23 FLOPs.

Divide by your hardware's real throughput (peak FLOP/s × utilization, often 30–50%) to get a time estimate.

## 2. Counting memory

Training memory has four parts:

1. **Parameters:** N × bytes per value.
2. **Gradients:** the same size as the parameters.
3. **Optimizer state:** Adam keeps two values per parameter (m and v, m41).
4. **Activations:** everything saved for the backward pass (m57). These grow with batch size × sequence length × depth, and are often the largest part.

In plain float32 with Adam, parts 1–3 take 4 + 4 + 8 = **16 bytes per parameter**. A 7-billion-parameter model therefore needs about 112 GB *before* activations, which is why large models are trained across many GPUs (ZeRO, FSDP) and fine-tuned with tricks like LoRA (Phase 7).

## 3. Floating-point formats

| Format | Bits (sign/exponent/mantissa) | Max | Smallest normal | Precision (eps) |
|---|---|---|---|---|
| float32 | 1/8/23 | 3.4e38 | 1.2e-38 | 1.2e-7 |
| float16 | 1/5/10 | 65,504 | 6.1e-5 | 9.8e-4 |
| bfloat16 | 1/8/7 | 3.4e38 | 1.2e-38 | 7.8e-3 |

(`torch.finfo(dtype)` reports these.) Half-precision formats halve memory and run much faster on modern GPUs' tensor cores. But:

- **float16** has a tiny range: values above 65,504 overflow to `inf`, and small gradients (below ~6e-8, even counting subnormals) **underflow to zero**.
- **bfloat16** keeps float32's range (same 8 exponent bits) with less precision. It rarely overflows or underflows, which is why it's the default for training LLMs on hardware that supports it.

## 4. Mixed precision and autocast

**Mixed precision** (Micikevicius et al., 2018) runs most operations (matmuls, convolutions) in 16-bit, keeps precision-sensitive ones (reductions, softmax, losses) in float32, and keeps a float32 **master copy** of the weights for the updates (small updates added to big weights would be lost in 16-bit).

PyTorch does the per-operation choice for you:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):    # "cpu" works too
    logits = model(xb)               # matmuls run in bf16
    loss = F.cross_entropy(logits, yb)
loss.backward()                      # outside the autocast block
optimizer.step()                     # parameters stay float32
```

## 5. Loss scaling (for float16)

To stop small float16 gradients from underflowing, multiply the loss by a big **scale** S before `backward()` (all gradients get multiplied by S too, shifting them into range), then divide the gradients by S before the optimizer step. **Dynamic** loss scaling picks S automatically:

- If any gradient is `inf` or `nan`, the scale was too big: **skip this step** and halve S.
- After `growth_interval` consecutive good steps, double S (try to use more of the range).

PyTorch's `torch.amp.GradScaler` implements exactly this. With bfloat16 you usually don't need it.

## 6. Gradient accumulation: big batches on small hardware

If a batch of 256 doesn't fit in memory, run 8 micro-batches of 32, call `backward()` on each (gradients **add up**, m56), and step once:

```python
optimizer.zero_grad()
for xb, yb in micro_batches:                 # accum_steps of them
    loss = loss_fn(model(xb), yb) / accum_steps
    loss.backward()
optimizer.step()
```

Dividing each loss by `accum_steps` makes the accumulated gradient equal the **mean** over the whole big batch, so the result matches a true large-batch step exactly (for models without batch-dependent layers like BatchNorm).

## 7. Activation checkpointing

Instead of storing every activation for the backward pass, store only some and **recompute** the rest during backward. This costs about one extra forward pass (~33% more compute) and can cut activation memory dramatically:

```python
from torch.utils.checkpoint import checkpoint
out = checkpoint(block, x, use_reentrant=False)
```

The results and gradients are identical; only the memory/compute trade-off changes.

## 8. Measuring speed correctly

- **Warm up** first: the first calls include one-time costs (memory allocation, kernel selection, compilation).
- **Repeat** and report the **median** (robust to outliers, m43).
- On a GPU, operations are **asynchronous**: call `torch.cuda.synchronize()` before reading the clock, or you'll time only the launch.
- Use `time.perf_counter()`, not `time.time()`.
- Use `torch.inference_mode()` for pure inference and `torch.compile(model)` (PyTorch 2) to fuse operations for speed.

Before optimizing anything, **profile** (`torch.profiler`, m34's lesson): the bottleneck is often the data loader, not the model.

---

## Problem-solving habit #64: estimate before you run

Before launching anything expensive, write down the expected FLOPs, memory and time. If reality differs from the estimate by more than 2×, find out why: it's either a bug or something you didn't understand about the system. Both are worth knowing. This habit is called a **Fermi estimate**, and it's how research leads decide which experiments are worth running.

## Common mistakes

- Calling `backward()` inside the autocast block (it should be outside).
- Forgetting to divide the loss by `accum_steps`, so the effective learning rate is multiplied by it.
- Casting the whole model to float16 (`model.half()`) for training, without master weights or loss scaling.
- Timing GPU code without synchronizing.

## Go deeper (optional, research-level)

1. Read *Mixed Precision Training* (Micikevicius et al., 2018) and *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (Rajbhandari et al., 2020). How does ZeRO split the 16 bytes per parameter across GPUs?
2. Read *Training Compute-Optimal Large Language Models* (Hoffmann et al., 2022, "Chinchilla"). Using C ≈ 6ND, how many tokens should a model with a 1e21 FLOP budget be trained on?
3. Estimate the **arithmetic intensity** (FLOPs per byte moved) of a matrix multiply and of an elementwise ReLU. Why is one compute-bound and the other memory-bound? Look up the **roofline model**, and read how FlashAttention (Dao et al., 2022) uses this insight.

## Your turn

Open the **Exercises** tab. Everything runs on the CPU; the tests check exact equivalences (accumulation = big batch, checkpointing = no checkpointing) rather than speed.
