"""m64 exercises: compute and memory estimates, mixed precision, accumulation, checkpointing."""

import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# 1. FLOPs of the forward pass of an MLP (a list of nn.Linear sizes like [784, 256, 10]) on a
#    batch of `batch` examples, counting only the matrix multiplies (2 * B * d_in * d_out
#    per layer). training_flops is 3x the forward FLOPs.
def mlp_forward_flops(sizes, batch):
    raise NotImplementedError


def mlp_training_flops(sizes, batch):
    raise NotImplementedError


# 2. Estimated training time in seconds for a model with n_params parameters trained on
#    n_tokens tokens, using C = 6 * N * D FLOPs, on hardware with peak_flops FLOP/s at the
#    given utilization (fraction of peak actually achieved).
def training_time_seconds(n_params, n_tokens, peak_flops, utilization=0.4):
    raise NotImplementedError


# 3. Bytes needed for parameters + gradients + Adam's two states, all in float32
#    (README section 2). Activations not included.
def adam_training_bytes(n_params):
    raise NotImplementedError


# 4. The number of bytes the model's parameters occupy right now (sum of numel * element_size).
def parameter_bytes(model):
    raise NotImplementedError


# 5. Of the NON-ZERO values in x (float32), the fraction that become exactly zero when cast to
#    dtype (underflow). Return a float.
def underflow_fraction(x, dtype):
    raise NotImplementedError


# 6. Dynamic loss scaling (README section 5).
#    - self.scale starts at init_scale; self.good_steps counts consecutive good steps.
#    - unscale_and_check(params): divide every existing .grad by self.scale IN PLACE, and
#      return True if all of them are finite afterwards, else False.
#    - update(found_finite): if not found_finite, halve the scale and reset good_steps to 0;
#      otherwise add one good step, and when good_steps reaches growth_interval, double the
#      scale and reset good_steps to 0.
#    - step(optimizer, params): unscale_and_check, call optimizer.step() ONLY if the gradients
#      are finite, then update(...). Return True if the optimizer stepped.
class DynamicLossScaler:
    def __init__(self, init_scale=2.0 ** 16, growth_interval=2000):
        raise NotImplementedError

    def unscale_and_check(self, params):
        raise NotImplementedError

    def update(self, found_finite):
        raise NotImplementedError

    def step(self, optimizer, params):
        raise NotImplementedError


# 7. One training step with autocast (README section 4): zero the grads, run the forward pass
#    and cross-entropy inside torch.autocast(device_type=device, dtype=dtype), backward and
#    step outside it. Return the loss as a float.
def autocast_train_step(model, optimizer, xb, yb, device="cpu", dtype=torch.bfloat16):
    raise NotImplementedError


# 8. Gradient accumulation (README section 6): zero the grads once, then for each (xb, yb)
#    in micro_batches add the gradient of cross-entropy / len(micro_batches), then take ONE
#    optimizer step. Return the total loss (the sum of the scaled micro-batch losses) as a float.
def accumulated_step(model, optimizer, micro_batches):
    raise NotImplementedError


# 9. A deep MLP of `depth` blocks, each nn.Sequential(nn.Linear(width, width), nn.GELU()),
#    stored in an nn.ModuleList self.blocks. When self.use_checkpoint is True AND the model
#    is in training mode, run every block through
#    checkpoint(block, x, use_reentrant=False); otherwise call it normally.
class CheckpointedMLP(nn.Module):
    def __init__(self, width, depth, use_checkpoint=True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 10. Benchmark fn(): call it `warmup` times untimed, then `repeats` times, timing each call
#     with time.perf_counter(). If torch.cuda.is_available(), call torch.cuda.synchronize()
#     before reading the clock on both sides of each call. Return the median time in seconds.
def benchmark(fn, warmup=3, repeats=10):
    raise NotImplementedError
