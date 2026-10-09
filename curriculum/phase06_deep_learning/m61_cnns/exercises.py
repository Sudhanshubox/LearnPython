"""m61 exercises: convolution from scratch and a small CNN.

In functions 1-5, don't use F.conv2d, nn.Conv2d or F.max_pool2d (F.unfold and F.pad are fine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. The output size of a convolution or pooling along one dimension (README section 2).
def conv_output_size(n, kernel, stride=1, padding=0, dilation=1):
    raise NotImplementedError


# 2. 2-D convolution with explicit Python loops over output positions (you may vectorize
#    over the batch and channels inside the loop body). x: (N, C_in, H, W),
#    w: (C_out, C_in, kH, kW), b: (C_out,). Zero-pad with F.pad first.
def conv2d_naive(x, w, b, stride=1, padding=0):
    raise NotImplementedError


# 3. The same convolution with im2col: F.unfold, one matrix multiply, then reshape.
#    No Python loops.
def conv2d_im2col(x, w, b, stride=1, padding=0):
    raise NotImplementedError


# 4. Max pooling with a k x k window and stride k, for H and W divisible by k, using
#    reshape and max (no loops).
def maxpool2d(x, k=2):
    raise NotImplementedError


# 5. The receptive field size (in input pixels) of one unit after a stack of layers, given
#    as a list of (kernel, stride) pairs (README section 5). An empty list gives 1.
def receptive_field(layers):
    raise NotImplementedError


# 6. The CNN from README section 6: self.features (an nn.Sequential with the two
#    conv-ReLU-maxpool blocks) and self.classifier (nn.Linear(128, n_classes)).
#    Input: (N, 1, 8, 8). Output: (N, n_classes) logits.
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# 7. Data augmentation: shift each image in the batch x (N, C, H, W) by its own random
#    offsets dy, dx in [-max_shift, max_shift], filling the uncovered area with zeros, so
#    out[n, :, i, j] = x[n, :, i - dy, j - dx] (or 0 outside the image).
#    Draw the offsets as torch.randint(-max_shift, max_shift + 1, (N, 2), generator=generator),
#    column 0 for dy and column 1 for dx. Loop over the batch if you like.
def random_shift(x, max_shift=1, generator=None):
    raise NotImplementedError


# 8. Train a SmallCNN on the digits images (X_train: (N, 1, 8, 8) float32, y_train int64)
#    with AdamW(lr, weight_decay=1e-4), cross-entropy, mini-batches of batch_size shuffled by
#    torch.Generator().manual_seed(seed), for the given epochs. If augment is True, apply
#    random_shift(xb, 1, generator) to every training batch (same generator).
#    Call torch.manual_seed(seed) before creating the model.
#    Return (model, validation accuracy as a float) after the last epoch.
def train_cnn(X_train, y_train, X_val, y_val, epochs=15, lr=3e-3, batch_size=64,
              augment=False, seed=0):
    raise NotImplementedError
