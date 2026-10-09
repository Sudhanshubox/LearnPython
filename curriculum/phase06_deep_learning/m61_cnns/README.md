# m61 · Convolutional neural networks

**By the end you can:** implement 2-D convolution from scratch (first with loops, then fast with the im2col trick), compute output sizes and receptive fields, implement max pooling, build a small CNN in PyTorch, use data augmentation, and train it to over 97% accuracy on handwritten digits.

**Why it matters for AI:** CNNs launched the deep learning revolution (AlexNet, 2012) and still power much of computer vision, medical imaging and audio. Their core ideas, **weight sharing**, **locality** and **building in the right inductive bias**, reappear everywhere, and vision transformers are best understood in contrast to them. The im2col trick is also a lesson in how all deep learning becomes matrix multiplication on a GPU.

---

## 1. Why not just an MLP on pixels?

An MLP treats a 224×224×3 image as 150,528 unrelated numbers: the first layer alone would need hundreds of millions of weights, and a cat in the top-left corner would have to be learned separately from a cat in the bottom-right. Images have structure that we can build in:

- **Locality:** nearby pixels are related; a small patch is enough to detect an edge.
- **Translation equivariance:** an edge detector useful in one place is useful everywhere, so **share** the same weights across all positions.

A convolution does exactly that: slide a small filter over the image and take a dot product at every position.

## 2. The convolution operation

Input x: shape (N, C_in, H, W). Weights w: (C_out, C_in, kH, kW). Bias b: (C_out,).

out[n, o, i, j] = b[o] + Σ_c Σ_p Σ_q w[o, c, p, q] · x_padded[n, c, i·s + p, j·s + q]

with stride s, after zero-padding x by `padding` pixels on every side. (Strictly this is cross-correlation, but everyone calls it convolution.)

**Output size** for input size n, kernel k, stride s, padding p (and dilation d, which spaces out the kernel's taps):

out = ⌊(n + 2p − d·(k − 1) − 1) / s⌋ + 1

So a 3×3 kernel with padding 1 and stride 1 keeps the size ("same" padding); stride 2 halves it.

## 3. Making it fast: im2col

The loop version has 6 nested loops: hopelessly slow in Python. The trick used by real libraries: copy every receptive-field patch into a column of a big matrix (**im2col**), then the whole convolution becomes **one matrix multiply**:

```python
cols = F.unfold(x, kernel_size=(kH, kW), padding=p, stride=s)   # (N, C_in*kH*kW, L)  L = out_h*out_w
out = w.view(C_out, -1) @ cols                                   # (N, C_out, L)
out = out + b.view(1, -1, 1)
out = out.view(N, C_out, out_h, out_w)
```

It uses more memory (each pixel is copied up to kH·kW times) but turns the work into the operation GPUs are best at. This "reshape everything into a matmul" move is worth remembering.

## 4. Pooling

**Max pooling** with a k×k window and stride k keeps the largest value in each window, halving (for k=2) the spatial size. It adds a little translation invariance and reduces computation. For non-overlapping windows it's just a reshape and a max:

(N, C, H, W) → (N, C, H/k, k, W/k, k) → max over the two k dims.

Many modern networks replace pooling with stride-2 convolutions, and end with **global average pooling** (the mean over all positions) before the classifier.

## 5. Receptive field

The **receptive field** of a unit is the patch of the input that can influence it. With layers of kernel kᵢ and stride sᵢ, the receptive field grows as:

r ← r + (k − 1) · j,   j ← j · s     (starting from r = 1, j = 1)

where j is the "jump" (the distance in input pixels between adjacent units). Two 3×3 convs see 5×5; three see 7×7, with fewer parameters than one 7×7 conv and more nonlinearity: this is VGG's insight. Stride and pooling make the receptive field grow much faster.

## 6. A CNN in PyTorch

```python
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8x8 -> 4x4
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 4x4 -> 2x2
        )
        self.classifier = nn.Linear(32 * 2 * 2, n_classes)

    def forward(self, x):                      # x: (N, 1, 8, 8)
        return self.classifier(self.features(x).flatten(1))
```

The pattern: blocks of conv → (norm) → nonlinearity → downsample, with channels growing as the spatial size shrinks, then a classifier head. `flatten(1)` keeps the batch dimension.

## 7. Data augmentation

The cheapest regularizer in vision: show the model randomly altered training images that should keep their labels (shifts, flips, crops, colour changes). It teaches the invariances you want and effectively enlarges the dataset. Only augment **training** data. Choose augmentations that preserve the label: horizontally flipping a "6" is fine for cats but not for digits. (In real projects, use `torchvision.transforms`.)

---

## Problem-solving habit #61: from slow and obviously right to fast and verified

Write the naive loop version first: it's easy to check by hand and against the definition. Then write the fast version and test it against the slow one on random inputs. This **reference implementation** pattern is how kernel engineers write GPU code (FlashAttention was verified against plain attention exactly like this).

## Common mistakes

- Feeding (N, H, W) instead of (N, C, H, W): add the channel dim with `x.unsqueeze(1)`.
- Getting the flattened size wrong in the first linear layer. Print the shape after `features` once, or use `nn.LazyLinear`.
- Using `view` after a `permute` (not contiguous): use `reshape` (m58).
- Augmenting the validation or test set.

## Go deeper (optional, research-level)

1. Read the AlexNet paper (*ImageNet Classification with Deep Convolutional Neural Networks*, Krizhevsky et al., 2012). Which of its tricks are still used today and which were abandoned?
2. Convolution is translation *equivariant*: shifting the input shifts the output. Verify this numerically with your `conv2d_im2col` (careful with borders). Is max pooling equivariant? Is the full CNN invariant? Read *Making Convolutional Networks Shift-Invariant Again* (Zhang, 2019).
3. Count the multiply-adds of a convolution layer as a function of its sizes. Then read about **depthwise separable convolutions** (MobileNet). How many fewer operations do they need for a 3×3 conv from 256 to 256 channels?

## Your turn

Open the **Exercises** tab. Don't use `F.conv2d`, `nn.Conv2d` or `F.max_pool2d` in your from-scratch functions (`F.unfold` is allowed). `SmallCNN` uses the built-in layers.
