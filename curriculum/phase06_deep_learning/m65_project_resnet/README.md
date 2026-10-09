# m65 · Project: reproduce the ResNet degradation result

**Phase 6 finale, and your first paper reproduction.** In 2015, He, Zhang, Ren and Sun published *Deep Residual Learning for Image Recognition*, one of the most cited papers in all of science. Its starting point was a surprising observation: making a plain convolutional network **deeper** made it **worse**, and not because of overfitting: the *training* error went up too. Their fix, the **residual connection**, is now in nearly every deep network, including every transformer.

Your job: reproduce the paper's central claim at small scale on a CPU, run an ablation, investigate *why* it happens, and write it up like a researcher.

---

## Background: read the paper first

Read sections 1, 3.1, 3.2 and 4.1 of the paper (search for "Deep Residual Learning for Image Recognition", arXiv 1512.03385). Use the three-pass method: skim the abstract, figures and conclusions first; then read the method carefully; then try to explain Figure 1 and Figure 6 in your own words. Ask the mentor to quiz you on it.

The key ideas:

- **Degradation:** a deeper plain network should be at least as good as a shallower one: it could copy the shallow one and set the extra layers to the identity. In practice, the optimizer fails to find that solution. Figure 1 shows a 56-layer plain network with *higher training error* than a 20-layer one.
- **Residual learning:** instead of asking a block to learn a mapping H(x), let it learn the **residual** F(x) = H(x) − x and output F(x) + x. If the identity is optimal, the block just has to push its weights toward zero, which is easy.

```
plain block:     x ─ conv ─ BN ─ ReLU ─ conv ─ BN ─ ReLU ─▶
residual block:  x ─ conv ─ BN ─ ReLU ─ conv ─ BN ─(+)─ ReLU ─▶
                 └────────────── identity ───────────┘
```

The identity "shortcut" is the same gradient highway as the LSTM's cell state (m62): gradients flow straight through the additions to the early layers.

## Your setup

The real paper used CIFAR-10 and GPUs. We scale down so each run takes seconds on a laptop, while keeping what matters:

- **Data:** scikit-learn's handwritten digits (8×8 grayscale, bundled, no download), 75/25 stratified split.
- **Model:** `DeepNet(n_blocks, residual)`: a stem (3×3 conv → BN → ReLU, 1 → 16 channels), `n_blocks` blocks of two 3×3 convolutions (16 channels), global average pooling, and a linear classifier. The number of conv layers is 2·n_blocks + 1.
- **Training:** SGD with momentum 0.9, weight decay 1e-4, learning rate 0.05 with cosine decay per step, batch 64, 8 epochs.
- **Comparison:** shallow (2 blocks, 5 conv layers) vs deep (12 blocks, 25 conv layers), plain vs residual, **3 seeds each**.

## Steps (`exercises.py`)

1. `load_data`: load and split the digits as image tensors.
2. `BasicBlock` and `DeepNet`: the plain and residual architectures, sharing all code except the shortcut.
3. `train`: the training loop from m59, with a cosine schedule.
4. `run_experiment` and `summarize`: run every (depth, residual, seed) combination and aggregate means and standard deviations (m43: never trust a single seed).
5. `layer_grad_norms`: investigate *why*. Measure the gradient norm of every conv layer at initialization, from first to last.
6. `zero_init_residual`: an **ablation** from later work (Goyal et al., 2017, *Accurate, Large Minibatch SGD*): initialize the last BatchNorm's γ in each residual block to zero, so every block starts as the identity.
7. `write_report`: your write-up.

## What you should find

If everything is right, the deep **plain** network ends with a **much higher training loss** than the shallow plain one (degradation reproduced!), while the deep **residual** network trains as well as the shallow ones. The gradient measurements are the interesting part: with BatchNorm, the deep plain network's gradients don't vanish; they **explode** toward the early layers. That's a known result (Yang et al., 2019, *A Mean Field Theory of Batch Normalization*) and a good example of a reproduction teaching you something the original paper didn't say.

## Report outline (`write_report`)

```markdown
# Residual vs plain networks
## Setup          (data, models, training, seeds: enough for someone to rerun it)
## Results        (a Markdown table: blocks, conv layers, residual, train loss mean ± std, val acc)
## Analysis       (was degradation reproduced? what do the gradient norms show?)
## Ablations      (what you changed and what happened)
## Limitations    (what would you need to claim this holds for CIFAR or ImageNet?)
```

## Stretch goals (optional)

- Extend to 4, 8, 16 and 24 blocks and plot final training loss against depth for both architectures (Figure 6 of the paper, in miniature).
- Plot training loss curves per epoch for all four configurations on one figure (m48).
- Run the `zero_init_residual` ablation over 3 seeds. Does it speed up early training?
- Remove BatchNorm entirely. What happens to the plain and residual networks now? (You may need a lower learning rate.)
- If you have a GPU, repeat on CIFAR-10 with `torchvision.datasets.CIFAR10` and the paper's 20- and 56-layer configurations.

## Go deeper (optional, research-level)

1. Read *Identity Mappings in Deep Residual Networks* (He et al., 2016). Why does moving BN and ReLU *before* the convolutions ("pre-activation") help even more? Transformers use the same idea: "pre-norm" blocks.
2. Read *Residual Networks Behave Like Ensembles of Relatively Shallow Networks* (Veit et al., 2016). Try their experiment: delete one block from your trained deep residual network and from the plain one. How much does accuracy drop in each?
3. A transformer block is x + Attention(LN(x)), then x + MLP(LN(x)). Find the residual connections and the normalizations. You'll build one in Phase 7.

## Your turn

Open the **Exercises** tab. The full experiment takes about a minute on a laptop CPU. When you've finished, ask the mentor to review your report as a paper reviewer would.
