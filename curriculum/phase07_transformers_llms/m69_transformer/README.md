# m69 · The Transformer: build GPT

**By the end you can:** assemble a complete GPT-style decoder-only Transformer (token and position embeddings, pre-norm blocks with attention and an MLP, residual connections, a final norm and a weight-tied output layer), explain each part's job, implement the main positional schemes (sinusoidal, learned, RoPE) and RMSNorm, and count parameters exactly, down to GPT-2's 124,439,808.

**Why it matters for AI:** this is *the* architecture. GPT-2, GPT-4, Claude, Llama, Mistral and Gemma are all variations of the ~150 lines you'll write here, scaled up and trained on more data. Once you've built it, model papers become readable: they're mostly about changes to one of the parts in this module.

---

## 1. The big picture

A decoder-only Transformer maps a sequence of token ids to next-token logits at every position:

```
idx (B, T) ─▶ token embedding (B, T, d) + position embedding (T, d)
           ─▶ N × Block:  x = x + Attention(LayerNorm(x))      # communicate between positions
                          x = x + MLP(LayerNorm(x))            # compute within each position
           ─▶ final LayerNorm ─▶ Linear to vocabulary ─▶ logits (B, T, V)
```

Training is exactly m67's language modelling objective: cross-entropy between the logits at position t and the token at t + 1, at every position at once (the causal mask, m68, keeps it honest).

## 2. Two kinds of layers

- **Attention** is where tokens **communicate**: each position gathers information from earlier positions.
- The **MLP** (Linear d → 4d, GELU, Linear 4d → d) is where each position **computes** on what it gathered, independently of the others. Roughly two-thirds of the parameters live here, and research suggests that much of a model's factual knowledge does too.

**GELU** (Hendrycks & Gimpel, 2016) is a smooth version of ReLU: x · Φ(x), where Φ is the standard normal CDF. Newer models often use SwiGLU.

## 3. Residual connections and pre-norm

Each sub-layer *adds* its output to a **residual stream** that flows from the embeddings to the output (m65's lesson, now essential). Gradients flow straight back through the additions, so 100-layer models train.

GPT-2 and nearly all modern models use **pre-norm**: normalize the *input* of each sub-layer (`x + attn(ln(x))`), not its output, plus one final LayerNorm before the output layer. The original Transformer used post-norm (`ln(x + attn(x))`), which needs careful learning-rate warmup to train at depth.

**RMSNorm** (used by Llama and most new models) drops LayerNorm's mean subtraction and bias:

RMSNorm(x) = γ · x / √(mean(x²) + ε)

It's cheaper and works just as well.

## 4. Where does order come from? Positional information

Attention is permutation-equivariant (m68), so position must be injected:

- **Learned absolute** (GPT-2): a second embedding table indexed by position 0..block_size−1, added to the token embeddings. Simple, but the model can't handle positions beyond `block_size`.
- **Sinusoidal** (the original Transformer): fixed vectors, with dimension pair (2i, 2i+1) at position p holding sin(p / 10000^(2i/d)) and cos(p / 10000^(2i/d)): a spectrum of wavelengths, like the hands of a clock.
- **RoPE**, rotary position embedding (Su et al., 2021; Llama, Mistral, Gemma): instead of adding anything, **rotate** each query and key vector by an angle proportional to its position. Split the vector into two halves x₁, x₂ (one frequency per pair of coordinates, θᵢ = base^(−i/half)):

  rope(x)ₚ = [x₁ cos(pθ) − x₂ sin(pθ),  x₁ sin(pθ) + x₂ cos(pθ)]

  Because rotations compose, the dot product of a query at position m with a key at position n depends only on **m − n**: attention becomes aware of *relative* position. You'll verify this property in the tests.

## 5. Weight tying

The input embedding maps token → vector (V × d); the output layer maps vector → token logits (d × V). GPT-2 **ties** them: the output layer uses the same matrix (`lm_head.weight = tok_emb.weight`). That saves V·d parameters (38.6M for GPT-2!) and usually improves results, because both matrices are learning "what does this token mean".

## 6. Initialization

GPT-2 initializes all weights from N(0, 0.02²) and biases to zero. Each block's output projections (attention `proj` and MLP `proj`) *add* to the residual stream, so with N layers the stream's variance grows with depth. GPT-2 scales those projections' std down by 1/√(2·n_layer) (two additions per block) to compensate. At initialization the logits are near zero, so the loss starts near **ln(V)** (m63).

## 7. Counting parameters

For d = d_model, L layers, vocabulary V, context T (with biases and LayerNorms, as in GPT-2):

| Part | Parameters |
|---|---|
| Token embedding (tied with the output) | V·d |
| Position embedding | T·d |
| Per block: 2 LayerNorms | 4d |
| Per block: attention qkv + proj | 3d² + 3d + d² + d |
| Per block: MLP fc + proj | 4d² + 4d + 4d² + d |
| Final LayerNorm | 2d |

That's about **12·L·d²** for the blocks. For GPT-2 small (V = 50257, T = 1024, L = 12, d = 768) the total is 124,439,808. You'll count this without allocating memory, using PyTorch's `meta` device: `with torch.device("meta"): GPT(config)` creates parameters with shapes but no data.

---

## Problem-solving habit #69: overfit one batch, again

Before training any new architecture on real data, check that it can drive the loss on one small batch to near zero (m63). For a GPT, this catches broken masks, wrong target shifts and tied-weight mistakes in seconds. It's in the tests.

## Common mistakes

- Using `nn.Embedding` for positions but indexing it with token ids (or vice versa).
- Forgetting the final LayerNorm.
- Tying weights by copying (`lm_head.weight.data = ...`) instead of sharing the same Parameter.
- Post-norm by accident: `x = ln(x + attn(x))`.
- Off-by-one targets: targets are the inputs shifted left by one, prepared by the data pipeline (m70), not inside the model.

## Go deeper (optional, research-level)

1. Read the GPT-2 paper (*Language Models are Unsupervised Multitask Learners*, Radford et al., 2019), section 2.3, and compare with your code. Then read the Llama paper (Touvron et al., 2023), section 2.2. List every architectural change.
2. Read *RoFormer* (Su et al., 2021). Prove that ⟨rope(q, m), rope(k, n)⟩ depends only on m − n, using 2-D rotation matrices.
3. Read *On Layer Normalization in the Transformer Architecture* (Xiong et al., 2020). Why does post-norm need warmup while pre-norm doesn't?

## Your turn

Open the **Exercises** tab. `CausalSelfAttention` is given (it's your m68 multi-head attention, using PyTorch's fast kernel now that you know what it does). Build the rest.
