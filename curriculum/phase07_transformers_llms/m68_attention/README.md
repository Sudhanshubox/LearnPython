# m68 · Attention from scratch

**By the end you can:** explain attention as a soft, learned dictionary lookup, implement scaled dot-product attention with causal and padding masks, build single-head and multi-head self-attention and match PyTorch's implementation exactly, reason about attention's quadratic cost, and implement the "online softmax" tiling trick behind FlashAttention.

**Why it matters for AI:** attention is the core operation of the Transformer, and therefore of GPT, Claude, Gemini, Llama, vision transformers, AlphaFold and Whisper. It solves the two problems you met in m62: any position can read from any other in one step (no vanishing over distance), and all positions are computed in parallel (no sequential loop). Every serious AI engineer can write it from memory.

---

## 1. The idea: a soft dictionary lookup

A Python dict lookup finds the one key that exactly matches the query and returns its value. **Attention** compares a query with *every* key, turns the similarities into weights that sum to 1, and returns the **weighted average** of the values:

```
weights = softmax(similarity(query, each key))
output = Σ weights_i · value_i
```

Because everything is differentiable, the model can *learn* what to look for (queries), what to advertise (keys) and what to hand over (values).

## 2. Scaled dot-product attention

With queries Q (T_q × d), keys K (T_k × d) and values V (T_k × d_v):

Attention(Q, K, V) = softmax(Q Kᵀ / √d) V

- Q Kᵀ is a (T_q × T_k) matrix of similarity **scores**: row i says how much query i matches each key.
- softmax is applied along each row (over the keys).
- **Why divide by √d?** If the entries of q and k have variance 1, their dot product has variance d. Large scores push the softmax into saturation (one weight ≈ 1, the rest ≈ 0), where gradients vanish (m44). Dividing by √d keeps the variance at 1.

The same formula works with any number of leading batch dimensions: (B, heads, T, d) inputs give (B, heads, T, T) scores.

## 3. Masks

A boolean mask says which (query, key) pairs are **allowed** (True = keep). Disallowed scores are set to −∞ before the softmax, so their weights become exactly 0.

- **Causal mask:** in a language model, position t must not see the future (it's what we're predicting). Allow key j for query i only if j ≤ i: a lower-triangular matrix, `torch.tril(torch.ones(T, T, dtype=torch.bool))`.
- **Padding mask:** sequences in a batch have different lengths and are padded to the same length; padding positions must never be attended to. Shape (B, 1, 1, T) so it broadcasts over heads and queries.

## 4. Self-attention

In **self-attention**, queries, keys and values all come from the same sequence x (B, T, d_model), through three learned linear maps:

```python
q, k, v = x @ W_q, x @ W_k, x @ W_v        # each (B, T, head_size)
out = attention(q, k, v, causal_mask(T))   # (B, T, head_size)
```

Every token produces a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what will I pass on?").

**Self-attention has no idea of order.** Shuffle the input tokens and the outputs are shuffled the same way (it's *permutation equivariant*), because it only sees a *set* of vectors. "Dog bites man" and "man bites dog" would look the same. That's why transformers add **positional information** (m69).

## 5. Multi-head attention

One head computes one kind of relationship. **Multi-head attention** runs h heads in parallel, each with head size d_model / h, then concatenates their outputs and mixes them with a final linear projection. Different heads learn different patterns: one tracks the previous token, another matching brackets, another the subject of the sentence.

Efficient implementation, with no loop over heads:

```python
q, k, v = self.qkv(x).split(d_model, dim=-1)                 # one fused Linear(d, 3d)
q = q.view(B, T, h, d_model // h).transpose(1, 2)            # (B, h, T, head_size)
...                                                          # same for k and v
out = attention(q, k, v, mask)                               # (B, h, T, head_size)
out = out.transpose(1, 2).reshape(B, T, d_model)             # concatenate the heads
out = self.proj(out)
```

PyTorch's `nn.MultiheadAttention` stores the fused weight as `in_proj_weight` (3d × d), in the order q, k, v, and the output projection as `out_proj`; you'll match it exactly. In practice, `F.scaled_dot_product_attention` (which uses FlashAttention on GPUs) is what you'd call.

## 6. The cost: quadratic in sequence length

The score matrix has T × T entries per head, and computing Q Kᵀ and weights·V costs about 4·T²·d_model FLOPs per layer. Doubling the context length quadruples attention's cost and memory. This is why context windows were long limited to a few thousand tokens, and why so much research goes into efficient attention.

## 7. FlashAttention's key trick: online softmax

Materializing the full T × T matrix is the memory bottleneck. FlashAttention (Dao et al., 2022) processes the keys in **blocks** and never stores the full matrix, using an *online softmax* that keeps, for every query, a running maximum m, a running sum of exponentials s, and a running weighted sum of values acc:

```
for each block of keys/values:
    scores = q @ kbᵀ / √d
    m_new = max(m, max of scores)
    correction = exp(m - m_new)            # rescale what we accumulated under the old max
    p = exp(scores - m_new)
    s = s · correction + sum(p)
    acc = acc · correction + p @ vb
    m = m_new
output = acc / s
```

The result is **exactly** equal to normal attention (not an approximation), using memory proportional to the block size. On GPUs, keeping each block in fast on-chip memory makes it several times faster too.

---

## Problem-solving habit #68: draw the shapes

For attention code, write the shape of every tensor on paper: (B, T, C) → (B, T, h, hs) → (B, h, T, hs) → scores (B, h, T, T) → (B, h, T, hs) → (B, T, C). Most attention bugs are a transpose on the wrong dims or a reshape that silently mixes heads and positions. A wrong reshape doesn't crash; it scrambles data.

## Common mistakes

- Applying the softmax over the wrong dimension (it must be over the **keys**, the last dim).
- Using `view` after `transpose` without making it contiguous (use `reshape`, m58).
- Reshaping (B, T, C) straight to (B, h, T, hs) with `view`, which mixes positions and heads; split heads with view(B, T, h, hs) and *then* transpose.
- A mask that's True where it should be False (convention: here True = allowed; PyTorch's `nn.MultiheadAttention` uses the opposite!).

## Go deeper (optional, research-level)

1. Read *Attention Is All You Need* (Vaswani et al., 2017), sections 3.2 and 4. Why do they argue self-attention beats recurrence and convolution? Look at their Table 1.
2. Read *FlashAttention* (Dao et al., 2022). Why is attention **memory-bound** rather than compute-bound (m64's roofline), and how does tiling fix it?
3. Read *A Mathematical Framework for Transformer Circuits* (Elhage et al., Anthropic, 2021), the sections on attention heads as "QK" and "OV" circuits. How would an "induction head" find the previous occurrence of the current token and copy what came next?

## Your turn

Open the **Exercises** tab. Don't use `F.scaled_dot_product_attention` or `nn.MultiheadAttention` in your code; the tests compare your versions against them.
