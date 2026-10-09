# m71 · Inference: sampling, the KV cache and quantization

**By the end you can:** control generation with temperature, top-k and nucleus (top-p) sampling, make generation dramatically cheaper with a **KV cache** (and verify it gives identical results), estimate KV-cache memory, and shrink a model with int8 and grouped int4 **weight quantization** while measuring the error it introduces.

**Why it matters for AI:** training happens once; inference happens billions of times. Inference cost decides what a model costs per token, how fast it answers, and whether it fits on a phone or a single GPU. Every serving system (vLLM, llama.cpp, TensorRT-LLM) is built on the techniques in this module, and sampling settings are something every LLM user tunes.

---

## 1. Decoding strategies

At each step the model gives logits over the vocabulary. How you pick the next token changes the text a lot:

- **Greedy** (temperature 0): always the most likely token. Deterministic, but tends to loop ("the the the...") and is bland.
- **Temperature τ:** sample from softmax(logits / τ). τ < 1 sharpens toward the top choices; τ > 1 flattens toward randomness.
- **Top-k:** sample only among the k most likely tokens (set the others' logits to −∞). Cuts off the long tail of nonsense, but a fixed k is too many when the model is confident and too few when it isn't.
- **Top-p (nucleus sampling;** Holtzman et al., 2020): keep the **smallest** set of most-likely tokens whose total probability reaches p (e.g. 0.9). It adapts: when the model is confident, the nucleus is one or two tokens; when it isn't, it's wide. Implementation: sort the probabilities in descending order, take the cumulative sum, and remove a token if the cumulative probability *before* it already reached p (so the top token is always kept).

Typical chat settings: temperature 0.7–1.0 with top-p 0.9–0.95. For code or maths, lower temperatures.

## 2. Why naive generation is wasteful

m70's `generate` reruns the whole model on the whole context for every new token: generating n tokens costs O(n²) token-forward-passes, and attention O(n³) in total. But the keys and values of earlier tokens **never change** (thanks to the causal mask, a token's representation depends only on tokens before it). So compute them once and keep them.

## 3. The KV cache

For each layer, store the keys and values of all previous tokens: tensors of shape (B, heads, T_past, head_size). Generation then has two phases:

1. **Prefill:** run the prompt through the model once, computing all positions in parallel and filling the cache.
2. **Decode:** for each new token, run the model on **just that one token**: compute its q, k, v; append k and v to the cache; attend from its q to all cached keys; continue through the layer.

Two details:

- **Positions:** the new token's position embedding is at index T_past, not 0.
- **Masking:** when processing several new tokens at once (prefill), query i (at absolute position T_past + i) may attend to keys at positions ≤ T_past + i. With a single new token, it may attend to everything in the cache.

The outputs must be **exactly** the same as without a cache; that's your test. A forward hook confirms that after prefill the model only ever sees one token at a time.

**Memory:** the cache holds 2 (k and v) × layers × batch × sequence × d_model values. For a 7B Llama-2 (32 layers, d = 4096) in 16-bit, that's 0.5 MB per token: a 4,096-token conversation needs 2 GB of cache per sequence. That's why serving systems obsess over cache memory (PagedAttention, grouped-query attention, which shares K/V across heads, and cache quantization).

## 4. Quantization

Weights in 16-bit take 2 bytes each; a 70B model needs 140 GB just for weights. **Quantization** stores them in fewer bits.

**Symmetric absmax int8, per output channel (row):** for each row of W, scale = max|w| / 127; store q = round(w / scale) as int8 (−127..127) plus one float scale per row. Dequantize as q · scale. That's 4× smaller than float32 with a small error.

**Why per row, and why groups?** A single huge weight (an **outlier**) in a row forces a large scale, so all the small weights in that row round to 0. Smaller groups limit the damage. **Grouped int4** (as in GPTQ, AWQ and llama.cpp's Q4 formats) splits each row into groups of, say, 32 weights, each with its own scale: max|w| / 7, and q in −8..7. Four bits plus a small per-group overhead: about 7× smaller than float32.

The simplest way to use quantized weights: dequantize on the fly inside the layer (`x @ (q · scale)ᵀ + b`). Real kernels do the matmul directly on the packed integers, which is much faster on hardware that supports it.

How to evaluate quantization: compare the quantized model's outputs (or perplexity) with the original's. A good int8 quantization barely changes perplexity; naive int4 can hurt noticeably, grouped int4 much less.

## 5. Other speed tricks (to know about)

- **Batching** many users' requests together (continuous batching) keeps the GPU busy.
- **Speculative decoding:** a small model drafts several tokens and the big model verifies them in one pass; the output distribution is exactly the big model's.
- **FlashAttention** (m68) and fused kernels; **torch.compile** (m64).

---

## Problem-solving habit #71: an optimization must not change the answer

Every optimization in this module (except quantization, which deliberately trades accuracy) must produce **identical** outputs. So the first test of an optimized implementation is always equality with the slow, obviously correct one on random inputs, as in m61. Only after that do you measure speed. And for approximate optimizations like quantization, *measure* the error rather than assuming it's fine.

## Common mistakes

- Forgetting the position offset in cached decoding (every new token gets position 0).
- Applying top-k or top-p before dividing by the temperature (it changes which tokens survive top-p).
- A top-p filter that removes everything when the top token alone exceeds p.
- Quantizing with one scale for the whole matrix, so outliers wipe out the small weights.
- Counting the tied embedding/output matrix twice when measuring model size.

## Go deeper (optional, research-level)

1. Read *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2020). Reproduce their observation that greedy and beam search produce repetitive text, using your m70 model.
2. Read *LLM.int8()* (Dettmers et al., 2022). What are "emergent outlier features", and how does mixed-precision decomposition handle them?
3. Read *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2023) and implement it with two minigpt models of different sizes. Verify that the outputs follow the large model's distribution.

## Your turn

Open the **Exercises** tab. `minigpt.py` is the m69 model. Your KV-cache forward pass reuses its weights (`model.tok_emb`, `model.blocks[i].attn.qkv`, ...), so read `minigpt.py` first.
