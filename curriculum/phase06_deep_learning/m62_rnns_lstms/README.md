# m62 · Recurrent networks and LSTMs

**By the end you can:** implement a vanilla RNN and an LSTM cell from scratch and match PyTorch's results exactly, measure the vanishing gradient problem yourself, clip gradients, and train a character-level language model that generates text with temperature sampling.

**Why it matters for AI:** before transformers, RNNs and LSTMs ran machine translation, speech recognition and text generation. More importantly, the **problems** they ran into (sequential computation that can't be parallelized, and information fading over long distances) are exactly what attention (Phase 7) was invented to solve. And the character-level language model you build here is a GPT in miniature: same task, same loss, same sampling.

---

## 1. Sequences and shared weights

A sequence x₁, x₂, …, x_T (words, characters, audio frames, prices) can have any length. An RNN reads it one step at a time, carrying a **hidden state** h that summarizes everything seen so far, and uses the **same weights at every step** (weight sharing again, this time across time instead of space):

hₜ = tanh(xₜ W_xh + hₜ₋₁ W_hh + b)

Shapes with batch-first data: x is (B, T, D), each xₜ is (B, D), h is (B, H), W_xh is (D, H), W_hh is (H, H). The outputs are all the hidden states stacked: (B, T, H).

PyTorch's `nn.RNN` computes the same thing with two biases and transposed weights: `x @ weight_ih_l0.T + bias_ih_l0 + h @ weight_hh_l0.T + bias_hh_l0`. Pass `batch_first=True` to use (B, T, D) inputs.

## 2. Backpropagation through time and vanishing gradients

Training **unrolls** the RNN over T steps and backpropagates through the whole chain: *backpropagation through time* (BPTT). The gradient from step T back to step t passes through T − t copies of the recurrence:

∂h_T/∂h_t = Π (diag(1 − hₖ²) · W_hhᵀ)  over k from t+1 to T

A product of many matrices either **shrinks exponentially** (if their norms are below 1: vanishing gradients) or **blows up** (above 1: exploding gradients). You'll measure it: in a default-initialized RNN, the gradient from the last output back to the first of 50 inputs is about **10⁻¹³** times the gradient to the last input. The network simply can't learn that the first input matters.

**Exploding gradients** have a simple fix: **gradient clipping**. If the total norm of all gradients exceeds `max_norm`, scale them all down to that norm:

```python
total = sqrt(sum(||p.grad||² for every parameter))
if total > max_norm: every p.grad *= max_norm / (total + 1e-6)
```

PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`. Transformer training uses it too (usually max_norm = 1.0). Vanishing gradients need an architectural fix.

## 3. The LSTM: a memory with gates

The **LSTM** (Hochreiter & Schmidhuber, 1997) adds a **cell state** c that is updated *additively*, controlled by gates (sigmoids between 0 and 1):

```
gates = xₜ W_x + hₜ₋₁ W_h + b        # (B, 4H), split into four (B, H) chunks: i, f, g, o
i = σ(·)   input gate: how much new information to write
f = σ(·)   forget gate: how much of the old cell to keep
g = tanh(·) candidate values to write
o = σ(·)   output gate: how much of the cell to expose
cₜ = f ⊙ cₜ₋₁ + i ⊙ g
hₜ = o ⊙ tanh(cₜ)
```

When f ≈ 1, the cell state passes through time almost unchanged, and so does its gradient: ∂cₜ/∂cₜ₋₁ = f. That additive path is a "gradient highway", the same idea as the residual connections you'll use in m65 and in every transformer. A common trick is to initialize the forget-gate bias to 1 or more, so the LSTM remembers by default.

PyTorch orders the gates **i, f, g, o** in its weight matrices (`weight_ih_l0` has shape (4H, D)). The **GRU** is a popular simplification with two gates and no separate cell state.

## 4. Character-level language modelling

A **language model** predicts the next token given the previous ones. With characters as tokens:

- Build a vocabulary (the sorted unique characters) and map each to an integer.
- Cut the text into chunks: the input is characters [k, k+T), the target is [k+1, k+T+1), the same text shifted by one.
- Model: `nn.Embedding` (a learned vector per character; m44's one-hot times a matrix, done efficiently) → `nn.LSTM` → `nn.Linear` to vocabulary-size logits at every position.
- Loss: cross-entropy over every position. Flatten (B, T, V) logits to (B·T, V) and targets to (B·T,).

This is **exactly** how GPT is trained (Phase 7), with a transformer instead of an LSTM and subword tokens instead of characters.

## 5. Generating text: sampling and temperature

To generate, feed a prompt, take the logits for the next character, pick one, feed it back in, repeat. Carry the LSTM state forward so you don't recompute the whole history each step (the RNN version of the KV cache you'll meet in Phase 7).

- **Greedy** (temperature 0): always take the argmax. Deterministic, often repetitive.
- **Sampling** with **temperature** τ: sample from softmax(logits / τ). τ < 1 sharpens the distribution (safer), τ > 1 flattens it (more creative, more mistakes).

---

## Problem-solving habit #62: measure the thing the theory predicts

The math in section 2 says gradients vanish exponentially. Don't just believe it: measure ∂output/∂input at every time step and plot it. Turning a theoretical claim into a measurement is the core move of empirical research, and it often reveals that the theory only holds under conditions you didn't notice.

## Common mistakes

- Mixing up (B, T, D) and (T, B, D): PyTorch's RNNs default to time-first unless `batch_first=True`.
- Forgetting to flatten logits and targets for `cross_entropy` (it expects classes in dimension 1).
- Sampling from logits instead of probabilities, or forgetting to divide by the temperature before the softmax.
- Recomputing the whole sequence at each generation step instead of carrying the state.

## Go deeper (optional, research-level)

1. Read Andrej Karpathy's blog post *The Unreasonable Effectiveness of Recurrent Neural Networks* (2015) and train your `CharLSTM` on a larger text (a public-domain book from Project Gutenberg). What does it learn first: spelling, words, or punctuation?
2. Read *On the difficulty of training recurrent neural networks* (Pascanu et al., 2013). Relate their condition on the largest singular value of W_hh to what you measured.
3. Recent "linear RNNs" and state-space models (S4, Mamba) are competitive with transformers on long sequences while training in parallel. Read the Mamba paper's introduction (Gu & Dao, 2023). What problem of classic RNNs do they solve, and how?

## Your turn

Open the **Exercises** tab. The tests check your cells against `nn.RNN` and `nn.LSTMCell`, then train your character model on a small text.
