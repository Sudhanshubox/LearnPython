# m70 · Train your own GPT

**By the end you can:** write the full pretraining pipeline for a GPT: a token dataset, random-window batching, AdamW with decoupled weight decay groups, warmup and cosine decay, gradient clipping, periodic evaluation, and generation. You'll train a small GPT on this course's own lessons and watch it learn to write (badly, then less badly) like your textbook.

**Why it matters for AI:** this is pretraining, the step that creates every foundation model. The code is essentially Karpathy's nanoGPT, which can reproduce GPT-2 (124M) on a GPU node. Scaled up thousands of times, with better data and more care, it's how GPT-4 and Claude start life. You're writing the real thing at toy scale.

---

## 1. The model is given

`minigpt.py` in this folder is the GPT you built in m69 (`GPT`, `GPTConfig`). Read it once: it should all look familiar. This module is about everything *around* the model.

## 2. Data: one long stream of tokens

Pretraining data is just text. Concatenate all documents into one long sequence of token ids (here: characters of this course's lessons; in real life, BPE tokens from m66 over trillions of tokens of web text, books and code). Split by position: the first 90% for training, the last 10% for validation.

A **batch** is B random windows of length T + 1 from the stream:

```python
starts = torch.randint(0, len(data) - T, (B,))
x = data[starts[:, None] + torch.arange(T)]          # (B, T)
y = data[starts[:, None] + torch.arange(T) + 1]      # the same windows shifted by one
```

Each window gives T training examples at once: predict y[t] from x[0..t] for every t, thanks to the causal mask. Random windows mean there are no fixed "epochs"; you count **steps** (or tokens seen).

## 3. The optimizer recipe

The GPT-2/GPT-3 recipe, used almost unchanged by most LLMs:

- **AdamW** with β = (0.9, 0.95) (a shorter memory for the second moment makes training more stable at scale).
- **Weight decay 0.1**, but only on matrices (2-D parameters), not on biases, norms or... embeddings? Embeddings are 2-D, so they're decayed here, as in nanoGPT (m60).
- **Learning rate:** linear warmup, then cosine decay to a minimum of about max_lr / 10 (m59). Set the lr on every param group by hand each step: `for g in opt.param_groups: g["lr"] = lr`.
- **Gradient clipping** at a global norm of 1.0 (m62), which protects against rare loss spikes.

## 4. Evaluating during training

The training loss on each batch is noisy. Every `eval_interval` steps, estimate the loss on the **validation** data by averaging over several random batches, in `eval()` mode, without gradients. Watch two things:

- Is validation loss still falling? If it rises while training loss falls, you're overfitting: you need more data (or a smaller model, or dropout).
- How does it compare with baselines? For this character-level data, a bigram model gets about 2.78 nats per character (m67); your GPT should get clearly below that within a few hundred steps.

Loss is in **nats per token**. Perplexity = e^loss (m67). For comparisons across tokenizers, convert to **bits per byte**: loss / ln(2) × tokens / bytes.

## 5. Generation

To sample, feed a prompt, take the logits at the **last** position, divide by the temperature, optionally keep only the **top-k** most likely tokens (set the rest to −inf), sample, append, repeat. If the sequence grows beyond `block_size`, keep only the last `block_size` tokens as context (the model has no position embeddings beyond that). m71 makes this much faster and adds better sampling methods.

## 6. What to expect

On a laptop CPU, a 2-layer, 64-dimensional GPT trains at about 30 steps per second. After a few hundred steps it produces word-like gibberish with the right "texture": Markdown headings, backticks, plausible English fragments, code-like snippets. Bigger models and longer training produce real words and then grammatical sentences. That progression (characters → words → syntax → meaning) is what scaling buys, and m72 measures it.

To go further, use a GPU (`device="cuda"`), BPE tokens, `torch.compile`, bfloat16 autocast (m64), and a bigger corpus. nanoGPT's `train.py` is worth reading line by line now; you'll recognize every part.

---

## Problem-solving habit #70: watch the right curves

Log the train loss, validation loss, learning rate and gradient norm at every step (or every few), and plot them. Most training problems are visible in these four curves long before the final number: a loss spike paired with a gradient-norm spike, a learning rate that decays too early, validation loss turning up. Experienced practitioners stare at curves constantly.

## Common mistakes

- Targets not shifted by one (the model learns to copy its input and the loss goes to 0).
- Sampling windows that run off the end of the data (start indices must be < len(data) − T).
- Evaluating on a single batch (too noisy to compare runs).
- Forgetting to put the model back into `train()` mode after evaluation.
- Generating with more context than `block_size`.

## Go deeper (optional, research-level)

1. Read nanoGPT's `train.py` and `model.py` (github.com/karpathy/nanoGPT), then watch "Let's reproduce GPT-2 (124M)". List three things it does that this module doesn't, and why they matter at scale.
2. Train the same model with the m66 BPE tokenizer instead of characters. Compare in bits per byte, not loss. Which is better at equal compute?
3. Read *Language Models are Few-Shot Learners* (GPT-3, Brown et al., 2020), section 2 and appendix B. How do their training hyperparameters compare with yours?

## Your turn

Open the **Exercises** tab. The tests train a tiny GPT for 600 steps (about 20 seconds on a CPU) and check it beats a bigram model. Then try a bigger model and more steps yourself, and read what it writes.
