# m67 · Language models: n-grams, perplexity and neural LMs

**By the end you can:** define a language model precisely, build count-based n-gram models with smoothing, evaluate any language model with **perplexity**, see the curse of sparsity in action, then build the neural alternatives: a neural bigram that rediscovers the count table, and Bengio's MLP language model with learned **embeddings**.

**Why it matters for AI:** an LLM is "just" a language model, scaled up. Everything about GPT (the training objective, the loss you watch, the way it generates text, how it's evaluated) is already present in the small models of this module. And the step from counting to learned embeddings is the conceptual leap that made modern NLP possible.

---

## 1. What a language model is

A language model assigns a probability to a sequence of tokens. By the chain rule of probability (m42), any such probability factorizes into next-token predictions:

P(x₁, …, x_T) = P(x₁) · P(x₂ | x₁) · P(x₃ | x₁, x₂) · … · P(x_T | x₁, …, x_{T−1})

So a language model only needs to answer one question: **given the context, what's the distribution of the next token?** Generating text is repeated sampling from that distribution (m62). Training is maximum likelihood (m43): minimize the average negative log-likelihood (NLL) of the true next tokens, which is the cross-entropy loss.

In this module the tokens are characters, so the vocabulary is the set of characters in the text.

## 2. n-gram models: counting

An n-gram model assumes the next token depends only on the previous n − 1 tokens:

P(xₜ | x₁…xₜ₋₁) ≈ P(xₜ | xₜ₋ₙ₊₁ … xₜ₋₁) = count(context, xₜ) / count(context)

A **bigram** (n = 2) uses only the previous token: a V × V table of counts, normalized row by row. A **unigram** (n = 1) ignores context entirely.

**Smoothing:** any pair never seen in training gets probability 0, and log(0) = −∞ on the test set. **Add-α smoothing** pretends every pair was seen α extra times:

P(b | a) = (count(a, b) + α) / (count(a) + α·V)

## 3. Perplexity

The standard metric for language models is **perplexity**: the exponential of the average NLL per token (m44).

perplexity = exp( −(1/N) Σ log P(xₜ | context) )

Intuition: a perplexity of 10 means the model is, on average, as uncertain as if it were choosing uniformly among 10 tokens. A uniform model over V tokens has perplexity exactly V; a perfect model has 1. Lower is better, and it's only comparable between models with the **same tokenizer** (a model with a bigger vocabulary predicts "bigger" tokens).

## 4. The curse of sparsity

On this course's own lessons (about 160,000 characters), you'll see something like:

| n | Train perplexity | Validation perplexity |
|---|---|---|
| 1 | 31 | 30 |
| 2 | 15 | 16 |
| 3 | 7 | 11 |
| 4 | 4 | 10 |
| 6 | 2.5 | 20 |

Longer contexts fit the training data better and better, but beyond n ≈ 4 validation perplexity gets *worse*: most 5-character contexts in new text were never seen in training, so the model falls back to near-uniform guesses. The number of possible contexts grows as Vⁿ⁻¹; the data doesn't. Counting can't generalize from "the cat sat" to "the dog sat", because it has no notion that cat and dog are similar.

## 5. Neural language models

**Neural bigram:** replace the count table with a learnable V × V table of logits (an `nn.Embedding(V, V)`: row a holds the logits for the token after a), and train it with cross-entropy and gradient descent. It converges to the same probabilities as counting (smoothing corresponds to a little weight decay). Same model, found by optimization instead of counting. That's the bridge: everything after this can be trained the same way.

**The MLP language model** (Bengio et al., 2003, *A Neural Probabilistic Language Model*):

```
context of k token ids ─▶ Embedding(V, d): each token -> a d-dim vector
                       ─▶ concatenate the k vectors: (k·d,)
                       ─▶ Linear + tanh (hidden layer)
                       ─▶ Linear to V logits ─▶ softmax over the next token
```

The key idea is the **embedding**: each token is a learned vector, and tokens used in similar contexts end up with similar vectors. That lets the model generalize to contexts it has never seen. With more context it beats the bigram's validation perplexity (and with enough training and capacity, all the n-grams).

The training data comes from sliding a window over the text: for every position t, the input is the k tokens before it and the target is token t.

## 6. Looking inside embeddings

After training, compare embedding vectors with **cosine similarity** (m38). In a character model, you'll often find the vowels near each other, digits near each other, and uppercase letters near their lowercase versions: structure the model discovered on its own because it helps predict the next character. In word-level models this gives the famous analogies (king − man + woman ≈ queen; Mikolov et al., 2013).

---

## Problem-solving habit #67: always have a baseline

Before celebrating a neural model's perplexity, compute what a simple baseline gets: uniform (V), unigram, bigram. A fancy model that barely beats a bigram is telling you something (too little training, a bug, or too little data). In research papers, a missing or weak baseline is one of the most common reasons for rejection.

## Common mistakes

- Comparing perplexities across different tokenizers or vocabularies.
- Evaluating on training data (n-grams look miraculous there).
- Splitting text randomly by character for validation: neighbouring windows overlap, so information leaks. Split by position (the first 90% for training, the last 10% for validation).
- Forgetting smoothing and getting infinite perplexity.

## Go deeper (optional, research-level)

1. Implement **Kneser-Ney smoothing** (the best classical n-gram smoothing) and compare it with add-α at n = 4 and 6. Read Chen & Goodman, *An Empirical Study of Smoothing Techniques for Language Modeling* (1998).
2. Read Bengio et al. (2003). Which of their ideas are still in GPT, and which were replaced?
3. Train word2vec-style embeddings with a skip-gram objective on this course's text (word level). Do "list", "tuple" and "dict" end up near each other?

## Your turn

Open the **Exercises** tab. The tests use this course's lessons as the corpus, so your models learn to write like your textbook.
