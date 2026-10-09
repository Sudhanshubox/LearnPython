# m44 · Information theory

**By the end you can:** measure information and uncertainty with entropy, explain cross-entropy and KL divergence and why they're used as losses, compute them stably with the log-sum-exp trick, and evaluate language models with perplexity.

**Why it matters for AI:** cross-entropy is *the* loss function for classification and language modelling. Perplexity is how language models are compared. KL divergence appears in knowledge distillation, variational autoencoders, and RLHF (keeping a fine-tuned model close to the original). And the idea that "a better model is a better compressor" (m22's Huffman coding) connects all of it.

---

## 1. Information and surprise

A rare event carries more information than a common one. The **information content** (surprise) of an outcome with probability p is

I(p) = −log₂ p bits

A fair coin flip: 1 bit. Rolling a 1 on a fair die: log₂ 6 ≈ 2.58 bits. A certain event: 0 bits.

## 2. Entropy

**Entropy** is the *average* surprise of a distribution:

H(p) = −Σ pᵢ log pᵢ    (with 0 · log 0 = 0)

- A fair coin: 1 bit. A biased coin (0.9 / 0.1): 0.47 bits, more predictable.
- Uniform over k outcomes: log₂ k, the maximum possible.
- Shannon's theorem: H is the minimum average number of bits per symbol needed to encode messages from p. Huffman coding (m22) gets within 1 bit of it.

In ML we often use natural logs (units called **nats**); divide by ln 2 to convert to bits.

## 3. Cross-entropy

If the true distribution is p but you encode (or predict) using q, the average cost is the **cross-entropy**:

H(p, q) = −Σ pᵢ log qᵢ

It's always ≥ H(p), with equality only when q = p. For a classifier, p is the one-hot true label, so H(p, q) = −log q(true class): exactly the NLL from m43. **Training a classifier with cross-entropy means making its predicted distribution q as close to the data's as possible.**

## 4. KL divergence

KL(p ‖ q) = Σ pᵢ log(pᵢ / qᵢ) = H(p, q) − H(p)

The *extra* cost of using q instead of the true p. It's ≥ 0, equals 0 only when p = q, and it's **not symmetric**: KL(p‖q) ≠ KL(q‖p). If q gives near-zero probability where p doesn't, KL(p‖q) explodes. That's why you clip or smooth probabilities.

Where it appears: distillation (make a small student's outputs match a big teacher's), VAEs, and RLHF/PPO's penalty for drifting too far from the reference model.

## 5. Computing it stably: log-sum-exp

Models output **logits** z, and softmax turns them into probabilities. Computing log(softmax(z)) naively overflows or takes log(0). Use

log softmax(z)ᵢ = zᵢ − logsumexp(z),   logsumexp(z) = m + log Σ exp(zᵢ − m),   m = max(z)

Subtracting the max keeps every exponent ≤ 0 (m37's trick again). Frameworks provide `log_softmax` and `cross_entropy(logits, labels)` that do exactly this. Never compute `log(softmax(z))` in two separate steps.

## 6. Perplexity

A language model assigns a probability to each next token. Its average cross-entropy on a test text, in nats per token, is H. **Perplexity** is

PPL = exp(H)

Intuition: the model is "as confused as if it were choosing uniformly among PPL tokens". A uniform guess over a 50,000-token vocabulary has perplexity 50,000; good LLMs reach single digits on typical text. Lower is better, but only compare perplexities measured with the same tokenizer (m02, Phase 7).

## 7. Mutual information

I(X; Y) = H(X) − H(X | Y): how much knowing Y reduces uncertainty about X. It's 0 for independent variables. It's used for feature selection, and in research on what neural representations encode.

---

## Problem-solving habit #41: sanity-check against the uniform baseline

For any classifier or language model, compute the loss of the dumbest model first: predicting uniformly gives cross-entropy ln(k). At initialization, a well-set-up network should start near that value. If your initial loss is far above ln(k), something is wrong (bad initialization, wrong labels); if training ends near it, the model learned nothing.

## Go deeper (optional, research-level)

1. Read Shannon's *A Mathematical Theory of Communication* (1948), section 6 ("Choice, Uncertainty and Entropy"). Which properties did he require entropy to have, and why do they force the −Σ p log p formula?
2. *Distilling the Knowledge in a Neural Network* (Hinton et al., 2015) trains a student on the teacher's softened probabilities (temperature, m42). Why do the "wrong" classes' small probabilities carry useful "dark knowledge"?
3. Compute the bits per character of a simple character-level bigram model (m06) on a text, and compare with gzip's compression ratio on the same text. What does *Language Modeling Is Compression* (Delétang et al., 2023) conclude?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Use natural logarithms unless a function says bits.
