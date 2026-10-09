# m42 · Probability

**By the end you can:** reason about random events with probability rules and Bayes' theorem, work with discrete and continuous distributions, compute expectations and variances, estimate anything by simulation (Monte Carlo), and see the law of large numbers and the central limit theorem happen in code.

**Why it matters for AI:** a classifier outputs a probability distribution over classes, and a language model outputs one over the next token. Generating text means **sampling** from that distribution, with **temperature** controlling how adventurous it is. Training objectives come from probability (maximum likelihood, m43). Dropout, data augmentation, weight initialization and SGD itself are random processes. Uncertainty is everywhere in AI, and probability is the language for it.

---

## 1. Events and rules

A probability is a number between 0 and 1 assigned to events:

- P(not A) = 1 − P(A)
- P(A or B) = P(A) + P(B) − P(A and B)
- **Conditional probability:** P(A | B) = P(A and B) / P(B), "the probability of A once we know B happened".
- **Independence:** P(A and B) = P(A) P(B), knowing one tells you nothing about the other.

## 2. Bayes' theorem

P(H | E) = P(E | H) · P(H) / P(E)

It updates a **prior** belief P(H) into a **posterior** P(H | E) after seeing evidence E.

**Classic example.** A disease affects 1% of people. A test catches 99% of cases (sensitivity) and has a 5% false-positive rate. You test positive. How likely is it that you're sick?

P(positive) = 0.99 × 0.01 + 0.05 × 0.99 = 0.0594
P(sick | positive) = 0.99 × 0.01 / 0.0594 ≈ **16.7%**

Most people guess around 95%. When the base rate is low, most positives are false positives. The same logic applies to fraud detectors and content-moderation classifiers.

## 3. Random variables and distributions

A **random variable** maps outcomes to numbers. Its **distribution** says how likely each value is.

| Distribution | Models | In NumPy (`rng = np.random.default_rng()`) |
|---|---|---|
| Bernoulli(p) | one yes/no trial | `rng.random() < p` |
| Binomial(n, p) | number of successes in n trials | `rng.binomial(n, p)` |
| Categorical(p₁…pₖ) | one of k classes (next token!) | `rng.choice(k, p=probs)` |
| Uniform(a, b) | anything equally likely in [a, b] | `rng.uniform(a, b)` |
| Normal(μ, σ) | sums of many small effects; initial weights | `rng.normal(mu, sigma)` |

Continuous distributions have a **density** (pdf) instead of probabilities for single values. The normal density is

p(x) = 1 / (σ√(2π)) · exp(−(x − μ)² / (2σ²))

and areas under the density give probabilities.

## 4. Expectation and variance

- **Expected value:** E[X] = Σ xᵢ P(xᵢ), the long-run average. A fair die: 3.5.
- **Variance:** Var[X] = E[(X − E[X])²], how spread out. **Standard deviation** = √Var, in the original units.
- E is linear: E[aX + bY] = a E[X] + b E[Y], always. For *independent* variables, variances add.

The **sample** variance divides by n − 1 instead of n (`np.var(x, ddof=1)`) to correct for estimating the mean from the same data.

## 5. Monte Carlo: estimate by simulating

When a probability is hard to compute, simulate the process many times and count:

```python
rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=(100_000, 2))
p_seven = (rolls.sum(axis=1) == 7).mean()       # ≈ 1/6
```

The error shrinks like 1/√n: 100× more samples give one more correct digit.

## 6. Two great theorems

**Law of large numbers:** the average of many independent samples approaches the expected value. This is why Monte Carlo works, and why more data gives better estimates.

**Central limit theorem:** the average of n independent samples (from almost any distribution with finite variance) is approximately **normal**, with mean μ and standard deviation σ/√n. That's why normal distributions are everywhere, and why error bars shrink like 1/√n.

## 7. Sampling from a distribution: how LLMs choose words

Given probabilities p₁…pₖ, draw u uniformly from [0, 1) and pick the first index where the **cumulative** sum exceeds u (inverse-CDF sampling). That's what `rng.choice` does, and what happens at every step of text generation.

**Temperature** reshapes the distribution before sampling: pᵢ ∝ exp(zᵢ / T) for logits zᵢ.

- T → 0: always the most likely token (greedy, deterministic, often repetitive).
- T = 1: the model's own distribution.
- T > 1: flatter, more random, more "creative" (and more mistakes).

---

## Problem-solving habit #39: simulate to check your math

Every probability calculation can be checked by simulation in a few lines. If your formula says 0.167 and 100,000 simulated trials give 0.31, one of them is wrong, and finding out which teaches you something. Do this for every exercise here.

## Go deeper (optional, research-level)

1. The **Monty Hall problem**: simulate it and explain the result with Bayes' theorem.
2. Read section 3 of *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2019) on why maximizing probability produces bland, repetitive text, and how top-p (nucleus) sampling works. Implement top-p using your `sample_categorical`.
3. Why does SGD's mini-batch gradient estimate have variance proportional to 1/batch_size? Connect it to section 6, and read about the "linear scaling rule" for learning rates in *Accurate, Large Minibatch SGD* (Goyal et al., 2017).

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Always use the `rng` you're given (or create one from the seed), so results are reproducible.
