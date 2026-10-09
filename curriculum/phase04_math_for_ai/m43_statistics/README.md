# m43 · Statistics: estimation, uncertainty and testing

**By the end you can:** fit distributions by maximum likelihood, put honest error bars on any number with the bootstrap, test whether a difference is real with a permutation test, and decide whether one model is genuinely better than another.

**Why it matters for AI:** the cross-entropy loss you'll minimize in every classifier and language model *is* a negative log-likelihood, so training is maximum likelihood estimation. And research is full of claims like "our method improves accuracy from 81.2% to 81.9%". Is that real, or noise from a small test set and a lucky random seed? Being able to answer that separates research from wishful thinking.

---

## 1. Maximum likelihood estimation (MLE)

Given data and a model with parameters θ, the **likelihood** is the probability of the observed data under θ. MLE picks the θ that makes the data most probable. We maximize the **log**-likelihood (sums are easier than products, and they don't underflow):

ℓ(θ) = Σᵢ log p(xᵢ | θ)

Classic results (each is "set the derivative to 0", m40):

| Model | MLE |
|---|---|
| Bernoulli (coin with P(heads) = p) | p̂ = fraction of heads |
| Normal(μ, σ) | μ̂ = sample mean, σ̂ = √(mean of (x − μ̂)²) (divides by n) |

**The connection to deep learning:** a classifier outputs probabilities pᵢ(class). The negative log-likelihood of the true labels is

NLL = −(1/n) Σᵢ log pᵢ(yᵢ)

which is exactly the **cross-entropy loss**. Minimizing cross-entropy *is* maximum likelihood. Similarly, minimizing MSE is MLE under Gaussian noise.

## 2. Uncertainty: standard errors and confidence intervals

An estimate from a sample is itself random: a different sample would give a different number. The **standard error** of a mean is s/√n (sample std over √n, the CLT from m42). An approximate 95% **confidence interval** is

mean ± 1.96 × SE

For an accuracy p measured on n test examples, SE = √(p(1 − p)/n). With 1,000 test examples and 80% accuracy, SE ≈ 1.3%, so the 95% interval is about 77.5–82.5%. A 0.7% "improvement" on that test set is well within the noise!

## 3. The bootstrap: error bars for anything

What's the standard error of a median, an F1 score, or a BLEU score? Usually there's no formula. The **bootstrap** (Efron, 1979) simulates new samples by **resampling your data with replacement**:

1. Draw n items from your n data points, with replacement. Compute the statistic.
2. Repeat B times (1,000–10,000).
3. The 2.5th and 97.5th percentiles of the B results give a 95% interval.

```python
idx = rng.integers(0, n, size=(B, n))     # B resamples at once
stats = np.median(x[idx], axis=1)
low, high = np.percentile(stats, [2.5, 97.5])
```

## 4. Hypothesis testing with permutations

Group A has a higher average than group B. Could that happen by chance even if there's no real difference?

The **null hypothesis** says the group labels don't matter. If that's true, shuffling the labels shouldn't change anything. So:

1. Compute the observed difference in means.
2. Shuffle the labels many times, recomputing the difference each time.
3. The **p-value** is the fraction of shuffles with a difference at least as extreme as the observed one.

A small p-value (say < 0.05) means "this would rarely happen by chance". It does **not** mean the effect is large or important, and with many comparisons, some will look "significant" by luck (the **multiple comparisons** problem).

## 5. Comparing two models properly

When two models are evaluated on the **same** test examples, compare them **paired**: per example, did A get it right, did B? Bootstrap the *difference* in accuracy by resampling examples (the same resampled indices for both models). If the 95% interval of the difference excludes 0, the improvement is probably real for this kind of data.

Also: run several **random seeds**. Deep learning results can vary by more between seeds than between methods.

## 6. Correlation is not causation

Pearson correlation r = cov(x, y) / (σₓ σᵧ) measures *linear* association, between −1 and 1. Two variables can be correlated because one causes the other, because a third causes both, or by chance (try correlating random walks). Data leakage in ML often hides behind a suspiciously strong correlation.

---

## Problem-solving habit #40: always ask "compared to what, and how sure?"

Every number needs a baseline (what does a trivial method get?) and an uncertainty (what would it be on a different sample?). Report "81.9% ± 1.2% (95% CI), baseline 50%" instead of "81.9%". This habit alone will make your work more trustworthy than much published research.

## Go deeper (optional, research-level)

1. Read *Deep Reinforcement Learning that Matters* (Henderson et al., 2018), sections on random seeds and statistical significance. How much did results vary between seeds?
2. Read *Show Your Work: Improved Reporting of Experimental Results* (Dodge et al., 2019). What is "expected validation performance" and why does tuning budget matter when comparing methods?
3. The bootstrap can fail: for example, when estimating the *maximum* of a distribution. Simulate it and explain why.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Use the seeds you're given, and vectorize the resampling.
