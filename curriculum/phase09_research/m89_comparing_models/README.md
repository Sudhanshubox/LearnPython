# m89 · Statistics for comparing models

**By the end you can:** choose and run the right test for the comparison in front of you (paired t-test across seeds, Welch's t-test for unpaired runs, McNemar's test for two classifiers on one test set, a permutation test when you don't want to assume normality), report effect sizes alongside p-values, and correct for testing many things at once. You'll also see by simulation why uncorrected multiple testing finds "significant" results in pure noise.

**Why it matters for AI:** "Model B beats model A" is a statistical claim. Many published ML improvements are within the noise of random seeds or test-set sampling. m43 taught you the foundations and m82 the bootstrap. This module gives you the specific tools researchers use to back up comparisons, and the judgement to know what a p-value does and doesn't mean.

---

## 1. Which test?

| Situation | Test |
|---|---|
| Two methods, each run with the **same** seeds (or folds, or datasets) | **Paired t-test** on the per-seed differences |
| Two methods, runs **not** matched up, possibly different variances | **Welch's t-test** |
| Two classifiers evaluated on the **same test examples** | **McNemar's test** on the examples where they disagree |
| Small samples or non-normal differences | **Permutation test** (randomization test) |
| Many comparisons at once | Any of the above, then **Holm–Bonferroni** correction |

Pairing matters: when both methods see the same seeds, the seed-to-seed variation (some splits are just easier) cancels out in the differences, making the test far more sensitive (m82's paired bootstrap, now as a classical test).

## 2. The tests

**Paired t-test.** With differences dᵢ = bᵢ − aᵢ over n seeds: t = mean(d) / (sd(d) / √n), compared with a t distribution with n − 1 degrees of freedom; the two-sided p-value is 2 · P(T > |t|).

**Welch's t-test.** t = (mean(b) − mean(a)) / √(s²_a/n_a + s²_b/n_b), with the Welch–Satterthwaite degrees of freedom df = (v_a + v_b)² / (v_a²/(n_a−1) + v_b²/(n_b−1)), where v = s²/n. It doesn't assume equal variances (use it instead of Student's t by default).

**McNemar's test.** Count the test examples that only A gets right (b) and that only B gets right (c). Examples both get right or both get wrong say nothing about which is better. Under "no difference", each disagreement is a fair coin flip, so test b against Binomial(b + c, ½) with an exact binomial test. This uses the *examples* as the unit of variation, so a single run of each model is enough to test the test-set uncertainty (it doesn't cover seed variation; you need both for a strong claim).

**Paired permutation test.** If there's no difference, each dᵢ is equally likely to be positive or negative. Randomly flip the signs thousands of times and see how often the shuffled mean is at least as extreme as the real one: p = (count + 1) / (permutations + 1). No normality assumption.

## 3. Effect sizes

A p-value says how surprising the data would be if there were no difference. It doesn't say how *big* the difference is: with enough seeds, a 0.01-point improvement becomes "significant". Always report the **effect size** too: the raw difference with a confidence interval (in the metric's own units, the most useful form), and/or a standardized one like **Cohen's d** for paired data, mean(d) / sd(d). Then decide whether the difference *matters* (m87's minimum effect).

## 4. Many tests: the multiple-comparisons trap

Test 20 configurations against a baseline at α = 0.05 when **none** is truly better, and the chance that at least one looks "significant" is 1 − 0.95²⁰ ≈ 64%. You'll reproduce this by simulation. The **Holm–Bonferroni** method fixes it: sort the m p-values; compare the smallest with α/m, the next with α/(m−1), and so on; reject until the first one that fails, then stop. It controls the chance of *any* false positive at α, and is uniformly better than plain Bonferroni (α/m for all).

This is also why p-hacking is so dangerous: trying many metrics, subsets, seeds or variants and reporting the one that "worked" is multiple testing in disguise. m87's pre-registration is the cure.

## 5. What a p-value is not

- Not the probability that the hypothesis is true.
- Not the probability the result is a fluke.
- Not a measure of importance (see effect sizes).
- p = 0.06 vs p = 0.04 is not a meaningful difference between "no effect" and "effect".

Report the estimate, its uncertainty, and the test; let readers judge.

---

## Problem-solving habit #89: ask "compared to what, and how sure?"

Every comparison needs a reference and an uncertainty. When you see "X is better", ask: better than what baseline, by how much, measured how many times, with what spread? When you write results, answer those questions before anyone asks.

## Common mistakes

- An unpaired test on paired data (throws away most of the sensitivity).
- Comparing two classifiers' accuracies with a t-test over test examples instead of McNemar.
- Reporting "p < 0.05" without the size of the difference.
- Running 30 comparisons and celebrating the 2 that came out significant.
- Treating "not significant" as "no difference" (it might just be too few seeds, m87).

## Go deeper (optional)

1. Read *Statistical Comparisons of Classifiers over Multiple Data Sets* (Demšar, 2006), the standard reference for comparing methods across many datasets (the Wilcoxon and Friedman tests).
2. Read *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms* (Dietterich, 1998). Why does he recommend McNemar's test and the 5×2 cross-validation t-test?
3. Read the American Statistical Association's 2016 statement on p-values (six short principles). Rewrite a results paragraph from one of your projects to follow them.

## Your turn

Open the **Exercises** tab. The tests check your tests against SciPy's implementations and on real classifiers.
