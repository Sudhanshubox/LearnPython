# m87 · Designing experiments

**By the end you can:** turn a vague idea into a testable hypothesis with a decision rule written down *before* running anything, choose baselines and controls, generate configuration grids and ablations, decide how many seeds you need with a power calculation, run an experiment across seeds, and reach a conclusion (supported, not supported, or inconclusive) that you'd defend in a paper.

**Why it matters for AI:** in ML research, the experiment *is* the argument. A clever idea with a sloppy experiment proves nothing; a simple idea with a careful experiment can change how people work. You've run many experiments in this course (m65, m72, m73, m75). This module turns the habits you picked up along the way into a method.

---

## 1. Start from a question, not a method

Bad: "I'll try adding feature scaling." Good: **"Does standardizing the features improve logistic regression's test accuracy on handwritten digits by at least 1 percentage point?"** A good research question names:

- the **intervention** (standardizing features) and the **control** (the same pipeline without it),
- the **metric** (test accuracy) and the **population** (the digits dataset, this model),
- the **smallest effect you care about** (1 point). Anything smaller isn't worth the complexity, even if it's "real".

Write it down **before** running the experiment, together with the decision rule. This is **pre-registration**. It protects you from the most common self-deception in research: running many variants, noticing the one that looks good, and telling a story afterwards ("garden of forking paths").

## 2. Baselines and controls

- **Trivial baselines:** majority class, random guessing, "copy the input". If your model barely beats predicting the most common class, you need to know.
- **Strong, simple baselines:** logistic regression, gradient boosting, the previous best method, **tuned as carefully as your method**. Weak baselines are the most common flaw in published ML.
- **Controls:** change **one thing at a time**. Everything else (data split, preprocessing, training budget, seeds) stays identical between treatment and control.

## 3. Grids and ablations

- A **grid** is the Cartesian product of the settings you vary: `{"model": ["logreg", "tree"], "scale": [True, False]}` → 4 configurations. Grids grow fast (5 settings × 4 values = 1,024 runs), so vary only what matters; for many hyperparameters, random search beats grids.
- An **ablation** removes one component of a method at a time, to show each part contributes: `full`, `no_scale`, `no_augmentation`, ... (m65's zero-init experiment was an ablation).
- Give every configuration a stable **ID** (a hash of its sorted settings), so results from different runs can be matched up later (m88).

## 4. How many seeds?

Random seeds change the data split, initialization and data order. Results vary between seeds, so one run per configuration can't separate a real effect from luck. A **power analysis** says how many seeds you need to detect an effect of size δ when results vary with standard deviation σ (two-sided test at level α with power 1 − β, normal approximation):

n per group = 2 · ((z₁₋α/₂ + z₁₋β) · σ / δ)²

With α = 0.05 and 80% power, z-values are 1.96 and 0.84. To detect a 1-point effect when seeds vary by 1 point, you need about 16 seeds per configuration; for a 2-point effect, about 4. Estimate σ from a quick pilot run with a few seeds.

## 5. Running and deciding

Run every configuration on the **same** seeds (paired design, m82), record everything (m88), then summarize each configuration's mean, standard deviation and number of runs. A simple, honest decision rule:

- **supported** if the improvement minus 2 standard errors still reaches the minimum effect,
- **not supported** if the improvement plus 2 standard errors stays below it,
- **inconclusive** otherwise: you need more seeds (go back to section 4), or the effect is too small to matter.

In the exercises you'll test hypotheses on handwritten digits with 5 seeds. "10× more training data improves accuracy by at least 5 points" is clearly supported. "Standardizing features improves logistic regression by at least 2 points" is *not* supported, and reporting that is just as valuable as a positive result: negative results save other people time. And "...by at least 1 point" comes out **inconclusive**: 5 seeds simply can't resolve an effect that small, exactly as the power calculation in section 4 predicts.

---

## Problem-solving habit #87: decide what would change your mind

Before running an experiment, write down which result would make you believe the hypothesis, and which would make you drop it. If no possible result would change what you do next, the experiment isn't worth running. This turns experiments from confirmation rituals into decisions.

## Common mistakes

- Choosing the metric, the threshold or the seeds after seeing the results.
- Comparing your tuned method with an untuned baseline.
- Changing two things at once, then crediting one of them.
- One seed per configuration.
- Reporting only the configurations that worked.
- Testing on the test set repeatedly during development (it becomes a training set, m49).

## Go deeper (optional)

1. Read *Deep Reinforcement Learning that Matters* (Henderson et al., 2018). How much did results change with random seeds and implementation details alone?
2. Read *Hyperparameter Optimization* chapters or *Random Search for Hyper-Parameter Optimization* (Bergstra & Bengio, 2012). Why does random search beat grid search when only a few hyperparameters matter?
3. Pre-register your capstone's main experiment (m92) using section 1's template, before writing any code for it.

## Your turn

Open the **Exercises** tab. The experiment trains small scikit-learn models on the digits dataset, so everything runs in a few seconds.
