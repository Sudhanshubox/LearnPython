# m49 · The ML workflow and evaluation

**By the end you can:** split data correctly (train/validation/test, stratified, k-fold), choose and compute the right metric (accuracy, precision, recall, F1, ROC AUC, MAE, RMSE, R²), compare against baselines, and recognize data leakage before it fools you.

**Why it matters for AI:** a model is only as good as the evidence that it works. Most embarrassing ML failures, from papers whose results didn't replicate to production models that collapsed on real users, come from evaluation mistakes: testing on data the model had seen, picking the wrong metric, or tuning on the test set. Getting evaluation right is what makes your results believable.

---

## 1. The workflow

1. **Define the task and the metric** before touching models. What does success mean for the user?
2. **Split** the data into train / validation / test.
3. **Baseline:** what does a trivial model score (always predict the majority class, or the mean)?
4. **Train** on the training set; **choose** models and hyperparameters on the validation set (or with cross-validation).
5. **Test once**, at the end, on the untouched test set. Report that number.

## 2. Splits

- **Training set:** the model learns from it.
- **Validation set:** you make decisions with it (which model, which hyperparameters, when to stop).
- **Test set:** an honest final estimate. If you look at it repeatedly and adjust, it becomes a second validation set and your reported score becomes optimistic.

**Stratified** splitting keeps class proportions the same in every split. That's essential when classes are imbalanced: with 2% fraud, a random split could put almost no fraud in the test set.

**k-fold cross-validation** splits the data into k folds, trains k times (each time validating on a different fold), and averages. It uses all the data for validation and shows how much the score varies, at k times the cost.

## 3. Classification metrics

For a binary problem, count the four outcomes (m03's exercise!): TP, FP, TN, FN.

| Metric | Formula | Question it answers |
|---|---|---|
| accuracy | (TP + TN) / all | How often is it right? (Misleading with imbalanced classes!) |
| precision | TP / (TP + FP) | When it says "positive", how often is it right? |
| recall | TP / (TP + FN) | Of the real positives, how many did it find? |
| F1 | 2PR / (P + R) | A balance of precision and recall |

Precision and recall trade off: lowering the decision threshold finds more positives (recall up) but with more false alarms (precision down). Which one matters depends on the cost of each mistake: missing a disease (recall) vs. bothering healthy patients (precision).

**ROC AUC** measures how well *scores* rank positives above negatives, across all thresholds: it's the probability that a random positive gets a higher score than a random negative. 0.5 = random, 1.0 = perfect. It can be computed from **ranks** (the Mann–Whitney U statistic): sort all scores, and AUC = (sum of the positives' ranks − n₊(n₊ + 1)/2) / (n₊ n₋), with ties given their average rank.

## 4. Regression metrics

- **MAE:** mean |y − ŷ|, in the target's units, robust to outliers.
- **RMSE:** √(mean (y − ŷ)²), penalizes large errors more.
- **R²:** 1 − SS_res / SS_tot, the fraction of variance explained. 0 = no better than predicting the mean; can be negative for terrible models.

## 5. Baselines

Always compute a baseline: majority class for classification, the training mean for regression, or a simple heuristic. "92% accuracy" sounds great until you learn that 91% of emails aren't spam.

## 6. Data leakage

**Leakage** is when information from outside the training data (often from the test set or the future) gets into training, so test scores look better than reality. Classic forms:

- **Duplicates** across train and test (m47).
- **Preprocessing on all the data** before splitting: e.g. fitting a scaler or imputer on the full dataset lets test statistics leak into training. Fit preprocessing on the training set only (m53's pipelines make this automatic).
- **Target leakage:** a feature that's only available *because* of the outcome (e.g. "refund_issued" when predicting fraud).
- **Time leakage:** randomly splitting time-ordered data, so the model trains on the future and is tested on the past. Split by time instead.
- **Group leakage:** several samples from the same patient or user in both train and test. Split by group.

---

## Problem-solving habit #45: if it's too good to be true, look for leakage

When a model scores far above what you expected, don't celebrate yet. Check for duplicates across splits, look at the most important features, and ask whether each feature would really be available at prediction time. Suspiciously good results are usually leakage, not genius.

## Go deeper (optional, research-level)

1. Read *Leakage and the Reproducibility Crisis in ML-based Science* (Kapoor & Narayanan, 2023). In how many fields did they find leakage, and which types were most common?
2. Benchmark contamination: LLMs may have seen test questions during pretraining. Read about how papers detect it (for example, the GPT-4 technical report's contamination analysis). Why is it so hard to rule out?
3. For a very imbalanced problem, compare ROC curves with precision–recall curves. Read *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets* (Saito & Rehmsmeier, 2015).

## Your turn

Open the **Exercises** tab. Implement everything with NumPy (no sklearn.metrics or sklearn.model_selection); the tests compare your results with scikit-learn's.
