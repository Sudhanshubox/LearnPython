# m55 · Project: customer churn prediction, end to end

**Phase 5 finale.** A subscription company is losing customers. You've been given a raw export of 3,000 customer records (`churn_raw.csv`) and asked: *which customers are about to leave, and what should we do about it?*

This is a complete, realistic ML project: messy data, a hidden leakage trap, a pipeline, honest evaluation, a decision threshold based on business costs, and a written report. It uses everything from m46–m54.

---

## The brief

- **Data:** one row per customer. `Churned` is "Yes" if they cancelled.
- **Goal:** predict churn for current customers so the retention team can contact the highest-risk ones.
- **Costs:** a missed churner (false negative) loses about ₹500 of future revenue; contacting a customer who wasn't going to leave (false positive) costs about ₹50 for a retention offer.
- **Deliverables:** a clean dataset, a leak-free model with honest test metrics, a cost-based decision threshold, a top-risk customer list, and a short report.

## Steps

1. **Explore** (in a notebook or with the mentor): `info()`, `describe()`, `value_counts()` for every column. Write down every data problem you find. There are several.
2. **Clean** (`load_and_clean`): normalize column names, fix inconsistent text, parse money and "14 months", parse dates, convert the target to 0/1, and remove duplicate rows.
3. **Hunt for leakage** (`leaky_columns`): one column would never be available at prediction time. Look at how each column relates to `churned`, especially its *missing values*. What happens to your model's AUC if you keep it?
4. **Model** (`feature_columns`, `build_pipeline`, `train_and_evaluate`): a scikit-learn pipeline with imputation, scaling and one-hot encoding, trained on a stratified 80/20 split (`random_state=0`). Start with logistic regression; try gradient boosting (m54) as a comparison.
5. **Choose a threshold** (`best_cost_threshold`): 0.5 isn't sacred. Pick the threshold that minimizes the total cost from the brief.
6. **Act** (`top_risk_customers`): rank customers by predicted churn probability.
7. **Report** (`write_report`): a Markdown report with what you found and what you recommend.

## What good looks like

A leak-free model on this data reaches a test ROC AUC of roughly **0.85**. If you see **0.95 or more, something is leaking**, and the tests will tell you so. The cost-based threshold should cut the cost dramatically compared with 0.5, because missed churners are 10× more expensive than false alarms.

## Report outline (`write_report`)

Your report must contain these sections (the tests check the headings), each with your own short, specific text:

```markdown
# Churn model report
## Data
## Leakage
## Results
## Recommendations
```

Explain in plain language: what you cleaned, what leaked and why, the honest metrics compared with the baseline, the chosen threshold and its cost, and 2–3 concrete actions for the business (which features drive churn? which customers should be contacted first?). Ask the mentor to review it like a manager would.

## Stretch goals (optional)

- Compare logistic regression, a random forest and gradient boosting with 5-fold cross-validation and confidence intervals (m43, m49).
- Plot the ROC curve, the cost as a function of the threshold, and churn rate by contract type (m48).
- Interpret the model: logistic regression coefficients, or permutation importance for the boosted model. Which features matter most, and do they make business sense?
- Calibration: when the model says 30%, do about 30% of those customers churn? Look up `sklearn.calibration.calibration_curve`.

## Go deeper (optional, research-level)

1. Read *Machine Learning: The High-Interest Credit Card of Technical Debt* (Sculley et al., 2014). Which of its risks apply to deploying this churn model?
2. Predicting churn isn't the same as preventing it: the customers most likely to leave may not be the ones a call can save. Read about **uplift modelling**. How would you design an experiment to measure whether calling customers actually reduces churn?
