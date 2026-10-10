# m88 · Tracking experiments

**By the end you can:** record every experiment so it can be found, compared and reproduced months later (configuration, code version, environment, metrics over time, artifacts and outcome) by building a small experiment tracker yourself, and you'll know how the same ideas look in Weights & Biases, MLflow and TensorBoard.

**Why it matters for AI:** research produces hundreds of runs. Without tracking, you can't answer "which settings gave that 97% result last month?", "was that before or after I fixed the data bug?", or "can you reproduce Figure 3?". Experiment tracking is the lab notebook of machine learning, and reviewers, collaborators and your future self will all depend on it.

---

## 1. What to record for every run

| What | Why | Example |
|---|---|---|
| **Config** | What you ran | learning rate, model size, dataset, seed |
| **Code version** | Which code produced it | git commit hash, and whether there were uncommitted changes ("dirty") |
| **Environment** | Library versions change results | Python version, platform, package versions |
| **Metrics over time** | Learning curves, not just the final number | `{"step": 300, "name": "val_loss", "value": 1.92}` |
| **Artifacts** | Outputs to inspect later | model checkpoints, predictions, plots, sample generations |
| **Status** | Did it finish? | running / finished / failed |

The **dirty** flag matters: if you ran with uncommitted changes, the commit hash alone doesn't identify the code. Commit before important runs.

## 2. The design

Your tracker stores each run as a folder of plain files:

```
runs/
  digits/                       <- a project
    run-0001/
      config.json
      metadata.json             <- status, times, python, platform, git
      metrics.jsonl             <- one JSON object per line, appended as training runs
      summary.json              <- the last value of each metric
      artifacts/predictions.csv
```

Design choices worth noticing:

- **Append-only JSON lines** for metrics: cheap to write during training, and a crash loses at most the last line.
- **Plain files**: readable with `cat`, greppable, diffable, easy to sync, no server needed.
- A **context manager** (m29) wraps each run: `with Run(...) as run:` marks it *finished* on success and *failed* if an exception escapes, so crashed runs never masquerade as completed ones.
- Never overwrite a run: a new name or an error.

## 3. Comparing runs

Once runs are recorded, research questions become queries: load every run in a project, filter out failed ones, find the best by a metric, show a table of configs and results, and **diff two configs** to see exactly what changed between a good run and a bad one.

## 4. The real tools

Your tracker has the same shape as the professional ones:

```python
import wandb                                      # Weights & Biases (hosted)
run = wandb.init(project="digits", config={"lr": 3e-3, "seed": 0})
for step in range(steps):
    wandb.log({"train_loss": loss, "val_acc": acc}, step=step)
run.finish()
```

- **Weights & Biases:** hosted dashboards, sweeps (hyperparameter search), artifact versioning, reports. Free for individuals and academics; widely used in research.
- **MLflow:** open source, self-hostable, with a model registry for deployment.
- **TensorBoard:** simple local learning-curve plots, built into PyTorch via `torch.utils.tensorboard`.

Use one of them for your capstone. The concepts transfer directly.

## 5. Reproducibility checklist

For a result you'll report: fixed seeds (m59), the config and commit recorded, `pip freeze > requirements-lock.txt` saved as an artifact, data version recorded (a hash of the dataset file), and the exact command. The test: could a classmate rerun it from your records alone?

---

## Problem-solving habit #88: write it down while it happens

Memory is unreliable and confident. Record decisions and results *as they happen*: a tracker for runs, plus a short dated research log ("Tried warmup 500 → loss spike gone; next: check if it's the LR or the warmup"). Ten minutes of logging a day saves days of confusion later, and the log becomes the first draft of your paper's method section.

## Common mistakes

- Overwriting results files with each new run.
- Tracking only the final metric, not the learning curve.
- Runs from uncommitted code with no record of the changes.
- Crashed runs left looking like finished ones.
- Tracking in a notebook's memory, which disappears when the kernel restarts.

## Go deeper (optional)

1. Make a free W&B account and log your m70 GPT training (loss, learning rate, gradient norm). Compare two learning rates in the dashboard.
2. Read *Hidden Technical Debt in Machine Learning Systems* (Sculley et al., 2015), the section on configuration debt. How does tracking help?
3. Add `pip freeze` output and a SHA-256 of the dataset to your tracker's metadata, then try to reproduce one of your runs exactly a week later.

## Your turn

Open the **Exercises** tab. The tests create runs in a temporary folder (and a tiny git repository to test the code-version recording).
