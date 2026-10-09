# m48 · Visualization

**By the end you can:** build clear charts with matplotlib's object-oriented API (figures and axes), choose the right chart for a question, and make the plots every ML practitioner relies on: loss curves, distributions, class scatter plots, confusion-matrix heatmaps and image grids.

**Why it matters for AI:** you can't debug what you can't see. A loss curve shows overfitting, divergence or a learning rate that's too high long before a final metric does. Histograms reveal skewed features and broken preprocessing. A grid of misclassified images explains a model's failures better than any number. And clear figures are how research results get communicated.

---

## 1. Figures and axes

matplotlib has two interfaces. Use the **object-oriented** one: create a **Figure** (the whole canvas) and one or more **Axes** (individual plots), then call methods on the axes.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs, train_loss, label="train")
ax.plot(epochs, val_loss, label="validation")
ax.set_title("Loss")
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy")
ax.legend()
fig.savefig("loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)        # free memory when making many figures
```

Functions that **return the figure** (instead of calling `plt.show()`) are easy to test, reuse, save and put in reports. In a Jupyter notebook, a returned figure is displayed automatically.

## 2. Choosing a chart

| Question | Chart | Method |
|---|---|---|
| How does something change over time or steps? | line | `ax.plot` |
| How are values distributed? | histogram | `ax.hist` |
| How do two variables relate? groups in 2-D? | scatter | `ax.scatter` |
| Compare amounts across categories | bar (horizontal for long labels) | `ax.bar`, `ax.barh` |
| A matrix of values (confusion matrix, attention) | heatmap | `ax.imshow` + `fig.colorbar` |
| Pictures | image grid | `ax.imshow`, one axes per image |

## 3. Several plots in one figure

```python
fig, axes = plt.subplots(2, 3, figsize=(9, 6))     # axes is a 2-D array of Axes
for ax, img in zip(axes.flat, images):
    ax.imshow(img, cmap="gray")
    ax.axis("off")
fig.tight_layout()
```

## 4. Plots every ML practitioner needs

- **Loss curves** (train vs validation, often with a log y-axis): validation loss going up while training loss goes down means overfitting. Mark the best epoch.
- **Histograms** of features, predictions and errors, with the mean or threshold marked by a vertical line (`ax.axvline`).
- **Scatter plots** of 2-D projections (PCA from m39) colored by class: do the classes separate?
- **Confusion-matrix heatmaps** with the counts written in each cell (`ax.text`).
- **Image grids** of examples, especially the misclassified ones.

## 5. Making figures honest and readable

- Always label axes, with units. Give the figure a title that states the takeaway when possible.
- Start bar charts at zero; a truncated axis exaggerates differences.
- Use colorblind-friendly palettes (matplotlib's default "tab10" is decent; avoid red vs green only).
- Use a log scale when values span orders of magnitude (losses, learning rates), and say so on the axis.
- Show uncertainty (error bars or shaded bands from m43's confidence intervals) when comparing results.

## 6. Higher-level libraries

**seaborn** builds statistical plots from DataFrames in one line (`sns.histplot`, `sns.heatmap`, `sns.pairplot`); **plotly** makes interactive charts. They're all built on the same ideas. Learn matplotlib first so you can fix any detail.

---

## Problem-solving habit #44: plot before you trust a number

Before reporting an average, look at the distribution. Before trusting a final accuracy, look at the loss curves. A single number hides outliers, bimodal distributions and training instabilities that a picture shows instantly. (Look up "Anscombe's quartet": four datasets with identical statistics and completely different plots.)

## Go deeper (optional, research-level)

1. Recreate Anscombe's quartet with `fig, axes = plt.subplots(2, 2)` and verify that the means, variances and correlations match.
2. Read the classic essay *Ten Simple Rules for Better Figures* (Rougier, Droettboom & Bourne, 2014). Which rules do your exercise figures follow?
3. Plot the loss curves of your m41 optimizers (GD, momentum, Adam) on the same axes with a log scale. What do the shapes tell you?

## Your turn

Open the **Exercises** tab. Every function returns a matplotlib `Figure`; the tests inspect its axes, lines, labels and patches, so follow the specifications exactly. Try displaying your figures in a notebook (or saving them) to see them.
