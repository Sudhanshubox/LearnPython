# m72 · Scaling laws

**By the end you can:** fit power laws in log-log space (with and without an irreducible-loss term), use the Chinchilla scaling law to find the compute-optimal model size and dataset size for a FLOP budget, run your own small scaling experiment, and critically evaluate a scaling-law result, including one that turned out to be wrong.

**Why it matters for AI:** scaling laws are why the AI industry looks the way it does. They let labs predict the loss of a $100M training run from a few cheap small runs, decide how big a model to train and on how much data, and they're the quantitative backbone of every "bigger is better" argument and of the debates about where it stops. Reading and doing scaling studies is a core research skill.

---

## 1. Power laws

Kaplan et al. (2020, *Scaling Laws for Neural Language Models*) found that test loss falls as a **power law** in model size N (non-embedding parameters), data D (tokens) and compute C, over many orders of magnitude:

L(N) ≈ (N_c / N)^α_N

A power law is a straight line on a **log-log** plot: log L = −α log N + const. So fit it with linear regression on (log N, log L) (m50); the slope is −α. The exponents are small (α ≈ 0.07 for N in Kaplan's fits), which is why each 10× in size buys a modest, but predictable, improvement.

Real curves flatten out at an **irreducible loss** E (the entropy of the text itself: no model can predict a coin flip). Then L = E + A·N^(−α). To fit the offset, try a grid of E values below the smallest observed loss, fit the power law to L − E for each, and keep the E with the smallest error in log space.

## 2. Compute: C ≈ 6ND

Training costs about 6 FLOPs per parameter per token (m64). With a fixed compute budget C, you choose how to split it: a bigger model on fewer tokens, or a smaller model on more tokens.

## 3. Chinchilla: the compute-optimal trade-off

Hoffmann et al. (2022, *Training Compute-Optimal Large Language Models*) fitted a parametric law in both N and D:

L(N, D) = E + A / N^α + B / D^β

For a budget C, the optimal choice minimizes L subject to D = C / (6N). Search over a log-spaced grid of N. Their headline result: parameters and tokens should grow **equally** with compute, about **20 tokens per parameter**. GPT-3 (175B parameters, 300B tokens, under 2 tokens per parameter) was badly undertrained; Chinchilla (70B, 1.4T tokens) beat the 280B-parameter Gopher with the same compute.

## 4. A replication story

Here's a twist worth learning from. In 2024, Besiroglu et al. (*Chinchilla Scaling: A replication attempt*) reconstructed the data from the paper's figures and refitted the parametric law. With the **published** constants (E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28), the compute-optimal ratio at Chinchilla's own budget is about **90 tokens per parameter**, inconsistent with the paper's own 20 and with how Chinchilla itself was trained. Their **corrected** fit (E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658) gives about 18–20, matching the other two methods in the paper. The published fit had a problem in the optimization (the loss function and optimizer settings), and the confidence intervals reported were implausibly narrow.

You'll compute both in the exercises. The lesson: even landmark papers contain errors; re-derive the numbers you rely on, and check that a paper's results agree with each other.

## 5. Your own scaling experiment

The exercises train tiny one-layer GPTs of increasing width (8 → 64) on this course's lessons for the same number of steps, and fit a power law to their validation losses. You'll see the loss fall smoothly with parameters, even at this microscopic scale. Two cautions apply to any small-scale scaling study:

- The **learning rate** and number of steps must be reasonable for every size, or the bigger models look worse (undertrained), and you'll "discover" a fake plateau. (Try a learning rate that's too high for the biggest model and watch the curve bend.)
- Count **non-embedding** parameters (Kaplan's convention): embedding tables are large but cheap.

Then use the fit to **extrapolate**: predict the loss of a model 10× bigger, train it, and see how close you were. That's exactly how labs plan big runs.

## 6. Beyond loss

Scaling laws predict **loss** well. Specific abilities (arithmetic, following instructions) can appear to emerge suddenly, though Schaeffer et al. (2023, *Are Emergent Abilities of Large Language Models a Mirage?*) argue much of the suddenness comes from discontinuous metrics. Scaling laws also exist for data repetition, inference compute, and fine-tuning.

---

## Problem-solving habit #72: think in orders of magnitude

When a quantity spans many orders of magnitude (parameters, compute, dataset sizes, latencies), plot it on log axes and reason about ratios, not differences. "10× more compute for 0.1 nats" is a meaningful statement; "1e21 more FLOPs" is not. Log-log plots turn power laws into straight lines and make extrapolation visible, including when it's unjustified.

## Common mistakes

- Fitting a power law with linear regression on the raw (not logged) values.
- Extrapolating far beyond the fitted range without error bars.
- Comparing model sizes trained with different numbers of tokens and attributing the difference to size.
- Including embedding parameters for tiny models (they dominate the count).

## Go deeper (optional, research-level)

1. Read Kaplan et al. (2020), sections 1–3, then Hoffmann et al. (2022), section 3. Why did they reach different conclusions about the optimal ratio? (Hint: learning-rate schedules and what was held fixed.)
2. Read Besiroglu et al. (2024). Reproduce their Figure 1 with your `compute_optimal` function for both sets of constants.
3. Modern models such as Llama 3 (8B parameters, 15T tokens, ~1,900 tokens per parameter) are trained far past "Chinchilla-optimal". Why does that make sense when you also account for **inference** cost? Read *Beyond Chinchilla-Optimal* (Sardana et al., 2023).

## Your turn

Open the **Exercises** tab. The scaling experiment trains four tiny models (about 30 seconds on a CPU).
