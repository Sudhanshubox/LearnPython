# m75 · Project: teaching arithmetic to a small transformer

**Phase 7 finale: a research reproduction.** Why are LLMs famously bad at arithmetic, and can data *format* fix it? Lee et al. (2023, *Teaching Arithmetic to Small Transformers*) trained small GPTs from scratch on addition and found something striking: simply writing the answer **backwards** (least significant digit first) makes addition dramatically easier to learn.

You'll reproduce that result with your own minigpt, analyse *why* it happens by grouping errors by the number of carries, deal honestly with seed-to-seed variance, and write it up.

---

## Background: read the paper

Read sections 1–4 of *Teaching Arithmetic to Small Transformers* (Lee, Sreenivasan, Lee, Lee & Papailiopoulos, 2023, arXiv 2307.03381). Use the three-pass method and ask the mentor to quiz you. Key points:

- Plain format: `128+367=0495`. To write the **first** answer digit, the model must already know whether a carry ripples all the way from the last digit: it has to "see" the whole computation at once.
- Reversed format: `128+367=5940`. Now the first digit written is the units digit (8 + 7 = 15 → 5, carry 1), and each following digit depends only on two input digits and a carry the model *has just computed and written down*. The task becomes a simple, local algorithm, the way you add on paper.
- They also found **phase transitions** (accuracy jumping from near 0 to near 100% within a few hundred steps) and that the effect is about **sample efficiency**: how much training it takes.

## Setup

- **Data:** 3-digit addition, operands zero-padded (`007+120=`), answers zero-padded to 4 digits. 2,000 training problems, 300 held-out test problems.
- **Model:** minigpt with 2 layers, 4 heads, d_model = 64, block size 16, vocabulary `0123456789+=\n`.
- **Training:** AdamW at lr 1e-3 (no weight decay), batches of 64 problems sampled with replacement, **loss only on the answer** (the targets for the prompt `aaa+bbb=` are −100, m73).
- **Evaluation:** greedy decoding of the 4 answer digits; **exact-match** accuracy (the whole answer right) and **digit** accuracy.

## Steps (`exercises.py`)

1. `format_example`, `make_pairs`, `encode_batch`: the data pipeline and loss masking.
2. `predict` and `evaluate`: batched greedy generation and the two metrics. Remember to **un-reverse** reversed answers before comparing.
3. `train_adder`: training with a learning curve (evaluate every `eval_every` steps).
4. `curve_area`: the mean exact accuracy over the curve, a simple measure of how *fast* a format learns (less sensitive to exactly when a phase transition happens than the final number).
5. `count_carries` and `accuracy_by_carries`: the analysis. If the theory is right, the plain format's errors should concentrate on problems with **many carries**.
6. `write_report`.

## What you should find

Reversed answers typically reach 95–100% exact accuracy within 1,200 steps; the plain format lags far behind (often 30–85%, varying a lot by seed), and its accuracy drops sharply with the number of carries. Expect **phase transitions** in both: digit accuracy creeps up while exact accuracy stays near zero, then jumps.

**On variance:** with small models and phase transitions, single runs can mislead. The tests run one seed to keep things fast; for your report, run at least 3 seeds per format and report means and spreads (m43, m59). If a seed disagrees with the others, say so. That's honest research.

## Report outline (`write_report`)

```markdown
# Teaching addition to a small transformer
## Question
## Setup
## Results        (a table: format, final exact accuracy, final digit accuracy, curve area; carries analysis)
## Analysis       (why does reversal help? what do the carries show?)
## Limitations
```

## Stretch goals (optional)

- More seeds, and a plot of the learning curves for both formats (m48), with exact and digit accuracy.
- **Length generalization:** train on 3-digit problems and test on 4-digit ones (pad the training data's operands to 4 digits). Does either format generalize? (Usually not, which is itself an important finding; see *Transformers Can Do Arithmetic with the Right Embeddings*, McLeish et al., 2024.)
- **Scratchpads:** add intermediate steps (each digit sum and carry) to the answer, as in the paper's "detailed scratchpad" format. How does sample efficiency change?
- Vary the training set size (500, 1,000, 2,000, 5,000) and plot final accuracy for both formats: the paper's Figure 2 in miniature.

## Go deeper (optional, research-level)

1. Why do LLMs that use BPE tokenizers (m66) struggle with arithmetic even more? Look at how GPT-2's tokenizer splits "123456 + 789". Read about right-to-left digit grouping in GPT-4's tokenizer, and *Tokenization counts* (Singh & Strouse, 2024).
2. Read *Progress measures for grokking via mechanistic interpretability* (Nanda et al., 2023). Could you find the "carry" computation inside your trained model by looking at its attention patterns?
3. Chain-of-thought prompting in large models is the same idea as reversing digits and scratchpads: make each generated token depend on information that's already written down. Read *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (Wei et al., 2022) with that lens.

## Your turn

Open the **Exercises** tab. The full comparison trains two models (about a minute on a laptop CPU). When you've finished, ask the mentor to review your report as a workshop reviewer would.
