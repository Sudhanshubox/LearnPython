# m90 · Writing about your work

**By the end you can:** structure a technical blog post or a short paper, write an abstract that people actually read, present results honestly (mean ± standard deviation over seeds, the best result marked but not oversold), make figures that stand on their own, and use small tools to catch unsupported claims, missing sections, unreadable prose and unlabelled plots before a reader does.

**Why it matters for AI:** work that isn't communicated might as well not exist. A clear blog post about a project can do more for an AI career than the project itself: it's how hiring managers, collaborators and researchers find you. And in research, the paper *is* the product. Clear writing is also clear thinking: if you can't explain why your result holds, you might not know.

---

## 1. Structure

**A paper:** Abstract · Introduction (problem, why it matters, what you did, contributions) · Related work · Method · Experiments (setup, results, ablations) · Discussion and limitations · Conclusion.

**A technical blog post:** the same logic, lighter: a hook (the question or the surprising result), background, what you did, what you found (figures!), what didn't work, what's next, and a link to the code. Write for a smart reader from a neighbouring field.

Either way, decide your **one main message** before writing. Every section should serve it.

## 2. The abstract

Most readers read only the title and abstract (m86's first pass), so this is where your effort pays off most. A reliable five-sentence formula:

1. **Context:** the area and why it matters.
2. **Gap:** what's missing or broken.
3. **Approach:** what you did.
4. **Result:** the key finding, *with a number*.
5. **Implication:** why it matters, what it enables.

## 3. Claims need evidence

Every claim of improvement ("outperforms", "faster", "state of the art", "significantly better") should come with its evidence in the same sentence: a number, a confidence interval, or a pointer to a table or figure. "Our method outperforms the baseline" is weak; "Our method improves accuracy from 91.2 ± 0.4 to 93.0 ± 0.3 over 8 seeds (Table 2)" is strong. Reserve "significant" for statistical significance, with the test named (m89).

The exercises include a checker that flags sentences making comparative claims with no number, citation or table/figure reference. It's crude (it can't judge whether the evidence is *good*), but it catches the most common overreach.

## 4. Tables

- Report **mean ± standard deviation over seeds**, and say how many seeds.
- **Bold the best** value in each column, and say whether higher or lower is better.
- Use sensible precision: "96.43 ± 0.71" claims more precision than seeds support; "96.4 ± 0.7" is honest.
- Keep the same baselines and the same budget for every row (m87).

## 5. Figures

A good figure can be understood from the figure and its caption alone:

- **Label both axes**, with units ("Validation loss (nats/token)", "Training step").
- A **legend** whenever there's more than one line.
- Show **uncertainty**: shade mean ± std across seeds rather than plotting one lucky run.
- One message per figure, stated in the caption's first sentence ("Reversed digits are learned 3× faster.").
- Readable when printed in black and white, and by colour-blind readers (m48).

## 6. Prose

Short sentences, active voice ("we trained", not "the model was trained by us"), concrete words, defined terms, no unexplained acronyms. Readability formulas like **Flesch reading ease** (206.835 − 1.015 × words per sentence − 84.6 × syllables per word; higher is easier, 60–70 is plain English, below 30 is very hard) are rough, but a very low score is a reliable sign of sentences that need splitting.

And be honest about **limitations**: what you didn't test, where the method fails, how far the results might generalize. Reviewers trust papers that state their limits.

## 7. The process

Outline first; write the figures and tables before the prose; then write fast and edit slowly; read it aloud; ask someone (or the mentor) to summarise it back to you: if their summary isn't your main message, rewrite.

---

## Problem-solving habit #90: write to think

Start writing *during* a project, not after. A one-page draft ("we expect X because Y; we'll test it with Z") exposes gaps in your reasoning before you spend weeks on experiments. The best researchers often write the paper's outline before running the main experiment.

## Common mistakes

- An abstract without a single number.
- Bolding every result where your method is best and hiding the rest.
- Plots with no axis labels, or with default "line 1" legends.
- "Significantly better" with no statistical test.
- A limitations section that only lists things you plan to do next.

## Go deeper (optional)

1. Read *How to Write a Great Research Paper* (Simon Peyton Jones, talk and slides). His "write the paper first" advice applies to blog posts too.
2. Read two well-known technical blog posts, such as Karpathy's *The Unreasonable Effectiveness of Recurrent Neural Networks* and Lilian Weng's posts. What do their structure and figures have in common?
3. Write a 1,000-word blog post about one project from this course (the m75 arithmetic project is a great candidate), run your checkers on it, and publish it.

## Your turn

Open the **Exercises** tab: helper tools for your own writing. Run them on your capstone write-up (m92).
