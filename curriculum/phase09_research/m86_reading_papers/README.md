# m86 · How to read a research paper

**By the end you can:** read a machine learning paper efficiently with the three-pass method, take structured notes that separate claims from evidence, keep a reference library with proper citations, decide what to read next, and sanity-check a paper's numbers instead of taking them on trust.

**Why it matters for AI:** AI moves through papers. New architectures, training tricks and evaluation methods appear on arXiv years before they reach textbooks or courses (including this one). Researchers read several papers a week; strong engineers read the important ones. The skill isn't reading every word: it's extracting what matters quickly and judging how much to believe it.

---

## 1. The three-pass method

From S. Keshav's classic two-page note *How to Read a Paper* (2007), adapted for ML:

**Pass 1: the bird's-eye view (5–10 minutes).** Read the title, abstract and introduction; the section headings; the figures and tables (with captions); the conclusion. Glance at the references you recognize. Afterwards you should be able to answer the **five Cs**: *Category* (new method? analysis? benchmark?), *Context* (which earlier work does it build on?), *Correctness* (do the assumptions look reasonable?), *Contributions* (what's actually new?), *Clarity*. Most papers stop here: you now know whether to go on.

**Pass 2: the content (about an hour).** Read the whole paper, but skip the proofs and fine details. Study every figure: what are the axes, are there error bars, which baselines are compared, is the comparison fair? Note the claims and *the evidence for each*. Mark terms and references you don't know.

**Pass 3: deep understanding (several hours).** Re-derive the key equations, re-implement the core idea (as you did in Phases 6–7), and think about what the authors didn't test. A good pass 3 lets you reconstruct the paper from memory and point out its weaknesses. Do this for the handful of papers central to your work.

## 2. Notes: claims and evidence

Write a note for every paper you read past pass 1. The most important habit: **pair every claim with its evidence**.

| Claim | Evidence |
|---|---|
| "Reversing the output digits makes addition far easier to learn" | Figure 2: exact-match accuracy vs training examples, plain vs reversed |
| "Our method is state of the art" | Table 3, one seed, no error bars ← weak |

Claims without evidence (or with weak evidence: a single seed, a cherry-picked example, an unfair baseline) are where papers overreach. Also note **limitations** (stated and unstated) and your own **questions**: they're future project ideas.

## 3. A reference library

Keep notes in one place (plain Markdown files in a Git repo work well; Zotero is a good reference manager). Record the arXiv ID (`2307.03381`, plus a version like `v2` when it matters, since arXiv papers get revised) and generate **BibTeX** for citing:

```bibtex
@article{vaswani2017attention,
  title = {{Attention Is All You Need}},
  author = {Ashish Vaswani and Noam Shazeer and ...},
  year = {2017},
  eprint = {1706.03762},
  archivePrefix = {arXiv}
}
```

The key `lastname + year + first meaningful title word` is a common convention.

## 4. What to read

You can't read everything. Prioritize by **relevance** to your current question first, then by influence (citations, adjusted for age: a 2-year-old paper with 500 citations is more notable than a 10-year-old one with 1,000). Sources: survey papers (best starting point in a new area), the reference lists of papers you liked, conference best-paper awards, and following a few researchers whose work you trust. Read classics too: many "new" ideas are old ones rediscovered.

## 5. Read critically

Some checks take minutes and catch real problems:

- **Recompute the numbers.** Does the "Average" column match the average of the other columns? Do percentages add up? Errors in tables are surprisingly common.
- **Improvement vs noise.** If the gain over the baseline is smaller than the variation between random seeds (m59, m75), the paper hasn't shown an improvement. Two standard deviations is a reasonable minimum bar for a quick check (m89 goes further).
- **Baselines.** Were baselines tuned as carefully as the new method? Are they recent?
- **Ablations.** Does each component of the method actually contribute (m87)?
- **Reproducibility.** Is code released? Are hyperparameters and compute reported?

None of this is cynicism: most papers are honest and useful. But "this paper shows X" should mean *you checked*.

---

## Problem-solving habit #86: explain it to learn it

After reading a paper, explain its core idea in three sentences to someone else (the mentor works well: "here's my summary of the ResNet paper; what did I get wrong?"). If you can't, you haven't understood it yet. This is the Feynman technique, and it works for code, maths and papers alike.

## Common mistakes

- Reading linearly from the first word to the last, then remembering nothing.
- Skipping the figures (they usually *are* the results).
- Taking the abstract's claims at face value.
- Not writing notes, then re-reading the same paper six months later.
- Only reading the newest papers and missing the foundational ones.

## Go deeper (optional)

1. Read Keshav's *How to Read a Paper* (2 pages). Then do all three passes on a paper from this course's "Go deeper" sections (for example the ResNet or LoRA paper), and write a note with the exercises' `PaperNote`.
2. Read *Troubling Trends in Machine Learning Scholarship* (Lipton & Steinhardt, 2018). Which of its four trends can you spot in a recent paper?
3. Start a reading log: one note per paper, one paper per week. After ten weeks, read your log back. What patterns do you see?

## Your turn

Open the **Exercises** tab: tools for your own paper notes and a few critical-reading checks.
