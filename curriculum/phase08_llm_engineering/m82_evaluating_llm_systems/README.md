# m82 · Evaluating LLM systems

**By the end you can:** build an evaluation for an LLM application, score outputs with reference-based metrics (exact match, token F1) and with an LLM judge (with a rubric and robust parsing), compare two systems pairwise while cancelling the judge's position bias, put confidence intervals on every number with the bootstrap, test whether a change really helped with a paired test, and check that your judge agrees with humans.

**Why it matters for AI:** LLM outputs are open-ended, so "does it work?" has no unit-test answer. Teams that ship good LLM products all have one thing in common: an evaluation set they run on every change. It turns prompt tweaking from guesswork (m77) into engineering, and it's the core skill behind model research too.

---

## 1. Build the eval set first

An evaluation is a set of **inputs** with what a good output looks like (a reference answer, or a rubric), plus a **metric**. Good eval sets:

- come from **real usage** (logs, user questions), not only from questions you made up,
- cover the important categories *and* the hard cases (multi-step questions, "I don't know" cases, adversarial inputs),
- are large enough to measure the differences you care about (section 4), and
- are **held out**: if you tune prompts on them, keep a separate test portion you only look at occasionally (m49's lesson, again).

## 2. Reference-based metrics

When there's a short correct answer, compare text:

- **Normalize** first (the SQuAD convention): lowercase, remove punctuation, remove the articles a/an/the, collapse whitespace. "The Eiffel Tower." and "eiffel tower" should match.
- **Exact match:** 1 if the normalized strings are equal.
- **Token F1:** precision and recall over the normalized words (counting repeats), combined into F1 (m49). It gives partial credit for nearly-right answers.

These are cheap, deterministic and good for extraction and short answers. They're useless for explanations, where a perfect answer can share few words with the reference.

## 3. LLM as a judge

For open-ended outputs, ask a strong model to grade, with:

- the question, a **reference answer**, and the answer to grade,
- an explicit **rubric** ("5: correct and complete; 3: correct but missing an important caveat; 1: wrong"),
- **reasoning before the score**, and a strictly parseable format (`<score>4</score>`). A reply without a valid score is recorded as `None` (a judge failure to look at), never silently as 0 or 5.

Judges have known biases, and you should measure them rather than assume them away:

- **Position bias:** in pairwise comparisons, judges tend to prefer whichever answer they see first. Fix: judge each pair **twice, swapping the order**, and only count a win if both orderings agree; otherwise it's a tie.
- **Verbosity and self-preference bias:** judges may favour longer answers or their own style. Rubrics that reward correctness and concision help.
- **Agreement with humans:** label a sample by hand (pass/fail) and compute **Cohen's kappa** between judge and human: agreement corrected for chance (κ = (p_observed − p_chance) / (1 − p_chance)). Above about 0.6 is substantial; below 0.4, fix the judge prompt before trusting it.

## 4. Uncertainty: never report a bare number

With 50 test questions, an accuracy of 80% could easily be 70% or 90% on a different 50. The **bootstrap** (m43) estimates this: resample the per-example scores with replacement thousands of times, compute the mean each time, and take the 2.5th and 97.5th percentiles as a 95% interval.

To compare two systems on the **same** questions, use a **paired** test: bootstrap the per-question *differences* (B − A). If the interval excludes 0, the improvement is unlikely to be noise. Paired comparisons are much more sensitive than comparing two separate intervals, because per-question difficulty cancels out.

## 5. The eval harness

```python
rows, summary = run_eval(system_fn, dataset, {"em": exact_match, "f1": token_f1})
```

Keep the per-example rows, not just the summary: reading the failures is where improvements come from. Log the model, the prompt version (m77) and the date with every run.

---

## Problem-solving habit #82: is the difference real?

Before believing any comparison (two prompts, two models, two retrievers) ask: how big is the difference relative to the noise? Compute a confidence interval, use paired comparisons, and be suspicious of improvements that only appear on the examples you tuned on. This one habit prevents most false conclusions in ML engineering and research.

## Common mistakes

- Evaluating on the examples you used to write the prompt.
- Reporting "accuracy went from 82% to 85%" on 40 examples as an improvement.
- Trusting an LLM judge without ever checking it against human labels.
- Treating a judge parsing failure as a score.
- Pairwise judging in one order only.

## Go deeper (optional)

1. Read *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (Zheng et al., 2023). What biases did they find, and how large were they?
2. Read Anthropic's guide on building evals ("Define success criteria and build evaluations"). Write success criteria for the m80 RAG assistant: what exactly should be measured?
3. Read *Adding Error Bars to Evals* (Miller, Anthropic, 2024). Why do clustered questions (several questions about the same document) need clustered standard errors?

## Your turn

Open the **Exercises** tab. The fake judges in the tests include one with a strong position bias: your pairwise comparison must not be fooled by it.
