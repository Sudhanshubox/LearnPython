# m92 · Capstone: your own research project

**This is the finale of the whole course.** You'll choose, plan, run and publish an original project or a novel extension of a paper, using everything from Phase 1's variables to Phase 9's statistics. It's the project you'll show in interviews and link from your CV, and it's your first piece of real research.

There are no hidden tests for the project itself: it's yours. The exercises in this module help you **plan** it well (a validated proposal, a schedule, a budget) and **assess** it honestly at the end (a rubric).

---

## 1. Choosing a project

A good capstone is:

- **Interesting to you.** You'll spend 6–16 weeks on it.
- **Specific:** a question with a measurable answer, not "build an AI".
- **Feasible** on your hardware and budget. Laptop-scale research is real research: the m75 arithmetic paper used small models.
- **Connected** to prior work you can cite and compare against.

Four types, in rough order of difficulty:

| Type | What | Example |
|---|---|---|
| **Reproduction** | Reproduce a paper's main result at small scale, report what held and what didn't | Reproduce *Grokking* (Power et al., 2022) on modular arithmetic |
| **Extension** | Take a known result and test it somewhere new | Does the reversed-digit trick (m75) help multiplication? |
| **Original** | Ask a new question | How does BPE vocabulary size (m66) affect a small GPT's (m70) loss on code vs English? |
| **Application** | Build and *evaluate* a real system | A RAG assistant for your college's syllabus or a local-language corpus, with a proper eval set (m80–m85) |

More ideas, building on this course:

- **Phase 6:** Does `zero_init_residual` (m65) change how deep a plain network can be trained? Measure the gradient norms (m63) across depths.
- **Phase 7:** Scaling laws (m72) for a character-level vs a BPE-level GPT on the same text. Which is more compute-efficient?
- **Phase 7:** LoRA rank vs forgetting (m73): plot new-domain gain against old-domain loss for r = 1 to 64.
- **Phase 7:** Can DPO (m74) teach a tiny GPT to prefer shorter answers without hurting accuracy?
- **Phase 8:** Does contextual chunking (m80) improve retrieval for an Indian-language corpus as much as for English?
- **Phase 8:** How well does an LLM judge (m82) agree with human grades on programming explanations? Measure Cohen's kappa.
- **Phase 9:** How many seeds do published comparisons in your favourite subfield need (m87's power analysis applied to reported results, m86)?

## 2. Write the proposal (`proposal.toml`)

Fill in the proposal before writing code. The test `test_my_proposal` passes only when it's complete:

- a title, a one-sentence summary (at most 40 words) and the project type;
- a **research question**, a **hypothesis** with a number, and why it matters;
- the **approach**, at least one **baseline**, and planned **ablations** (m87);
- **metrics**, **datasets**, a pre-registered **success criterion**, and **at least 3 seeds**;
- a **compute budget** (GPU hours, API dollars);
- at least **4 milestones** with concrete deliverables, within 16 weeks;
- at least **2 risks**, each with a mitigation.

Then ask the mentor: "Review my capstone proposal as a research supervisor would." Revise it at least once.

## 3. Run it like a researcher

- **Week 1:** set up a repo (m32), data pipeline and tests (m31). Track every run (m88) from day one.
- **Pilot early:** a tiny version of the main experiment in the first two weeks, to estimate variance and catch problems (m87's power analysis).
- **Baselines before your method.** Get the simple thing working and measured first.
- **Keep a research log** (m88's habit): dated notes on what you tried, what happened, what's next.
- **Expect surprises.** When results contradict the hypothesis, investigate: either something is wrong (m63's debugging) or you've found something real. Both are progress. Report negative results honestly.
- **Stop and write** when the milestones say so, not when everything is perfect.

## 4. Deliverables

1. A **public GitHub repository**: clean code, a README with how to reproduce every number, pinned requirements, tests for the core logic.
2. A **report** (4–8 pages) or a **technical blog post** structured as in m90: abstract with numbers, method, results with mean ± std over seeds and the right statistical tests (m89), ablations, limitations.
3. **Figures** that pass m90's checks.
4. Optional: a short talk or video, a contribution back to a library you used (m91), or a submission to a workshop or student research track.

## 5. Assess yourself

At the end, rate your project honestly from 0 to 4 on each criterion (the exercises compute a score out of 100), and ask the mentor to rate it too:

| Criterion | Weight | 4 means |
|---|---|---|
| Question | 15 | Specific, well motivated, connected to prior work |
| Baselines | 15 | Strong, fairly tuned, clearly described |
| Rigor | 20 | Pre-registered, enough seeds, correct tests, ablations |
| Results | 20 | Clear answer to the question, honest about uncertainty |
| Writing | 15 | A reader could summarise your contribution in a sentence |
| Code | 15 | Someone else could reproduce your numbers from the repo |

---

## Problem-solving habit #92: finish things

Researchers are judged by what they finish and share, not by what they start. Scope down until it's doable, set deadlines, and publish the honest version. A small, complete, well-measured project beats a big, impressive, unfinished one, every time.

## Where to go next

You've built a path from `print("hello")` to training transformers, building LLM systems and running research. From here:

- **Keep a weekly rhythm:** a paper a week (m86), problems for sharpness (Phase 2), one project at a time.
- **Join communities:** open-source projects (m91), paper reading groups, Kaggle competitions, local AI meetups, research labs at your university.
- **Share your work:** blog posts, GitHub, talks. Opportunities find people whose work is visible.
- **Go deeper in one direction** (interpretability, efficient training, retrieval, evaluation, agents, safety) and keep the mentor as your study partner.

## Your turn

Open `proposal.toml`, fill it in, and make the tests pass. Then build it.
