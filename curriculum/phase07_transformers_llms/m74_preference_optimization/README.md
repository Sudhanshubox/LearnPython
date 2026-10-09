# m74 · Preference optimization: reward models, RLHF and DPO

**By the end you can:** explain how human preferences are turned into a training signal, train a **reward model** with the Bradley-Terry loss, write down the RLHF objective with its KL penalty, compute sequence log-probabilities with a language model, implement **DPO** (Direct Preference Optimization) from scratch, and show how its β parameter controls how far the model moves from its starting point.

**Why it matters for AI:** SFT (m73) teaches a model to imitate good answers; preference optimization teaches it to prefer **better** answers over worse ones, which is much easier for people to judge than to write. It's a key ingredient in making models helpful and harmless, and the area where alignment research meets everyday engineering.

---

## 1. Preference data

Instead of writing ideal answers, people compare two model answers to the same prompt and pick the better one. A dataset is a list of triples **(prompt, chosen, rejected)**. Comparisons are faster to collect, more consistent, and can capture qualities that are hard to demonstrate ("more honest", "less verbose").

## 2. The Bradley-Terry model and reward models

Assume each response has a hidden score r(x, y), and people prefer y₁ over y₂ with probability

P(y₁ ≻ y₂) = σ(r(x, y₁) − r(x, y₂))

(the **Bradley-Terry** model, 1952, also behind chess Elo ratings). A **reward model** is a network trained to predict r: typically a pretrained transformer with its output layer replaced by a linear head producing one number, read at the **last token** of the sequence. Train it by maximum likelihood on the comparisons:

loss = −mean log σ(r_chosen − r_rejected)

Only *differences* of rewards matter, so the reward's absolute level is arbitrary.

## 3. RLHF: optimize the reward, but don't go far

Classic RLHF (Christiano et al., 2017; Ouyang et al., 2022, *InstructGPT*) then uses reinforcement learning (PPO) to train the policy π (the LLM) to maximize

E[ r(x, y) ] − β · KL( π(·|x) ‖ π_ref(·|x) )

The **KL penalty** keeps the policy close to the reference (the SFT model). Without it, the policy finds outputs that exploit the reward model's mistakes (**reward hacking**: gibberish that happens to score highly). Per sample, the penalized reward is r − β·(log π(y|x) − log π_ref(y|x)).

PPO with four models in memory (policy, reference, reward, value) is complicated and finicky. Which motivates...

## 4. DPO: preference optimization without RL

Rafailov et al. (2023) showed that the optimal policy for the KL-penalized objective satisfies

r(x, y) = β · log( π(y|x) / π_ref(y|x) ) + (a term that depends only on x)

So the policy **is** implicitly a reward model. Substitute this into the Bradley-Terry loss and the prompt-only term cancels:

loss = −mean log σ( β·[log π(y_c|x) − log π_ref(y_c|x)] − β·[log π(y_r|x) − log π_ref(y_r|x)] )

That's a simple classification-style loss on the policy's log-probabilities: no reward model, no sampling, no RL. The quantities β·(log π − log π_ref) are the **implicit rewards**; training accuracy is how often the chosen one is higher.

**β** plays the same role as in RLHF: a small β lets the policy move far from the reference to satisfy the preferences; a large β keeps it close.

## 5. Sequence log-probabilities

Both RLHF and DPO need log π(y|x): the log-probability of a whole response given the prompt. With a causal LM, run it on prompt + response and sum the log-softmax probabilities of the **response** tokens only (the same masking idea as SFT, m73):

```
tokens:   Q : _ a b c \n A : _ y e s ! \n
logits at position t predict token t + 1
sum log p(token) over the response tokens: y, e, s, !, \n
```

Batches mix different lengths, so pad and mask carefully. The reference model is frozen and evaluated without gradients.

## 6. The toy setup in the exercises

The prompts are "Q: abc\nA: " for random three-letter words. An SFT model is trained to answer "yes!\n" or "no.\n" at random (50/50), so it's indifferent. The preference data always prefers "yes!". After DPO, the probability that the model prefers "yes!" goes from about 0.5 to nearly 1, and with a larger β the policy's log-probabilities move less from the reference.

---

## Problem-solving habit #74: know what your metric can be gamed by

Any learned objective (a reward model, an automatic judge, a benchmark) can be over-optimized: the model gets better at the measure, not at the goal (Goodhart's law). Before optimizing hard against a metric, ask how a clever optimizer could satisfy it without doing what you want, and keep an independent check (human review, a held-out evaluation, a KL budget).

## Common mistakes

- Summing log-probabilities over the prompt tokens too.
- Letting gradients flow into the reference model, or updating it.
- Reading the reward model's output at a padding position instead of the last real token.
- Comparing implicit rewards across different β values (they're scaled by β).
- Running DPO for too long with a tiny β: the model drifts far and degrades on everything else.

## Go deeper (optional, research-level)

1. Read *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (Rafailov et al., 2023) and derive equation (4) → (7) yourself.
2. Read *Training language models to follow instructions with human feedback* (InstructGPT, Ouyang et al., 2022), section 3. Draw the full three-stage pipeline.
3. Read *Constitutional AI: Harmlessness from AI Feedback* (Bai et al., Anthropic, 2022). How are the preference labels produced, and what are the advantages and risks of AI-generated feedback?
4. Reward hacking: train a reward model on your toy data, then optimize a policy against it *without* the KL penalty. What does the policy do?

## Your turn

Open the **Exercises** tab. `make_tasks` and the character set are given. Everything trains in a few seconds.
