# m73 · Fine-tuning and LoRA

**By the end you can:** explain the stages that turn a pretrained model into an assistant, implement **LoRA** (low-rank adaptation) from scratch, apply it to a GPT, merge it back into the weights for free inference, prepare supervised fine-tuning (SFT) data with **loss masking**, and run a small experiment comparing LoRA with full fine-tuning on what they learn and what they forget.

**Why it matters for AI:** almost nobody pretrains a model; almost everybody adapts one. Fine-tuning is how a base model becomes a chatbot, a coding assistant or a domain expert, and LoRA (with its quantized cousin QLoRA) is how people fine-tune 7B–70B models on a single GPU. It's the most practical skill in this phase.

---

## 1. From base model to assistant

A **base model** (m70) just continues text. Ask it a question and it may continue with more questions. Assistants are built in stages:

1. **Pretraining** on trillions of tokens: knowledge and language (m70).
2. **Supervised fine-tuning (SFT)** on (prompt, ideal response) pairs: the model learns the *format* and *behaviour* of an assistant.
3. **Preference optimization** (RLHF, DPO; m74): the model learns which of two responses people prefer.

Fine-tuning also adapts models to **domains** (medicine, law, your company's code) by continuing training on domain text.

## 2. SFT data and loss masking

SFT examples are formatted with a **chat template**, for example `"Q: {prompt}\nA: {response}\n"` (real models use special tokens like `<|user|>`, m66). The model is trained with the usual next-token loss, but only on the **response** tokens: we don't want to teach it to generate user prompts. Set the targets of prompt positions (and padding) to **−100**, PyTorch's `ignore_index`, so they contribute nothing to the loss:

```
text:     Q : _ h i \n A : _ h e l l o \n
inputs:   every character except the last
targets:  -100 ... -100 (prompt)   h e l l o \n   -100 ... (padding)
```

`F.cross_entropy(logits, targets, ignore_index=-100)` averages over the remaining positions only.

## 3. Full fine-tuning is expensive

Fine-tuning all weights of a 7B model needs the 16 bytes per parameter of m64 (weights, gradients, Adam states): over 100 GB. And you'd store a full 14 GB copy of the model for every task.

## 4. LoRA: low-rank adaptation

Hu et al. (2021) observed that the weight *change* during fine-tuning has low "intrinsic rank". So freeze the pretrained weight W (out × in) and learn the update as a product of two thin matrices:

W' = W + (α / r) · B A,  with B: out × r and A: r × in, and r ≪ min(in, out)

- **A** starts random (N(0, 1/in) here), **B** starts at **zero**, so at the start W' = W exactly: fine-tuning begins from the pretrained model.
- Only A and B are trained: for a 4096 × 4096 matrix with r = 8, that's 65K parameters instead of 16.8M (0.4%).
- α / r scales the update so you can change r without retuning the learning rate much.
- LoRA usually needs a **higher learning rate** than full fine-tuning (here 1e-2 vs 3e-3).

The forward pass is `base(x) + (x @ A.T @ B.T) * scaling`: never form B A explicitly during training.

**Merging:** after training, compute W + (α/r)·B A once and replace the layer with a plain `nn.Linear`. Inference then costs exactly as much as the original model. Or keep the small A, B adapters separate and swap them per task: one base model, many cheap adapters.

**QLoRA** (Dettmers et al., 2023) keeps the frozen base weights in 4-bit (m71) and trains LoRA adapters in 16-bit, so a 65B model can be fine-tuned on one 48 GB GPU.

## 5. What LoRA learns and forgets

Fine-tuning on a new domain makes a model better there and often worse at what it knew before: **catastrophic forgetting**. Biderman et al. (2024) titled their paper *LoRA Learns Less and Forgets Less*: compared with full fine-tuning, LoRA improves less on the new domain but preserves more of the original abilities.

You'll reproduce this in miniature. Pretrain a tiny GPT on this course's English lessons, then fine-tune it on the course's **Python test code** two ways, and measure loss on both English and code:

| Model | Trainable params | Code loss ↓ | English loss (forgetting) |
|---|---|---|---|
| Base | all | ~3.05 | ~2.90 |
| LoRA (r = 8) | ~14% | ~2.6 | ~3.15 |
| Full fine-tuning | 100% | ~2.4 | ~3.30 |

(14% is large because this model is tiny and its embeddings are a big share; for a 7B model it'd be well under 1%.)

---

## Problem-solving habit #73: measure the side effects

Every intervention on a model (fine-tuning, quantization, pruning, safety training) has side effects. Don't only measure the metric you're trying to improve: keep an evaluation of the things you want to *preserve* (general ability, other languages, safety), and report both. A result that improves one number while silently damaging another isn't a result.

## Common mistakes

- Initializing both A and B randomly, so the model is perturbed before training even starts.
- Forgetting to freeze the base weights (then it's just full fine-tuning with extra steps).
- Training on prompt tokens in SFT (no loss masking).
- Using full fine-tuning learning rates for LoRA and concluding LoRA "doesn't work".
- Evaluating only on the fine-tuning domain.

## Go deeper (optional, research-level)

1. Read *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021), sections 4 and 7. Which weight matrices did they find most worth adapting? Try `targets=("qkv",)` vs all layers in your experiment.
2. Read *LoRA Learns Less and Forgets Less* (Biderman et al., 2024). Measure the effective rank (m39's SVD) of the weight change W_full − W_base after your full fine-tuning. Is it low?
3. Read *QLoRA* (Dettmers et al., 2023). Combine your m71 int4 quantization with LoRA: quantize the frozen base, train adapters, and compare the code loss with plain LoRA.

## Your turn

Open the **Exercises** tab. `load_corpora` is given. The final experiment pretrains a tiny GPT and fine-tunes it twice (about 40 seconds on a CPU).
