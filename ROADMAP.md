# Roadmap: Python → research-level AI

The goal is to become excellent at **practical building**, **problem solving**, and **research-level understanding**. So instead of a single line of topics, the program runs three tracks side by side:

| Track | What it trains | How |
|---|---|---|
| **Build** | Writing real, working software | Modules with auto-graded exercises, plus a project at the end of each phase |
| **Solve** | Problem solving and algorithms | 3–5 problems a week from Phase 2 on, with complexity analysis |
| **Research** | Deep understanding, reading papers | "Go deeper" questions in each lesson, then paper reproductions from Phase 6 on |

Times assume about 10–12 hours a week. Go slower if you need to: depth matters more than speed.

---

## Phase 1 · Python foundations (5–6 weeks)

`m01` values, names and types · `m02` strings in depth · `m03` conditionals and boolean logic · `m04` loops and iteration patterns · `m05` lists and tuples · `m06` dicts and sets · `m07` functions, scope and recursion · `m08` errors and exceptions · `m09` files, paths and JSON · `m10` debugging

**Project (`m11`):** a command-line expense tracker that saves to JSON.

## Phase 2 · Problem solving and data structures (6–8 weeks, then ongoing)

`m12` complexity and a problem-solving framework · `m13` two pointers and sliding windows · `m14` stacks and queues · `m15` recursion and backtracking · `m16` sorting and binary search · `m17` linked lists · `m18` trees and BSTs · `m19` heaps and priority queues · `m20` graphs · `m21` dynamic programming · `m22` greedy algorithms and intervals

**Project (`m23`):** a ten-problem contest with no technique hints. After that, keep solving 3–5 problems a week on LeetCode/Codeforces, reviewed by the mentor.

## Phase 3 · Professional Python (6–8 weeks)

`m24` classes and objects · `m25` inheritance, composition and interfaces · `m26` the data model (dunder methods) · `m27` dataclasses, enums and type hints · `m28` iterators and generators · `m29` decorators and context managers · `m30` modules, packages and environments · `m31` testing with pytest (graded by mutation testing) · `m32` Git and GitHub · `m33` async and concurrency · `m34` performance and how CPython works

**Project (`m35`):** `searchkit`, a typed, tested search engine package with BM25 ranking and a CLI: the retrieval half of RAG.

## Phase 4 · Mathematics for AI, in code (6–8 weeks)

`m36` NumPy fundamentals · `m37` vectorization and broadcasting · `m38` linear algebra I: vectors, matrices, least squares · `m39` linear algebra II: eigenvectors, SVD, PCA · `m40` calculus: gradients and the chain rule · `m41` optimization: GD, momentum, Adam, schedules · `m42` probability · `m43` statistics: MLE, bootstrap, significance · `m44` information theory: entropy, cross-entropy, KL, perplexity

**Project (`m45`):** a recommender system with matrix factorization, compared against baselines and SVD on held-out ratings.

Every concept is implemented from scratch in NumPy before you use a library for it.

## Phase 5 · Data and classical machine learning (6–7 weeks)

`m46` pandas fundamentals · `m47` data cleaning · `m48` visualization · `m49` the ML workflow and evaluation · `m50` linear and logistic regression from scratch · `m51` decision trees and random forests from scratch · `m52` unsupervised learning: k-means, silhouette, anomalies · `m53` feature engineering and scikit-learn pipelines · `m54` gradient boosting from scratch

**Project (`m55`):** end-to-end churn prediction on a messy dataset with a hidden leakage trap, a cost-based decision threshold and a written report.

## Phase 6 · Deep learning (8–10 weeks)

`m56` an autograd engine from scratch (micrograd-style) · `m57` neural networks from scratch in NumPy · `m58` PyTorch fundamentals · `m59` training loops: data loaders, schedules, checkpoints, early stopping · `m60` initialization, BatchNorm/LayerNorm, dropout, weight decay, label smoothing · `m61` CNNs: convolution from scratch, im2col, augmentation · `m62` RNNs and LSTMs: vanishing gradients, a character-level language model · `m63` debugging training: overfit one batch, gradient norms, NaN hunting, a bug hunt · `m64` performance: FLOPs, memory, mixed precision, gradient accumulation, checkpointing

**Research project (`m65`):** reproduce the ResNet degradation result (deep plain nets train worse; residual connections fix it) over several seeds, investigate the gradient norms, run an ablation, and write it up.

## Phase 7 · Transformers and LLMs (8–10 weeks)

Tokenization (build BPE) · embeddings · attention from scratch · the Transformer architecture · build and train a small GPT · scaling laws · pretraining, fine-tuning, LoRA · RLHF and preference optimization (concepts) · inference: sampling, KV cache, quantization

**Research:** read and reproduce *Attention Is All You Need*, GPT-2 and LoRA at small scale.

## Phase 8 · LLM engineering (6–8 weeks)

Calling LLM APIs (this repo's mentor is your first case study: read `mentor/`) · prompt engineering · structured outputs and tool use · embeddings and vector search · RAG · agents · evaluating LLM systems · cost, latency and safety · deploying with FastAPI and Docker

**Project:** a production-quality RAG or agent app, deployed with evals.

## Phase 9 · Research skills and capstone (ongoing → 8+ weeks)

How to read a paper (three-pass method) · experiment design and baselines · tracking experiments (Weights & Biases) · statistics for comparing models · writing a technical blog post or paper · contributing to open-source AI projects

**Capstone:** an original project or a novel extension of a paper, published with code and a write-up.

---

## Weekly rhythm (suggested)

- **4 days:** a module from the current phase (`learn` → exercises → `check` → `quiz`)
- **2 days:** problem solving (from Phase 2 on) or a paper (from Phase 6 on)
- **1 day:** review: redo anything you struggled with, `python -m mentor tip`, and rest
