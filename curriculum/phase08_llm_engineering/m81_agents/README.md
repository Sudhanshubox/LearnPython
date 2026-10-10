# m81 · Agents and workflows

**By the end you can:** choose between a single call, a fixed workflow and an autonomous agent; implement the core workflow patterns (prompt chaining, routing, parallel voting, evaluator–optimizer); and build a tool-using agent with the guardrails real agents need: a sandboxed workspace, human approval for risky actions, step and token budgets, and a full trajectory log.

**Why it matters for AI:** "agents" are the fastest-moving area in applied AI, and also the most over-applied. Knowing the simple patterns that solve most problems, and how to make a real agent safe and debuggable when you do need one, is what separates useful systems from expensive demos.

---

## 1. Start simple

From Anthropic's *Building effective agents* (2024): use the **simplest** thing that works.

| Tier | What | When |
|---|---|---|
| Single call | One prompt, one answer (m76–m78) | Classification, extraction, summarization, Q&A |
| **Workflow** | LLM calls orchestrated by *your code* along fixed paths | The steps are known in advance |
| **Agent** | The *model* decides which tools to call, in a loop, until done | Open-ended tasks where the steps can't be scripted |

Agents trade predictability, latency and cost for flexibility. Before building one, ask: Is the task genuinely open-ended? Is it worth the cost? Can mistakes be caught and undone?

## 2. Workflow patterns

- **Prompt chaining:** a sequence of calls, each transforming the previous output (outline → draft → polish). Each step is easy, and you can check outputs between steps.
- **Routing:** classify the input, then send it to a specialised handler (billing questions to one prompt, bug reports to another). Unknown categories go to a safe default.
- **Parallelization / voting:** run the same prompt several times (concurrently, m33) and take the majority answer. The agreement rate is a useful confidence signal: 5/5 is safer than 3/5.
- **Evaluator–optimizer:** one call drafts, another critiques against the task, and the draft is revised until the critic says it passes (or a round limit is reached).
- **Orchestrator–workers:** a model breaks a task into subtasks, workers solve them, and the orchestrator combines the results (Phase 9's capstone can explore this).

## 3. Anatomy of an agent

An agent is m78's tool loop plus everything that makes it safe to leave running:

```
while steps < max_steps and tokens < budget:
    response = model(messages, tools)
    if no tool calls: return the answer
    for each tool call:
        if the tool needs approval and the human says no: result = "The user declined this action." (error)
        else: run it (errors become is_error results)
        log it in the trajectory
```

- **Budgets.** Cap steps *and* tokens. Report *why* the agent stopped (`end_turn`, `max_steps`, `token_budget`) so callers can handle each case.
- **Human in the loop.** Reading is cheap to allow; writing, sending, paying and deleting should need approval. A declined action is returned to the model as an error result, so it can adapt ("OK, I won't overwrite it; here's the summary instead").
- **Least privilege and sandboxing.** Tools should only reach what the task needs. File tools must stay inside a workspace directory: resolve the path and reject anything outside it. `../../etc/passwd` is the classic **path traversal** attack, and an agent that has read a malicious document might try it.
- **Trajectory logging.** Record every tool call, input, output, approval and the final answer. When an agent misbehaves, the trajectory is the only way to understand why, and it's also your evaluation data (m82).

## 4. Production options

You built the loop yourself to understand it. In practice:

- The SDK's **tool runner** (m78) runs the loop for tools you define.
- **Claude Managed Agents** runs the loop *and* hosts a sandboxed workspace (bash, files, code execution) for each session.
- The **Claude Agent SDK** packages the Claude Code harness (file editing, bash, search, subagents) as a library.

The guardrails in this module (budgets, approval, sandboxing, logging) apply to all of them.

---

## Problem-solving habit #81: make autonomy reversible

Before giving a system autonomy, make its actions **observable** (logs), **bounded** (budgets, permissions) and **reversible** (approval for irreversible steps, writes to a scratch area, version control). Then increase autonomy step by step as you gain evidence that it behaves well. It's how you'd onboard a new colleague, too.

## Common mistakes

- Building an agent for a task a three-step workflow would solve.
- No step or cost limit.
- File tools that accept any path.
- Letting a tool exception crash the run instead of returning it to the model.
- No trajectory log: when it goes wrong, nobody can tell why.
- Asking for approval for everything (people stop reading the prompts) or for nothing.

## Go deeper (optional)

1. Read Anthropic's *Building effective agents* (2024) and identify which pattern each of these uses: m80's RAG pipeline, the mentor in your right panel, Claude Code.
2. Read the ReAct paper (Yao et al., 2023) and *SWE-bench* (Jimenez et al., 2024). What makes software engineering a good benchmark for agents?
3. Give your agent a `run_python` tool. Then list everything that could go wrong and how you'd sandbox it (containers, timeouts, no network, resource limits).

## Your turn

Open the **Exercises** tab. The agent tests run a scripted "model" through a small workspace in a temporary folder.
