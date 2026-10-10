# m85 · Project: "Ask the course", a production RAG assistant

**Phase 8 finale.** Build a complete, deployable assistant that answers questions about this course from its own lessons, with citations, and *prove* how well it works. This is the kind of system AI engineers build and maintain every day, and a strong portfolio piece: put it on GitHub with your evaluation report.

It brings together every module in this phase: API calls and refusal fallbacks (m76), prompt structure (m77), retrieval (m79–m80), evaluation with an LLM judge and confidence intervals (m82), cost tracking and caching (m83), and a FastAPI service (m84).

---

## Architecture

```
lessons (Phases 1-7) ─▶ section-aware chunks ─▶ keyword + vector indexes
                                                      │
question ─▶ hybrid retrieval (RRF) ─▶ context budget ─▶ grounded prompt ─▶ Claude (with fallbacks)
                                                                              │
          ◀── {answer, cited lessons, cost, cached?, refused?} ◀── citation check
                                                                              │
FastAPI: GET /health, POST /ask (API key) ─────────────────────────────────────┘
evaluation: eval_set.json (20 questions) ─▶ retrieval hit rate, citation hit rate, judge score ± CI, cost
```

## What's given

- `retrievers.py`: the keyword and vector retrievers from m80.
- In `exercises.py`: chunking and loading (from m80), `load_eval_set`, `escape_xml`.
- `eval_set.json`: 20 learner questions, each with the lesson that answers it and a short reference answer. **Add your own**: the best evaluation questions are the ones you actually asked the mentor while studying.
- `fakeclaude.py` for offline tests.

## Steps (`exercises.py`)

1. `HybridRetriever`: keyword + vector search fused with reciprocal rank fusion.
2. `build_prompt`: the grounded, citation-requesting prompt.
3. `message_cost`: dollars per call from `usage`.
4. `CourseAssistant.retrieve` and `.ask`: the pipeline, with server-side refusal fallbacks, a friendly refusal message, `AssistantUnavailable` on API errors, citation checking, cost tracking and a response cache.
5. `judge` and `evaluate`: the evaluation harness.
6. `create_app`: the FastAPI service.
7. `write_report`: your evaluation report.

## Running it for real

With an API key (m76):

```python
# app.py, next to exercises.py
import anthropic
from exercises import CourseAssistant, create_app, load_course_chunks

assistant = CourseAssistant(anthropic.Anthropic(), load_course_chunks())
app = create_app(assistant, api_keys={"dev-key"})
```

```bash
uvicorn app:app --reload
curl -X POST localhost:8000/ask -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
     -d '{"question": "What does the KV cache store?"}'
```

Then run `evaluate(assistant, anthropic.Anthropic(), load_eval_set())` and write the report. Expect a few cents per question with Claude Opus 5.5 at low effort, and check the cost line in your report.

## Report outline (`write_report`)

```markdown
# Ask the course: evaluation report
## System            (what you built, in a few sentences)
## Results           (a table: retrieval hit rate, citation hit rate, judge score with its CI,
                      judge failures, refusal rate, cost per question)
## Failure analysis  (the low-scoring questions, and which pipeline stage failed for each, m80 habit #80)
## Next steps        (what you'd change next, and how you'd measure it)
```

## Stretch goals (optional)

- **Streaming:** add `POST /ask/stream` (m84) that streams the answer and sends the sources at the end.
- **Conversation memory:** let follow-up questions refer to earlier ones ("and how is that different from a tuple?"). Rewrite the follow-up into a standalone question before retrieval.
- **Reranking:** have Claude choose the 5 best of the top 20 chunks (m80 Go deeper) and measure the change with a paired bootstrap (m82).
- **Connect it to the mentor:** add a "Search the course" button to the study screen (`mentor/web/`) that calls your service.
- **Deploy it:** build the Docker image (m84) and run it on a cloud container service, with your key in the platform's secret store.

## Go deeper (optional)

1. Read *Evaluating Retrieval Quality in Retrieval-Augmented Generation* (Salemi & Zamani, 2024) or the RAGAS paper (Es et al., 2023). Which of their metrics would you add?
2. Collect 50 real questions from your own study sessions, label them, and compare your assistant with simply asking the mentor (no retrieval). Where does each win?

## Your turn

Open the **Exercises** tab. All the tests run offline with a fake model and a fake judge.
