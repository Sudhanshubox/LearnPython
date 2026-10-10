# m84 · Deploying an LLM service with FastAPI and Docker

**By the end you can:** turn an LLM function into a production-style web API with FastAPI (typed request and response models, validation, API-key authentication, per-key rate limiting, request IDs, structured logs, safe error handling, and a streaming endpoint), configure it from environment variables, test it without a server, and package it in a Docker image.

**Why it matters for AI:** a model in a notebook helps one person. An API helps a whole product, a team, or thousands of users. Almost every LLM feature ships as a service like this one, and interviewers for AI engineering roles expect you to be able to build and reason about it.

---

## 1. FastAPI in five minutes

FastAPI builds web APIs from type-annotated Python functions (m27), using Pydantic (m78) for validation:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    max_sources: int = Field(default=3, ge=1, le=10)

@app.post("/ask")
def ask(body: AskRequest):
    return {"answer": "..."}
```

- The JSON body is parsed and **validated** automatically; invalid input gets a **422** response describing the problem, before your code runs.
- `response_model=` validates and documents what you return.
- Visit `/docs` on a running server for interactive documentation generated from your types.
- Run it with `uvicorn app:app --reload` while developing.

## 2. Design the endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check for load balancers and monitoring: cheap, no auth, no LLM call |
| `POST /ask` | Question in, `{answer, sources, request_id}` out |
| `POST /ask/stream` | The same, streamed as server-sent events (m76): `data: ...\n\n` per chunk, then a `done` event |

Limit input sizes (here 2,000 characters): an unbounded question is an unbounded bill.

## 3. Production concerns, one by one

- **Authentication:** clients send an `X-API-Key` header; unknown or missing keys get **401**. (Real systems use OAuth or signed tokens; the principle is the same.) Never put *your* Anthropic API key in the client: it stays on the server.
- **Rate limiting:** a token bucket (m83) **per API key**, so one heavy user can't starve the rest. Over the limit: **429** with a `Retry-After` header.
- **Request IDs:** a middleware gives every request an ID (reusing the client's `X-Request-ID` if sent), returns it in the response headers, and includes it in logs and error bodies. When a user reports a problem, the ID finds the exact log lines.
- **Structured logs:** one record per request with method, path, status and latency, as data (dicts / JSON lines), not prose, so you can query them. Don't log full prompts containing personal data (m83).
- **Errors:** if the LLM call fails, return **503** with a friendly message and the request ID. Never send stack traces or exception text to clients (they can leak internals).
- **Configuration:** settings come from **environment variables** (API keys, rate limits, model name), validated at startup, so a missing setting fails immediately with a clear message instead of at the first request. This is the "twelve-factor app" rule: config in the environment, never in the code.

`create_app(settings, answer_fn, ...)` is an **app factory**: it receives its dependencies as arguments. Tests pass a fake `answer_fn` and a fake clock; production passes the real RAG pipeline (m80). FastAPI's `TestClient` calls the app in-process, so the tests need no running server.

## 4. Docker

A **container image** packages your code, its dependencies and the Python version into one artifact that runs the same on your laptop and in the cloud:

```dockerfile
FROM python:3.12-slim                         # small official base image
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .                       # dependencies first: this layer is cached
RUN pip install --no-cache-dir -r requirements.txt
COPY . .                                      # then the code, which changes often
RUN useradd --create-home appuser
USER appuser                                  # don't run as root
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t course-assistant .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e APP_API_KEYS=dev-key course-assistant
```

Secrets are passed at **run time** with `-e` (or your platform's secret manager), never written into the image with `ENV ANTHROPIC_API_KEY=...`: anyone who can pull the image can read its layers. Add a `.dockerignore` (like `.gitignore`) so `.venv`, `.git` and local data don't end up in the image.

From there, the image runs on any container platform (Cloud Run, AWS App Runner or ECS, Fly.io, Kubernetes). Put HTTPS and the platform's own protections in front of it.

---

## Problem-solving habit #84: make it observable before it's live

Before launching, ask: when this breaks at 2 a.m., what will I look at? Health checks, request IDs, structured logs with latency and status, and cost per request (m83) should exist *before* the first real user. Debugging a production system you can't see into is guesswork.

## Common mistakes

- Calling the LLM in `/health` (slow, and costs money every few seconds).
- One global rate limit, or none.
- Returning `str(exception)` to clients.
- Reading environment variables deep inside request handlers instead of once at startup.
- `ENV ANTHROPIC_API_KEY=sk-ant-...` in a Dockerfile, or running as root.
- Copying the code before installing requirements (every code change reinstalls everything).

## Go deeper (optional)

1. Run your app for real: `uvicorn` locally, then in Docker. Open `/docs` and try every endpoint, including the error cases.
2. Read *The Twelve-Factor App* (12factor.net). Which factors does this module follow, and which are missing (hint: backing services, logs as event streams)?
3. Add an async version of `/ask` using `anthropic.AsyncAnthropic`. Load test both with 50 concurrent requests and compare throughput. Why does async help for I/O-bound LLM calls?

## Your turn

Open the **Exercises** tab. The tests use FastAPI's `TestClient`; no server or network is needed.
