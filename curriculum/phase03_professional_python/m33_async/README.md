# m33 · Async and concurrency

**By the end you can:** explain the difference between concurrency and parallelism, use `asyncio` to run many I/O-bound tasks at once, limit concurrency, add timeouts and retries, stream results, and know when to use threads or processes instead.

**Why it matters for AI:** calling an LLM API takes seconds per request. Evaluating a model on 5,000 prompts one at a time takes hours; with 20 concurrent requests, it takes minutes. But APIs have **rate limits**, so you must cap concurrency and retry politely. Streaming chat responses, web servers that serve models (FastAPI is async), and data downloading all use these tools. This repo's own mentor streams replies the same way.

---

## 1. Concurrency vs parallelism

- **Concurrency:** dealing with many things at once, by switching between them while each one *waits* (for the network, the disk, an API). One cook juggling several dishes.
- **Parallelism:** *doing* many things at the same instant on several CPU cores. Several cooks.

Most AI application code waits on I/O (APIs, databases, files), so concurrency gives huge speed-ups. CPU-heavy work (number crunching) needs parallelism, or better, NumPy/PyTorch, which run in optimized C/CUDA.

## 2. async / await

```python
import asyncio

async def fetch(prompt):                 # an async function ("coroutine function")
    await asyncio.sleep(1)               # pretend to wait for the network
    return f"answer to {prompt}"

async def main():
    a = await fetch("q1")                # waits 1s
    b = await fetch("q2")                # waits another 1s: still sequential!
    return [a, b]

asyncio.run(main())                       # start the event loop
```

- Calling `fetch("q1")` doesn't run it; it creates a **coroutine** object.
- `await` runs it, and while it waits, the **event loop** can run other tasks.
- `asyncio.run()` is the entry point from normal code.

## 3. Running things concurrently

```python
async def main():
    results = await asyncio.gather(fetch("q1"), fetch("q2"), fetch("q3"))
    return results                       # about 1s total, not 3s; results stay in order
```

`asyncio.gather` runs coroutines concurrently and returns results in the order you passed them. `asyncio.create_task(coro)` starts a coroutine in the background so you can do other things and `await` it later.

## 4. Limiting concurrency: semaphores

Firing 5,000 requests at once will get you rate-limited (HTTP 429) or banned. A **semaphore** allows at most N tasks inside a block at the same time:

```python
sem = asyncio.Semaphore(10)

async def limited_fetch(prompt):
    async with sem:                      # waits here if 10 are already running
        return await fetch(prompt)

results = await asyncio.gather(*(limited_fetch(p) for p in prompts))
```

## 5. Timeouts, retries and backoff

```python
try:
    result = await asyncio.wait_for(fetch(prompt), timeout=30)
except asyncio.TimeoutError:
    result = None
```

When a request fails, wait and retry, doubling the wait each time (**exponential backoff**: 1s, 2s, 4s…). This gives an overloaded server time to recover instead of hammering it.

## 6. Queues: producers and workers

For a stream of jobs, start a fixed number of **worker** tasks that take jobs from an `asyncio.Queue`:

```python
async def worker(queue, results):
    while True:
        item = await queue.get()
        results[item] = await process(item)
        queue.task_done()
```

## 7. Async generators: streaming

```python
async def stream_tokens(text):
    for token in text.split():
        await asyncio.sleep(0.05)
        yield token                      # an async generator

async for token in stream_tokens("hello there"):
    print(token, end=" ", flush=True)
```

This is how chat UIs show an answer word by word as the model produces it.

## 8. Threads, processes and the GIL

CPython has a **Global Interpreter Lock (GIL)**: only one thread runs Python bytecode at a time. So:

| Workload | Use |
|---|---|
| Many I/O waits, async-friendly libraries | `asyncio` |
| I/O with blocking (non-async) libraries | threads: `concurrent.futures.ThreadPoolExecutor` |
| CPU-heavy pure Python | processes: `ProcessPoolExecutor` (each has its own interpreter and GIL) |
| Numeric work | NumPy / PyTorch: they release the GIL and use all cores or the GPU |

(Python 3.13 added an experimental "free-threaded" build without the GIL; it's an active area of change.)

**Never call a blocking function** like `time.sleep()` or `requests.get()` inside async code: it freezes the whole event loop, and every other task stops too. Use the async version, or `await asyncio.to_thread(blocking_func, ...)`.

---

## Problem-solving habit #31: find what you're waiting for

Before optimizing, ask whether the program is slow because it *computes* (CPU-bound) or because it *waits* (I/O-bound). Measure it: if the CPU is mostly idle while the program runs, you're waiting, and concurrency is the fix. If a core is at 100%, you need a better algorithm, vectorization or parallelism.

## Go deeper (optional, research-level)

1. Read how this repo's mentor streams responses (`mentor/client.py` and `mentor/web/server.py`). It uses a *synchronous* stream and threads. How would you rewrite it with `AsyncAnthropic` and an async web framework?
2. Read the PEP 703 summary ("Making the Global Interpreter Lock Optional"). Why did the scientific Python and AI community push for it?
3. What is "structured concurrency"? Read about `asyncio.TaskGroup` (Python 3.11) and explain why it's safer than loose `create_task` calls when one task fails.

## Your turn

Open the **Exercises** tab, solve each part, then press **Run tests**. The tests use fake "API calls" built with `asyncio.sleep`, and measure timing and concurrency.
