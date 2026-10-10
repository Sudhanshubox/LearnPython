# m76 · Calling LLM APIs

**By the end you can:** call Claude from Python with the official `anthropic` SDK, manage a multi-turn conversation correctly, stream replies, read stop reasons and usage, compute what a call costs, handle every kind of API error, retry the right failures with exponential backoff, and handle refusals with server-side fallbacks. Then you'll read this repo's own mentor as a real-world case study.

**Why it matters for AI:** Phases 6–7 taught you how LLMs work inside. Most AI engineering jobs are about building reliable products *on top of* them, and that starts with the API call. The details in this module (statelessness, stop reasons, retries, cost) are where real applications break.

---

## 1. Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # from console.anthropic.com
```

```python
import anthropic
client = anthropic.Anthropic()           # reads ANTHROPIC_API_KEY from the environment
```

Never put the key in your code or commit it to Git (m32). The exercises in this module and the rest of Phase 8 run **offline**: the tests give your functions a client connected to `fakeclaude.py`, a fake API that returns scripted replies through the real SDK. Your code doesn't know the difference.

## 2. The basic request

```python
message = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,                                   # a hard cap on the reply's length
    system="You are a concise Python tutor.",          # instructions about behaviour
    messages=[{"role": "user", "content": "What is a list comprehension?"}],
    output_config={"effort": "low"},                   # low | medium | high | xhigh | max
)
text = "".join(b.text for b in message.content if b.type == "text")
```

- `message.content` is a **list of blocks** (text, thinking, tool_use, ...). Always check `block.type`.
- **Effort** trades depth of reasoning against cost and latency. Claude Opus 5.5 thinks before answering on every request; effort controls how much. Use `low` for simple tasks, higher for hard ones, and measure.
- `message.usage` reports input and output tokens: that's what you pay for.

## 3. The API is stateless

The model remembers nothing between calls. A conversation is a list you send **in full** every time:

```python
messages = [
    {"role": "user", "content": "My name is Asha."},
    {"role": "assistant", "content": reply1.content},     # the full blocks, not just text
    {"role": "user", "content": "What's my name?"},
]
```

Two rules that prevent subtle bugs:

1. **Append the assistant's full `content`** (all its blocks), not a re-typed string. Some blocks (thinking, tool use) must be sent back unchanged.
2. **Only commit a turn to the history after the call succeeds.** If the request fails, the history must be exactly as before, so a retry doesn't send the user's message twice.

## 4. Streaming

Long replies take seconds. Streaming shows text as it's generated, which feels much faster:

```python
with client.messages.stream(model=..., max_tokens=..., messages=messages) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()        # the complete Message, with usage and stop_reason
```

Use streaming for any request with long input or output: it also avoids HTTP timeouts.

## 5. Stop reasons

Always check `message.stop_reason` before trusting the content:

| stop_reason | Meaning | What to do |
|---|---|---|
| `end_turn` | Finished normally | Use the reply |
| `max_tokens` | Hit your `max_tokens` cap mid-reply | Raise the cap or ask it to continue |
| `tool_use` | Wants to call a tool (m78) | Run the tool and send the result |
| `refusal` | Declined for safety reasons | Don't treat empty content as an answer |

## 6. What it costs

You pay per token, with different prices for input and output (output is usually 5× more expensive). With **prompt caching** (m83), repeated prefixes are read from cache at a fraction of the input price, and writing to the cache costs a bit more than normal input. For Claude Opus 5.5: $4 per million input tokens, $20 per million output, $0.20 per million cached-input reads. A typical chat turn of 2,000 input + 500 output tokens costs 0.8 + 1.0 = 1.8 cents. Multiply by your users before you launch.

## 7. Errors, and which ones to retry

The SDK raises typed exceptions. Catch them **most specific first**:

| Exception | Status | Retry? |
|---|---|---|
| `AuthenticationError` | 401 | No: fix the key |
| `BadRequestError` | 400 | No: fix the request |
| `RateLimitError` | 429 | Yes, after waiting (respect the `retry-after` header) |
| `InternalServerError` and other 5xx (529 = overloaded) | 5xx | Yes |
| `APIConnectionError` | network | Yes |

**Exponential backoff with jitter:** wait base · 2^attempt (capped), times a random factor in [0.5, 1]. Doubling backs off quickly from an overloaded service; the randomness stops thousands of clients from retrying in lockstep. The SDK already retries twice by default (`max_retries`); you write your own loop when you need more control (logging, a total time budget, user-visible messages).

## 8. Refusals and fallbacks

Safety classifiers occasionally decline a request (`stop_reason == "refusal"`), sometimes wrongly for legitimate work. With **server-side fallbacks**, the API automatically re-runs a declined request on an appropriate fallback model within the same call:

```python
message = client.beta.messages.create(
    model="claude-opus-5-5", max_tokens=4096, messages=messages,
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
)
if message.stop_reason == "refusal":      # the whole chain declined
    ...
```

## 9. Case study: this repo's mentor

Open `mentor/client.py` (the mentor in your right-hand panel) and find each idea from this module:

- `ask()` streams with `client.beta.messages.stream`, calls `on_text` for each chunk, and returns the final message.
- It sets adaptive thinking, an effort level from `config.py`, **prompt caching** (`cache_control`, m83), and **server-side fallbacks**.
- It maps each SDK exception to a `MentorError` with a message a learner can act on.
- It checks `stop_reason` for `refusal` and `max_tokens` and tells the user what happened.
- `session()` appends `reply.content` (full blocks) to the history.

Then open `mentor/chat.py` and check how it keeps the history unchanged when a send fails. Ask the mentor: "what would you improve in your own client code?"

---

## Problem-solving habit #76: design for failure first

Networks drop, services overload, models refuse, outputs get cut off. For every external call, write down what happens when it fails, before writing the happy path. Code that handles failure deliberately is the difference between a demo and a product.

## Common mistakes

- Reading `message.content[0].text` without checking the block type.
- Sending only the latest message and expecting the model to remember the conversation.
- Appending the user's turn to the history before the call succeeds (duplicates on retry).
- Retrying 400 errors (they'll fail forever) or retrying 429s immediately (makes it worse).
- Hard-coding an API key.

## Go deeper (optional)

1. Read the Messages API reference and the "Handling stop reasons" guide at docs.anthropic.com (platform.claude.com/docs). What does `pause_turn` mean?
2. Read the AWS Architecture Blog post *Exponential Backoff and Jitter* (Marc Brooker). Simulate "full jitter" vs "equal jitter" vs no jitter with 100 clients and compare the load spikes.
3. With a real API key: measure the latency to the first streamed token and the total time for `low` vs `high` effort on the same question. How do they compare?

## Your turn

Open the **Exercises** tab. Every function takes a `client`; the tests pass in a fake one. Once they pass, try `ask(anthropic.Anthropic(), "...")` with your real key.
