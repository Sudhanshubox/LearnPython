# m78 · Structured outputs and tool use

**By the end you can:** get guaranteed, validated JSON out of Claude with structured outputs and Pydantic, define tools with strict JSON schemas, validate tool inputs yourself, build a tool registry, write the tool-use loop by hand (and know when the SDK's tool runner can do it for you), and execute model-chosen actions safely.

**Why it matters for AI:** this is the bridge from "a model that writes text" to "a model that does things". Extracting data into your database, calling your APIs, querying search, running calculations: all of it goes through structured outputs and tools. It's also where security matters most, because the model decides what code paths run.

---

## 1. Structured outputs: JSON you can trust

Asking "reply in JSON" works most of the time, which in production means it fails some of the time. **Structured outputs** constrain the model's generation to a JSON schema, so the reply always parses and validates. The nicest interface uses a Pydantic model (m27's dataclasses, with validation):

```python
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    category: Literal["billing", "bug", "feature_request", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str
    customer_email: str | None          # None when the message has no email

response = client.messages.parse(
    model="claude-opus-5-5", max_tokens=1024,
    messages=[{"role": "user", "content": email_text}],
    output_format=Ticket,
)
ticket = response.parsed_output          # a validated Ticket instance
```

`Literal[...]` becomes an `enum` in the schema: the model can't invent a fifth category. `str | None` lets it say "not present" instead of making something up. Under the hood the SDK sends `output_config={"format": {"type": "json_schema", "schema": ...}}`.

## 2. Tools: letting the model act

A **tool** is a function you describe to the model with a name, a description and a JSON schema for its input:

```python
{
    "name": "get_weather",
    "description": "Current weather for a city. Use when the user asks about weather.",
    "strict": True,                                  # guarantee schema-valid inputs
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
        "required": ["city", "unit"],
        "additionalProperties": False,               # required by strict mode
    },
}
```

The model never runs anything itself. It replies with `stop_reason == "tool_use"` and one or more `tool_use` blocks (each with an `id`, a `name` and an `input`); **your code** runs the function and sends back a `tool_result`. Good descriptions matter as much as good prompts: say *when* to use the tool, not just what it does.

Recent models (including Claude Opus 5.5) don't allow forcing a specific tool with `tool_choice`; they decide based on the request. Steer with the prompt ("use the get_weather tool"), and use structured outputs when you just want JSON back.

## 3. The tool-use loop

```
messages = [user question]
loop:
    response = create(messages, tools)
    append {"role": "assistant", "content": response.content}
    if stop_reason != "tool_use": done; the text is the answer
    for every tool_use block: run it -> a tool_result {tool_use_id, content, is_error?}
    append ONE user message containing ALL the tool_results
```

Details that matter:

- The model may request **several tools at once** (parallel tool use). Run them all and return all results **in a single user message**.
- Each `tool_result` must carry the matching `tool_use_id`.
- If a tool fails, don't crash and don't hide it: return the error text with `"is_error": True`. The model can often recover (fix its input, try another tool, or explain the problem).
- Always cap the number of turns, so a confused model can't loop forever (and run up your bill).

The SDK's **tool runner** (`client.beta.messages.tool_runner` with `@beta_tool`-decorated functions) runs this loop for you, and is the right choice in most applications. Write it by hand once (in the exercises) so you know exactly what it does.

## 4. Validate, then execute

With `strict: True` the API guarantees schema-valid inputs, but validate anyway at your boundary: not every path is strict (streamed inputs, other providers, your own tests), and validation errors make great `is_error` messages. You'll write a small JSON-schema validator covering types, `enum`, `required`, `additionalProperties` and arrays.

## 5. Security: the model is not trusted

Tool inputs come from a model that may have read malicious text (a web page, an email, a document: prompt injection, m83). Treat tool inputs exactly like untrusted user input:

- **Never** `eval()` or `exec()` model output, or pass it to a shell. The exercises build a calculator that parses arithmetic with `ast` and evaluates only numbers and operators: `__import__('os').system('rm -rf /')` is rejected, not run.
- Give tools the **least privilege** they need (read-only where possible), validate every argument, and put limits on sizes and costs.
- Require **human approval** for actions with real-world effects (sending emails, payments, deleting data).

---

## Problem-solving habit #78: make invalid states unrepresentable

Design schemas so wrong outputs can't be expressed: enums instead of free text, `None` instead of an empty-string convention, required fields instead of optional ones with defaults you have to guess. The same idea makes ordinary code safer (m27): the fewer ways a value can be wrong, the fewer checks you need everywhere else.

## Common mistakes

- Parsing "JSON" from free text with `json.loads(reply)` and no fallback.
- Returning tool results in separate user messages, or forgetting `tool_use_id`.
- Raising an exception out of the loop when a tool fails, instead of returning `is_error`.
- No turn limit.
- Vague tool descriptions ("does stuff with data").
- `eval(tool_input["expression"])`.

## Go deeper (optional)

1. Read the tool use and structured outputs guides in Anthropic's docs. What do `strict: true` and `output_config.format` each guarantee, and how do they combine?
2. Rewrite `run_tool_loop` using the SDK's tool runner and `@beta_tool`, and compare the code.
3. Read *Toolformer* (Schick et al., 2023) and *ReAct* (Yao et al., 2023). How did tool use evolve from research prompts to an API feature?

## Your turn

Open the **Exercises** tab. Everything runs against the fake API.
