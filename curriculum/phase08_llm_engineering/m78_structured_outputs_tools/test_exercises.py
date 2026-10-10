import json

import pytest

from code_checks import names_used
from exercises import (
    MODEL,
    Ticket,
    TooManyTurns,
    ToolRegistry,
    extract_ticket,
    make_tool,
    run_tool_loop,
    safe_calculate,
    validate,
)
from fakeclaude import FakeClaude, text_reply, tool_reply


def test_ticket_model():
    t = Ticket(category="bug", priority="high", summary="App crashes", customer_email=None)
    assert t.customer_email is None
    with pytest.raises(Exception):
        Ticket(category="complaint", priority="high", summary="x", customer_email=None)
    schema = Ticket.model_json_schema()
    assert schema["properties"]["priority"]["enum"] == ["low", "medium", "high"]
    assert set(schema["required"]) == {"category", "priority", "summary", "customer_email"}


def test_extract_ticket():
    fake = FakeClaude()
    fake.add_text(json.dumps({"category": "billing", "priority": "high",
                              "summary": "Charged twice", "customer_email": "asha@example.com"}))
    ticket = extract_ticket(fake.client(), "I was charged twice! asha@example.com")
    assert isinstance(ticket, Ticket) and ticket.category == "billing"
    body = fake.bodies[-1]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "low"
    assert body["system"] == "Extract a support ticket from the customer's message."


def test_make_tool():
    tool = make_tool("get_weather", "Weather for a city.", {"city": {"type": "string"}, "unit": {"type": "string"}})
    assert tool == {"name": "get_weather", "description": "Weather for a city.", "strict": True,
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                                     "required": ["city", "unit"], "additionalProperties": False}}
    assert make_tool("t", "d", {"a": {"type": "string"}}, required=[])["input_schema"]["required"] == []


SCHEMA = {"type": "object", "required": ["city", "days"], "additionalProperties": False,
          "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                         "unit": {"type": "string", "enum": ["c", "f"]},
                         "tags": {"type": "array", "items": {"type": "string"}},
                         "note": {"type": ["string", "null"]}, "ratio": {"type": "number"}}}


def test_validate():
    assert validate(SCHEMA, {"city": "Pune", "days": 3, "unit": "c", "tags": ["a"], "note": None, "ratio": 2}) == []
    errs = validate(SCHEMA, {"city": 5, "unit": "k", "tags": ["a", 2], "extra": 1})
    joined = " | ".join(errs)
    assert "$.days" in joined and "$.city" in joined and "$.unit" in joined
    assert "$.tags[1]" in joined and "$.extra" in joined
    assert len(errs) == 5
    assert validate(SCHEMA, {"city": "x", "days": True}) != [], "booleans are not integers"
    assert validate({"type": "number"}, 3) == [] and validate({"type": "number"}, 2.5) == []
    assert len(validate({"type": "object"}, [1])) == 1


@pytest.fixture
def registry():
    reg = ToolRegistry()

    @reg.register("add", "Add two integers.", {"a": {"type": "integer"}, "b": {"type": "integer"}})
    def add(a, b):
        return a + b

    @reg.register("divide", "Divide a by b.", {"a": {"type": "number"}, "b": {"type": "number"}})
    def divide(a, b):
        return a / b

    @reg.register("echo", "Echo text.", {"text": {"type": "string"}})
    def echo(text):
        return text

    assert add(2, 3) == 5, "the decorator must return the function unchanged"
    return reg


def test_registry(registry):
    assert [d["name"] for d in registry.definitions()] == ["add", "divide", "echo"]
    assert registry.definitions()[0]["strict"] is True
    assert registry.execute("add", {"a": 2, "b": 3}) == ("5", False)
    assert registry.execute("echo", {"text": "hi"}) == ("hi", False)
    assert registry.execute("nope", {}) == ("Error: unknown tool 'nope'", True)
    content, err = registry.execute("add", {"a": "2", "b": 3})
    assert err and content.startswith("Error: invalid input: ") and "$.a" in content
    assert registry.execute("divide", {"a": 1, "b": 0}) == ("Error: ZeroDivisionError: division by zero", True)


def test_tool_loop(registry):
    fake = FakeClaude()
    fake.add(tool_reply([("add", {"a": 2, "b": 3}), ("divide", {"a": 1, "b": 0})], text="Let me compute."))
    fake.add(tool_reply([("echo", {"text": "done"})]))
    fake.add(text_reply("2 + 3 = 5, and you can't divide by zero."))
    text, messages = run_tool_loop(fake.client(), registry, "What is 2+3 and 1/0?", system="Be precise.")
    assert text == "2 + 3 = 5, and you can't divide by zero."
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    second = fake.bodies[1]
    assert second["system"] == "Be precise." and second["tools"] == registry.definitions()
    results = second["messages"][2]["content"]
    assert len(results) == 2, "all tool results go in ONE user message"
    first_call_ids = [b["id"] for b in second["messages"][1]["content"] if b["type"] == "tool_use"]
    assert [r["tool_use_id"] for r in results] == first_call_ids
    assert results[0] == {"type": "tool_result", "tool_use_id": first_call_ids[0], "content": "5"}
    assert results[1]["is_error"] is True and "ZeroDivisionError" in results[1]["content"]


def test_tool_loop_turn_limit(registry):
    fake = FakeClaude(responder=lambda body: tool_reply([("echo", {"text": "again"})]))
    with pytest.raises(TooManyTurns):
        run_tool_loop(fake.client(), registry, "loop forever", max_turns=3)
    assert len(fake.bodies) == 3


def test_safe_calculate():
    assert safe_calculate("2 + 3 * (4 - 1)") == 11
    assert safe_calculate("-2 ** 2") == -4
    assert safe_calculate("7 % 4 + 10 / 4") == 5.5
    for bad in ["__import__('os').system('echo hacked')", "x + 1", "[1, 2]", "'a' * 3", "2 ** 1000",
                "1" + "+1" * 150, "abs(-3)"]:
        with pytest.raises(ValueError):
            safe_calculate(bad)
    assert not {"eval", "exec", "compile"} & names_used(safe_calculate)
