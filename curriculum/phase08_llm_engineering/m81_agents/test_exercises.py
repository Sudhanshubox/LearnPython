import threading
from pathlib import Path

import pytest

from exercises import (
    MODEL,
    Agent,
    AgentResult,
    complete,
    evaluator_optimizer,
    parallel_vote,
    prompt_chain,
    route,
    safe_path,
    workspace_tools,
)
from fakeclaude import FakeClaude, text_reply, tool_reply


def last_prompt(body):
    return body["messages"][-1]["content"]


def test_complete():
    fake = FakeClaude().add_text("hi")
    assert complete(fake.client(), "hello", system="Be kind.", effort="high") == "hi"
    body = fake.bodies[-1]
    assert body["model"] == MODEL and body["system"] == "Be kind." and body["output_config"] == {"effort": "high"}
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_prompt_chain():
    fake = FakeClaude(responder=lambda body: text_reply(f"<{last_prompt(body)}>"))
    out = prompt_chain(fake.client(), ["outline: {input}", "draft: {input}"], "topic")
    assert out == "<draft: <outline: topic>>"
    assert len(fake.bodies) == 2


def test_route():
    fake = FakeClaude()
    handlers = {"billing": lambda t: "B:" + t, "bug": lambda t: "G:" + t, "other": lambda t: "O:" + t}
    fake.add_text("  Bug \n")
    assert route(fake.client(), "it crashes", handlers) == "G:it crashes"
    assert "billing, bug, other" in last_prompt(fake.bodies[-1])
    fake.add_text("weather")
    assert route(fake.client(), "hmm", handlers) == "O:hmm"
    fake.add_text("weather")
    with pytest.raises(ValueError):
        route(fake.client(), "hmm", {"billing": lambda t: t})


def test_parallel_vote():
    answers = iter(["42", "41", "42 ", "42", "41"])
    lock = threading.Lock()

    def responder(body):
        with lock:
            return text_reply(next(answers))

    fake = FakeClaude(responder=responder)
    answer, agreement = parallel_vote(fake.client(), "What is 6*7?", n=5)
    assert answer == "42" and agreement == pytest.approx(0.6)
    assert len(fake.bodies) == 5


def test_evaluator_optimizer():
    fake = FakeClaude()
    fake.add_text("draft v1").add_text("Missing an example.").add_text("draft v2").add_text("PASS")
    answer, rounds = evaluator_optimizer(fake.client(), "Explain recursion")
    assert (answer, rounds) == ("draft v2", 2)
    assert "Missing an example." in last_prompt(fake.bodies[2]), "the rewrite must see the feedback"
    assert "draft v1" in last_prompt(fake.bodies[1])
    fake2 = FakeClaude(responder=lambda b: text_reply("still wrong"))
    assert evaluator_optimizer(fake2.client(), "x", max_rounds=2) == ("still wrong", 2)
    assert len(fake2.bodies) == 5


def test_safe_path(tmp_path):
    (tmp_path / "notes").mkdir()
    assert safe_path(tmp_path, "notes/a.txt") == (tmp_path / "notes" / "a.txt").resolve()
    assert safe_path(tmp_path, "notes/../b.txt") == (tmp_path / "b.txt").resolve()
    for bad in ["../secret.txt", "notes/../../x", "/etc/passwd"]:
        with pytest.raises(PermissionError):
            safe_path(tmp_path, bad)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "todo.txt").write_text("1. learn agents\n2. write tests\n", encoding="utf-8")
    return tmp_path


def scripted_agent_fake():
    fake = FakeClaude()
    fake.add(tool_reply([("list_files", {})], text="Let me look around."))
    fake.add(tool_reply([("read_file", {"path": "todo.txt"}), ("read_file", {"path": "../escape.txt"})]))
    fake.add(tool_reply([("write_file", {"path": "summary.txt", "content": "2 tasks"})]))
    fake.add(text_reply("The todo list has 2 tasks.", input_tokens=100, output_tokens=20))
    return fake


def test_agent_with_approval(workspace):
    fake = scripted_agent_fake()
    approvals = []
    agent = Agent(fake.client(), workspace_tools(workspace), system="You manage files.",
                  needs_approval={"write_file"}, approve=lambda name, inp: approvals.append((name, inp)) or True)
    result = agent.run("Summarize my todo list into summary.txt")
    assert isinstance(result, AgentResult)
    assert (result.text, result.stop, result.steps) == ("The todo list has 2 tasks.", "end_turn", 4)
    assert result.tokens == 3 * 15 + 120
    assert approvals == [("write_file", {"path": "summary.txt", "content": "2 tasks"})]
    assert (workspace / "summary.txt").read_text() == "2 tasks"
    tools = [e for e in result.trajectory if e["type"] == "tool"]
    assert [e["tool"] for e in tools] == ["list_files", "read_file", "read_file", "write_file"]
    assert tools[0]["output"] == "todo.txt" and tools[1]["output"].startswith("1. learn agents")
    assert tools[2]["is_error"] and "PermissionError" in tools[2]["output"], "path traversal is blocked"
    assert result.trajectory[-1] == {"type": "final", "text": "The todo list has 2 tasks."}
    second = fake.bodies[2]["messages"][-1]["content"]
    assert len(second) == 2 and second[1]["is_error"] is True and "is_error" not in second[0]
    assert fake.bodies[0]["system"] == "You manage files." and fake.bodies[0]["output_config"] == {"effort": "medium"}


def test_agent_declined_action(workspace):
    fake = scripted_agent_fake()
    agent = Agent(fake.client(), workspace_tools(workspace), needs_approval={"write_file"})
    result = agent.run("Summarize my todo list into summary.txt")
    assert not (workspace / "summary.txt").exists(), "declined actions must not run"
    approval = [e for e in result.trajectory if e["type"] == "approval"]
    assert approval == [{"type": "approval", "tool": "write_file", "approved": False}]
    declined = fake.bodies[3]["messages"][-1]["content"][0]
    assert declined["content"] == "The user declined this action." and declined["is_error"] is True


def test_agent_budgets(workspace):
    looping = FakeClaude(responder=lambda b: tool_reply([("list_files", {})]))
    result = Agent(looping.client(), workspace_tools(workspace), max_steps=3).run("loop")
    assert (result.stop, result.steps, result.text) == ("max_steps", 3, "")
    assert len(looping.bodies) == 3
    pricey = FakeClaude(responder=lambda b: tool_reply([("list_files", {})], input_tokens=600, output_tokens=0))
    result = Agent(pricey.client(), workspace_tools(workspace), max_tokens_total=1000).run("loop")
    assert (result.stop, result.steps, result.tokens) == ("token_budget", 2, 1200)
    unknown = FakeClaude().add(tool_reply([("delete_everything", {})])).add_text("ok")
    result = Agent(unknown.client(), workspace_tools(workspace)).run("x")
    assert result.trajectory[0]["output"] == "Error: unknown tool 'delete_everything'"
