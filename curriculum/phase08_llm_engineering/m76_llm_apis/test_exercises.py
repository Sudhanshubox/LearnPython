import random

import anthropic
import pytest

from exercises import (
    FALLBACK_BETA,
    MODEL,
    Conversation,
    LLMError,
    ask,
    ask_with_fallback,
    backoff_delay,
    call_with_retries,
    classify_error,
    cost_usd,
    count_tokens,
    reply_text,
    stream_reply,
)
from fakeclaude import FakeClaude, error_reply, refusal_reply, text_reply, tool_reply


@pytest.fixture
def fake():
    return FakeClaude()


def test_reply_text(fake):
    fake.add_tool_use([("search", {"q": "x"})], text="Let me look.")
    fake.add(text_reply("Hello"))
    client = fake.client()
    m1 = client.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "a"}])
    m2 = client.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "b"}])
    assert reply_text(m1) == "Let me look."
    assert reply_text(m2) == "Hello"


def test_ask(fake):
    fake.add_text("A list comprehension builds a list.")
    assert ask(fake.client(), "What is a list comprehension?") == "A list comprehension builds a list."
    body = fake.bodies[-1]
    assert body["model"] == MODEL and body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": "What is a list comprehension?"}]
    assert body["output_config"] == {"effort": "low"}
    assert "system" not in body
    fake.add_text("ok")
    ask(fake.client(), "q", system="Be brief.", max_tokens=50, effort="high")
    assert fake.bodies[-1]["system"] == "Be brief." and fake.bodies[-1]["output_config"] == {"effort": "high"}


def test_conversation(fake):
    fake.add_text("Nice to meet you, Asha.").add_text("Your name is Asha.")
    conv = Conversation(fake.client(), system="You are friendly.")
    assert conv.send("My name is Asha.") == "Nice to meet you, Asha."
    assert conv.send("What's my name?") == "Your name is Asha."
    second = fake.bodies[1]
    assert second["system"] == "You are friendly."
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["user", "assistant", "user"], "send the whole history every time"
    assert second["messages"][1]["content"] == [{"type": "text", "text": "Nice to meet you, Asha."}], \
        "append the reply's full content blocks"
    assert len(conv.messages) == 4


def test_conversation_failed_send_keeps_history(fake):
    fake.add_text("Hi!").add_error(500)
    conv = Conversation(fake.client())
    conv.send("Hello")
    before = list(conv.messages)
    with pytest.raises(anthropic.APIError):
        conv.send("This one fails")
    assert conv.messages == before, "a failed call must not change the history"


def test_stream_reply(fake):
    fake.add_text("Streaming is great for long replies.")
    chunks = []
    final = stream_reply(fake.client(), [{"role": "user", "content": "hi"}], chunks.append)
    assert len(chunks) > 1 and "".join(chunks) == "Streaming is great for long replies."
    assert final.stop_reason == "end_turn" and reply_text(final) == "".join(chunks)
    assert fake.bodies[-1]["stream"] is True


def test_cost_usd(fake):
    fake.add(text_reply("x", input_tokens=2000, output_tokens=500))
    fake.add(text_reply("x", input_tokens=100, output_tokens=0, cache_read=10_000, cache_write=1000))
    client = fake.client()
    m1 = client.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "a"}])
    m2 = client.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "a"}])
    assert cost_usd(m1.usage) == pytest.approx(0.018)
    assert cost_usd(m2.usage) == pytest.approx((100 * 4 + 10_000 * 0.2 + 1000 * 5) / 1e6)
    assert cost_usd(m1.usage, "claude-haiku-5-5") == pytest.approx((2000 * 0.1 + 500 * 0.5) / 1e6)
    usage = m1.usage.model_copy(update={"cache_read_input_tokens": None, "cache_creation_input_tokens": None})
    assert cost_usd(usage) == pytest.approx(0.018)


def raise_from(fake, reply):
    fake.add(reply)
    try:
        fake.client().messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "a"}])
    except anthropic.APIError as e:
        return e
    raise AssertionError("expected an error")


@pytest.mark.parametrize("status, kind, retryable", [
    (401, "auth", False), (400, "bad_request", False), (429, "rate_limit", True),
    (500, "server", True), (529, "server", True), (404, "client", False),
])
def test_classify_error(fake, status, kind, retryable):
    err = classify_error(raise_from(fake, error_reply(status)))
    assert isinstance(err, LLMError) and err.kind == kind and err.retryable is retryable


def test_classify_connection_error():
    import httpx2
    client = anthropic.Anthropic(api_key="x", max_retries=0, http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(lambda r: (_ for _ in ()).throw(httpx2.ConnectError("down")))))
    with pytest.raises(anthropic.APIConnectionError) as info:
        client.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "a"}])
    err = classify_error(info.value)
    assert err.kind == "connection" and err.retryable
    with pytest.raises(TypeError):
        classify_error(ValueError("not an API error"))


def test_backoff_delay():
    rng = random.Random(0)
    delays = [backoff_delay(a, 1.0, 30.0, rng) for a in range(8)]
    for attempt, d in enumerate(delays):
        cap = min(30.0, 2 ** attempt)
        assert 0.5 * cap <= d <= cap
    r = random.Random(1)
    expected = min(30.0, 1.0 * 2 ** 3) * (0.5 + 0.5 * random.Random(1).random())
    assert backoff_delay(3, rng=r) == pytest.approx(expected)


def test_call_with_retries(fake):
    fake.add_error(529).add_error(500).add_text("finally")
    client = fake.client()
    sleeps = []
    result = call_with_retries(lambda: ask(client, "q"), sleep=sleeps.append, rng=random.Random(0))
    assert result == "finally" and len(sleeps) == 2 and len(fake.bodies) == 3
    assert sleeps[1] > sleeps[0] * 0.9


def test_call_with_retries_respects_retry_after(fake):
    fake.add_error(429, retry_after=7).add_text("ok")
    sleeps = []
    assert call_with_retries(lambda: ask(fake.client(), "q"), sleep=sleeps.append) == "ok"
    assert sleeps[0] >= 7


def test_call_with_retries_gives_up(fake):
    fake.add_error(400)
    sleeps = []
    with pytest.raises(LLMError) as info:
        call_with_retries(lambda: ask(fake.client(), "q"), sleep=sleeps.append)
    assert info.value.kind == "bad_request" and sleeps == [], "never retry a bad request"
    assert isinstance(info.value.__cause__, anthropic.BadRequestError)
    fake.add_error(500).add_error(500).add_error(500)
    with pytest.raises(LLMError) as info:
        call_with_retries(lambda: ask(fake.client(), "q"), max_attempts=3, sleep=sleeps.append)
    assert info.value.kind == "server" and len(sleeps) == 2


def test_ask_with_fallback(fake):
    fake.add_text("Here's the answer.").add(refusal_reply())
    client = fake.client()
    assert ask_with_fallback(client, "Explain buffer overflows for my security class.") == "Here's the answer."
    req = fake.requests[-1]
    assert req["headers"].get("anthropic-beta") == FALLBACK_BETA
    assert req["body"]["fallbacks"] == "default" and req["body"]["output_config"] == {"effort": "medium"}
    assert ask_with_fallback(client, "something declined") is None


def test_count_tokens(fake):
    n = count_tokens(fake.client(), [{"role": "user", "content": "hello " * 50}], system="Be brief.")
    assert isinstance(n, int) and n > 0
    assert fake.requests[-1]["path"].endswith("/count_tokens")
    assert fake.requests[-1]["body"]["system"] == "Be brief."
