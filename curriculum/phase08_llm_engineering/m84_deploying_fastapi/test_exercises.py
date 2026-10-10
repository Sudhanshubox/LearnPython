import re
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)   # a harmless warning inside the test client library
    from fastapi.testclient import TestClient

from exercises import VERSION, AskRequest, AskResponse, Settings, create_app, dockerfile, load_settings


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def answer_fn(question, max_sources):
    if question == "explode":
        raise RuntimeError("secret internal detail: db password is hunter2")
    return f"Answer to: {question}", ["m01", "m02", "m03", "m04", "m05"]


def stream_fn(question):
    yield "Hello"
    yield " world"


@pytest.fixture
def setup():
    clock = Clock()
    settings = Settings(api_keys=frozenset({"key-a", "key-b"}), requests_per_minute=3)
    app = create_app(settings, answer_fn, stream_fn, clock=clock)
    return app, TestClient(app), clock


KEY = {"X-API-Key": "key-a"}


def test_load_settings():
    s = load_settings({"APP_API_KEYS": " a, b ,,c ", "RATE_PER_MINUTE": "10", "MODEL": "claude-sonnet-5-5"})
    assert s == Settings(api_keys=frozenset({"a", "b", "c"}), requests_per_minute=10, model="claude-sonnet-5-5")
    assert load_settings({"APP_API_KEYS": "x"}) == Settings(api_keys=frozenset({"x"}))
    for bad in [{}, {"APP_API_KEYS": " , "}, {"APP_API_KEYS": "x", "RATE_PER_MINUTE": "0"}]:
        with pytest.raises(ValueError):
            load_settings(bad)


def test_models():
    assert AskRequest(question="hi").max_sources == 3
    for bad in [{"question": ""}, {"question": "x" * 2001}, {"question": "q", "max_sources": 0},
                {"question": "q", "max_sources": 11}]:
        with pytest.raises(Exception):
            AskRequest(**bad)
    assert AskResponse(answer="a", sources=["m01"], request_id="r").sources == ["m01"]


def test_health(setup):
    app, client, _ = setup
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "version": VERSION}
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])


def test_ask(setup):
    app, client, _ = setup
    r = client.post("/ask", json={"question": "What is a list?", "max_sources": 2}, headers={**KEY, "X-Request-ID": "abc123"})
    assert r.status_code == 200
    assert r.json() == {"answer": "Answer to: What is a list?", "sources": ["m01", "m02"], "request_id": "abc123"}
    assert r.headers["x-request-id"] == "abc123"
    log = app.state.logs[-1]
    assert {k: log[k] for k in ("request_id", "method", "path", "status")} == \
        {"request_id": "abc123", "method": "POST", "path": "/ask", "status": 200}
    assert isinstance(log["latency_ms"], float)


def test_validation_and_auth(setup):
    app, client, _ = setup
    assert client.post("/ask", json={"question": ""}, headers=KEY).status_code == 422
    assert client.post("/ask", json={"question": "x" * 3000}, headers=KEY).status_code == 422
    assert client.post("/ask", json={"question": "hi"}).status_code == 401
    assert client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert app.state.logs[-1]["status"] == 401


def test_rate_limit_per_key(setup):
    app, client, clock = setup
    codes = [client.post("/ask", json={"question": "q"}, headers=KEY).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = client.post("/ask", json={"question": "q"}, headers=KEY)
    assert r.status_code == 429 and r.headers["retry-after"] == "20"
    assert client.post("/ask", json={"question": "q"}, headers={"X-API-Key": "key-b"}).status_code == 200, \
        "each key has its own bucket"
    clock.t = 20.0
    assert client.post("/ask", json={"question": "q"}, headers=KEY).status_code == 200


def test_errors_are_safe(setup):
    app, client, _ = setup
    r = client.post("/ask", json={"question": "explode"}, headers=KEY)
    assert r.status_code == 503
    body = r.json()
    assert body["detail"] == "the assistant is temporarily unavailable"
    assert body["request_id"] == r.headers["x-request-id"]
    assert "hunter2" not in r.text and "RuntimeError" not in r.text


def test_stream(setup):
    app, client, _ = setup
    r = client.post("/ask/stream", json={"question": "hi"}, headers=KEY)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text == "data: Hello\n\ndata:  world\n\nevent: done\ndata: \n\n"
    assert client.post("/ask/stream", json={"question": "hi"}).status_code == 401
    no_stream = TestClient(create_app(Settings(api_keys=frozenset({"k"})), answer_fn))
    assert no_stream.post("/ask/stream", json={"question": "hi"}, headers={"X-API-Key": "k"}).status_code == 404


def test_dockerfile():
    text = dockerfile()
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert lines[0].startswith("FROM python:3.12-slim")
    req_copy = next(i for i, l in enumerate(lines) if l.startswith("COPY") and "requirements.txt" in l)
    install = next(i for i, l in enumerate(lines) if l.startswith("RUN") and "pip install" in l)
    code_copy = next(i for i, l in enumerate(lines) if l.startswith("COPY . "))
    assert req_copy < install < code_copy, "install dependencies before copying the code"
    assert "--no-cache-dir" in lines[install]
    user = [l for l in lines if l.startswith("USER")]
    assert user and user[-1] != "USER root", "don't run as root"
    assert any(l.startswith("EXPOSE") and "8000" in l for l in lines)
    assert "uvicorn" in lines[-1] and "app:app" in lines[-1] and "0.0.0.0" in lines[-1]
    assert "sk-ant" not in text and "ANTHROPIC_API_KEY" not in text, "no secrets in the image"
