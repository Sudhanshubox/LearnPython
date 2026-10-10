import re
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)   # a harmless warning inside the test client library
    from fastapi.testclient import TestClient

from exercises import (
    FALLBACK_BETA,
    MODEL,
    REFUSAL_MESSAGE,
    SYSTEM,
    Answer,
    AssistantUnavailable,
    CourseAssistant,
    HybridRetriever,
    build_prompt,
    create_app,
    evaluate,
    judge,
    load_course_chunks,
    load_eval_set,
    message_cost,
    write_report,
)
from fakeclaude import FakeClaude, refusal_reply, text_reply


def course_model(body):
    """A fake assistant: answers with the first sentence of document 1 and cites it."""
    prompt = body["messages"][0]["content"]
    first_doc = re.search(r"<document_content>\n(.*?)\n</document_content>", prompt, re.S)
    if first_doc is None:
        return text_reply("I don't know based on the course material.", input_tokens=500, output_tokens=40)
    sentence = first_doc.group(1).split(". ")[0][:200]
    return text_reply(f"{sentence}. [1] (see also [1] and [42])", input_tokens=2000, output_tokens=100)


def fair_judge(body):
    prompt = body["messages"][0]["content"]
    answer = re.search(r"<answer_to_grade>(.*?)</answer_to_grade>", prompt, re.S).group(1)
    return text_reply("<score>4</score>" if "[1]" in answer else "<score>1</score>")


@pytest.fixture(scope="module")
def chunks():
    return load_course_chunks()


@pytest.fixture(scope="module")
def retriever(chunks):
    return HybridRetriever(chunks)


def test_hybrid_retriever(chunks, retriever):
    results = retriever.search("What does the KV cache store?", k=5)
    assert len(results) == 5 and len(set(results)) == 5
    assert "m71" in [chunks[i]["source"] for i in results]
    hits = sum(item["expected_module"] in [chunks[i]["source"] for i in retriever.search(item["question"], 5)]
               for item in load_eval_set())
    assert hits / 20 >= 0.85


def test_build_prompt():
    p = build_prompt("Why <x>?", [{"source": "m68", "heading": "Scaling", "text": "Divide by √d & more."}])
    assert p.startswith("<documents>") and '<document index="1">' in p
    assert "<source>m68: Scaling</source>" in p and "Divide by √d &amp; more." in p
    assert "[1]" in p and "I don't know based on the course material." in p
    assert p.rstrip().endswith("<question>Why &lt;x&gt;?</question>")


def test_message_cost():
    fake = FakeClaude().add(text_reply("x", input_tokens=2000, output_tokens=100, cache_read=10_000))
    m = fake.client().messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "a"}])
    assert message_cost(m.usage) == pytest.approx((2000 * 4 + 100 * 20 + 10_000 * 0.2) / 1e6)


def test_ask(chunks, retriever):
    fake = FakeClaude(responder=course_model)
    assistant = CourseAssistant(fake.client(), chunks, retriever)
    answer = assistant.ask("What does the KV cache store during generation?")
    assert isinstance(answer, Answer) and not answer.cached and not answer.refused
    retrieved = assistant.retrieve("What does the KV cache store during generation?")
    assert answer.sources == [retrieved[0]["source"]], "valid, de-duplicated citations only"
    assert answer.cost_usd == pytest.approx((2000 * 4 + 100 * 20) / 1e6)
    req = fake.requests[-1]
    assert req["headers"].get("anthropic-beta") == FALLBACK_BETA and req["body"]["fallbacks"] == "default"
    assert req["body"]["system"] == SYSTEM and req["body"]["output_config"] == {"effort": "low"}
    again = assistant.ask("  what does the KV cache   store during generation? ")
    assert again.cached and again.cost_usd == 0.0 and again.text == answer.text
    assert len(fake.bodies) == 1, "a cached question doesn't call the API"
    assert assistant.total_cost == pytest.approx(answer.cost_usd)


def test_retrieve_budget(chunks, retriever):
    assistant = CourseAssistant(FakeClaude().client(), chunks, retriever, k=5, max_context_chars=1500)
    kept = assistant.retrieve("What does the KV cache store?")
    assert sum(len(c["text"]) for c in kept) <= 1500
    all5 = [chunks[i] for i in retriever.search("What does the KV cache store?", 5)]
    assert kept == all5[:len(kept)]


def test_refusal_and_errors(chunks, retriever):
    fake = FakeClaude().add(refusal_reply()).add_error(529).add_text("ok [1]")
    assistant = CourseAssistant(fake.client(), chunks, retriever)
    refused = assistant.ask("some question")
    assert refused.refused and refused.text == REFUSAL_MESSAGE and refused.sources == []
    with pytest.raises(AssistantUnavailable) as info:
        assistant.ask("another question")
    assert info.value.__cause__ is not None
    assert not assistant.ask("some question").cached, "refusals are not cached"


def test_judge():
    fake = FakeClaude().add_text("Solid. <score>5</score>").add_text("no score here").add_text("<score>7</score>")
    client = fake.client()
    assert judge(client, "Q", "R", "A") == 5
    assert "<reference_answer>R</reference_answer>" in fake.bodies[0]["messages"][0]["content"]
    assert fake.bodies[0]["output_config"] == {"effort": "medium"}
    assert judge(client, "Q", "R", "A") is None and judge(client, "Q", "R", "A") is None


@pytest.fixture(scope="module")
def results(chunks, retriever):
    assistant = CourseAssistant(FakeClaude(responder=course_model).client(), chunks, retriever)
    return evaluate(assistant, FakeClaude(responder=fair_judge).client(), load_eval_set())


@pytest.mark.timeout(60)
def test_evaluate(results):
    assert results["n"] == 20
    assert results["retrieval_hit_rate"] >= 0.85
    assert 0.3 <= results["citation_hit_rate"] <= results["retrieval_hit_rate"]
    mean, low, high = results["judge_score"]
    assert mean == pytest.approx(4.0) and low <= mean <= high
    assert results["judge_failures"] == 0 and results["refusal_rate"] == 0.0
    assert results["cost_per_question"] == pytest.approx((2000 * 4 + 100 * 20) / 1e6)
    assert results["failures"] == []


def test_app(chunks, retriever):
    fake = FakeClaude(responder=course_model)
    client = TestClient(create_app(CourseAssistant(fake.client(), chunks, retriever), api_keys={"k"}))
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/ask", json={"question": "What is a closure?"}, headers={"X-API-Key": "k"})
    assert r.status_code == 200 and set(r.json()) == {"answer", "sources", "cached"}
    assert client.post("/ask", json={"question": "What is a closure?"}, headers={"X-API-Key": "k"}).json()["cached"]
    assert client.post("/ask", json={"question": "x"}).status_code == 401
    assert client.post("/ask", json={"question": ""}, headers={"X-API-Key": "k"}).status_code == 422
    fake.responder = None
    fake.add_error(500)
    assert client.post("/ask", json={"question": "new one"}, headers={"X-API-Key": "k"}).status_code == 503


def test_report(tmp_path, results):
    path = tmp_path / "report.md"
    failing = dict(results, failures=[{"question": "What is LoRA?", "answer": "?", "score": 1}])
    write_report(failing, path)
    text = path.read_text(encoding="utf-8")
    for heading in ["## System", "## Results", "## Failure analysis", "## Next steps"]:
        assert heading in text
    assert f"{results['retrieval_hit_rate']:.2f}" in text and f"{results['judge_score'][0]:.2f}" in text
    assert "What is LoRA?" in text and "|" in text
    assert len(text.split()) > 80
