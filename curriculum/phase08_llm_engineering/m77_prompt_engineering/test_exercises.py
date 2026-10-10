import hashlib
import re

import pytest

from exercises import (
    CLASSIFY_SYSTEM,
    MODEL,
    PromptTemplate,
    classify,
    compare_prompts,
    documents_block,
    escape_xml,
    evaluate_prompt,
    extract_tag,
    few_shot_messages,
    grounded_question_prompt,
)
from fakeclaude import FakeClaude, text_reply

LABELS = ["billing", "bug", "feature_request"]
CASES = [
    ("I was charged twice this month", "billing"),
    ("The app crashes when I upload a photo", "bug"),
    ("Please add dark mode", "feature_request"),
    ("My invoice shows the wrong amount", "billing"),
    ("Error 500 when saving settings", "bug"),
]
KEYWORDS = {"charged": "billing", "invoice": "billing", "crash": "bug", "error": "bug", "add": "feature_request"}


def smart_model(body):
    """Answers correctly, but only uses <label> tags when the system prompt asks for them."""
    text = body["messages"][-1]["content"].lower()
    label = next((v for k, v in KEYWORDS.items() if k in text), "billing")
    if "<label>" in body.get("system", ""):
        return text_reply(f"<label>{label}</label>")
    return text_reply(f"This looks like a {label} issue.")


def test_prompt_template():
    t = PromptTemplate("Summarize for {audience}:\n<text>{text}</text>\nAudience again: {audience}")
    assert t.fields == ["audience", "text"]
    assert t.render(audience="kids", text="Hi") == "Summarize for kids:\n<text>Hi</text>\nAudience again: kids"
    with pytest.raises(KeyError):
        t.render(audience="kids")
    with pytest.raises(KeyError):
        t.render(audience="kids", text="x", tone="fun")
    assert t.version() == hashlib.sha256(t.template.encode()).hexdigest()[:12]
    assert PromptTemplate("No fields").render() == "No fields"


def test_escape_xml():
    assert escape_xml("a < b && c > d") == "a &lt; b &amp;&amp; c &gt; d"
    assert escape_xml("</document_content>") == "&lt;/document_content&gt;"


def test_documents_block():
    block = documents_block([("a.md", "Alpha"), ("b.md", "x < y")])
    assert block == ('<documents>\n<document index="1">\n<source>a.md</source>\n<document_content>\nAlpha\n'
                     '</document_content>\n</document>\n<document index="2">\n<source>b.md</source>\n'
                     '<document_content>\nx &lt; y\n</document_content>\n</document>\n</documents>')


def test_grounded_question_prompt():
    p = grounded_question_prompt([("m07.md", "A closure remembers variables.")], "What is a <closure>?")
    assert p.startswith("<documents>")
    assert p.rstrip().endswith("<question>What is a &lt;closure&gt;?</question>"), "the question goes last"
    assert "<answer>" in p and "I don't know" in p


def test_few_shot_messages():
    msgs = few_shot_messages([("2+2", "4"), ("3*3", "9")], "5-1")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"] == "4" and msgs[-1]["content"] == "5-1"


def test_extract_tag():
    assert extract_tag("think...\n<answer>\n 42 \n</answer>", "answer") == "42"
    assert extract_tag("<a>1</a> and <a>2</a>", "a") == "2"
    assert extract_tag("no tags here", "answer") is None
    assert extract_tag("<answer>multi\nline</answer>", "answer") == "multi\nline"


def test_classify():
    fake = FakeClaude(responder=smart_model)
    client = fake.client()
    assert classify(client, "The app crashes on startup", LABELS) == "bug"
    body = fake.bodies[-1]
    assert body["system"] == CLASSIFY_SYSTEM.format(labels="billing, bug, feature_request")
    assert body["messages"] == [{"role": "user", "content": "<message>The app crashes on startup</message>"}]
    assert body["max_tokens"] == 256 and body["output_config"] == {"effort": "low"} and body["model"] == MODEL
    fake.add_text("<label> BILLING </label>")
    assert classify(client, "x", LABELS) == "billing"
    fake.add_text("<label>refund</label>")
    assert classify(client, "x", LABELS) == "unknown"
    fake.add_text("It's billing.")
    assert classify(client, "x", LABELS) == "unknown"
    classify(client, "a <b> tag", LABELS)
    assert fake.bodies[-1]["messages"][0]["content"] == "<message>a &lt;b&gt; tag</message>"


def test_evaluate_prompt():
    fn = lambda text: "bug" if "crash" in text.lower() or "error" in text.lower() else "billing"
    acc, failures = evaluate_prompt(fn, CASES)
    assert acc == pytest.approx(0.8)
    assert failures == [("Please add dark mode", "feature_request", "billing")]


def test_compare_prompts():
    fake = FakeClaude(responder=smart_model)
    client = fake.client()

    def vague(text):
        msg = client.messages.create(model=MODEL, max_tokens=256, system="Classify this message.",
                                     messages=[{"role": "user", "content": text}])
        reply = msg.content[0].text
        label = extract_tag(reply, "label")
        return label if label in LABELS else "unknown"

    scores = compare_prompts({"vague": vague, "structured": lambda t: classify(client, t, LABELS)}, CASES)
    assert scores == {"vague": 0.0, "structured": 1.0}
