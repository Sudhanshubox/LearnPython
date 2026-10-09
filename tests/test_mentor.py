"""Tests for the mentor's offline parts (no API calls)."""

import pytest

from mentor import config, curriculum, progress, prompts


@pytest.fixture(autouse=True)
def temp_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROGRESS_FILE", tmp_path / "progress.json")


def test_new_learner_has_empty_progress():
    data = progress.load()
    assert data["completed"] == []
    assert "not set yet" in progress.summary(data)


def test_passing_attempt_completes_module_once():
    progress.record_attempt("m01", passed=False)
    progress.record_attempt("m01", passed=True)
    data = progress.record_attempt("m01", passed=True)
    assert data["completed"] == ["m01"]
    assert data["attempts"]["m01"] == {"runs": 3, "passed": True}


def test_summary_flags_struggling_modules_and_notes():
    for _ in range(3):
        progress.record_attempt("m01", passed=False)
    progress.add_note("confuses / and //")
    text = progress.summary(progress.load())
    assert "3+ failed test runs: m01" in text
    assert "confuses / and //" in text


def test_curriculum_discovers_modules_in_order():
    ids = [m.id for m in curriculum.all_modules()]
    assert ids[0] == "m01"
    assert ids == sorted(ids)
    assert curriculum.find("m01").lesson.startswith("# m01")
    assert curriculum.next_module([]).id == "m01"


def test_unknown_module_exits_with_message():
    with pytest.raises(SystemExit, match="Unknown module"):
        curriculum.find("m99")


def test_unsolved_exercises_fail_their_tests():
    passed, output = curriculum.run_tests(curriculum.find("m01"))
    assert not passed
    assert "NotImplementedError" in output


def test_prompt_templates_format():
    mod = curriculum.find("m01")
    for template in (prompts.LEARN, prompts.CHECK_FAILED, prompts.CHECK_PASSED, prompts.QUIZ):
        template.format(module_id=mod.id, phase=mod.phase, lesson=mod.lesson,
                        exercises=mod.exercises, code=mod.exercises, output="")


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text

    def model_dump(self, **kwargs):
        return {"type": "text", "text": self.text}


class FakeReply:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [FakeBlock(text)]


def test_chat_sends_lesson_once_and_code_only_when_changed(tmp_path, monkeypatch):
    from mentor import chat, client

    monkeypatch.setattr(config, "CHATS_DIR", tmp_path / "chats")
    sent = []

    def fake_ask(api, system, messages, on_text):
        sent.append(messages[-1]["content"])
        on_text("ok")
        return FakeReply("ok")

    monkeypatch.setattr(client, "ask", fake_ask)
    conversation = chat.Chat("m01")
    conversation.send("hello", lambda t: None, api=object())
    conversation.record_tests(False, "1 failed")
    conversation.send("help", lambda t: None, api=object())

    assert "--- Lesson (README.md) ---" in sent[0]
    assert "My current code" in sent[0] and "exercises.py" in sent[0]
    assert "--- Lesson" not in sent[1]
    assert "My current code" not in sent[1]  # code unchanged since last message
    assert "My latest test run (some failed)" in sent[1]

    reloaded = chat.Chat("m01")
    assert [m["text"] for m in reloaded.display()] == ["hello", "ok", "help", "ok"]
    assert len(reloaded.state["api"]) == 4


def test_failed_send_keeps_history_unchanged(tmp_path, monkeypatch):
    from mentor import chat, client

    monkeypatch.setattr(config, "CHATS_DIR", tmp_path / "chats")

    def failing_ask(api, system, messages, on_text):
        raise client.MentorError("no key")

    monkeypatch.setattr(client, "ask", failing_ask)
    conversation = chat.Chat("m01")
    with pytest.raises(client.MentorError):
        conversation.send("hello", lambda t: None, api=object())
    assert conversation.state["api"] == []


def test_module_files_lists_editable_files_only():
    files = curriculum.find("m30").files
    assert files[0] == "exercises.py"
    assert "textstats/stats.py" in files and "pyproject.toml" in files
    assert not any(f.startswith("test_") or "/test_" in f for f in files)
    assert curriculum.find("m01").files == ["exercises.py"]


def test_module_refuses_to_write_other_files():
    mod = curriculum.find("m01")
    for bad in ["test_exercises.py", "../m02_strings/exercises.py", "README.md", "new.py"]:
        with pytest.raises(KeyError):
            mod.write_file(bad, "x = 1")
