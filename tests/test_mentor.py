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
