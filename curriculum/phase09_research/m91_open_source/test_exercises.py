import subprocess

import pytest

from exercises import (
    ISSUE_SECTIONS,
    bug_report,
    changelog,
    contribute,
    ddmin,
    git,
    next_version,
    parse_commit,
    review_issue,
)


def is_one_minimal(result, fails):
    return fails(result) and all(not fails(result[:i] + result[i + 1:]) for i in range(len(result)))


def test_ddmin():
    calls = []

    def needs_3_and_7(xs):
        calls.append(1)
        return 3 in xs and 7 in xs

    assert ddmin(list(range(20)), needs_3_and_7) == [3, 7]
    assert len(calls) < 60, "much fewer checks than trying every subset"
    assert ddmin(list("abcdefgh"), lambda s: "c" in s) == ["c"]
    big_sum = lambda s: sum(s) >= 15
    assert is_one_minimal(ddmin([1, 2, 3, 4, 5, 6, 7, 8], big_sum), big_sum)
    rows = [f"row {i}" for i in range(50)]
    crash = lambda rs: "row 17" in rs and "row 42" in rs and "row 3" in rs
    assert sorted(ddmin(rows, crash)) == ["row 17", "row 3", "row 42"]
    with pytest.raises(ValueError):
        ddmin([1, 2, 3], lambda s: False)


def test_parse_commit():
    assert parse_commit("feat(tokenizer): support special tokens") == \
        {"type": "feat", "scope": "tokenizer", "description": "support special tokens", "breaking": False}
    assert parse_commit("fix: handle empty input") == {"type": "fix", "scope": None, "description": "handle empty input", "breaking": False}
    assert parse_commit("refactor!: rename log")["breaking"] is True
    assert parse_commit("feat(api): new client\n\nBREAKING CHANGE: removed old client")["breaking"] is True
    for bad in ["Fixed stuff", "feat:no space", "feat: ", "FEAT: shout", ""]:
        assert parse_commit(bad) is None


def test_next_version():
    assert next_version("1.4.2", ["fix: a", "docs: b"]) == "1.4.3"
    assert next_version("1.4.2", ["fix: a", "feat(x): b"]) == "1.5.0"
    assert next_version("1.4.2", ["feat: a", "perf!: b"]) == "2.0.0"
    assert next_version("1.4.2", ["docs: a", "chore: b", "random words"]) == "1.4.2"
    assert next_version("0.9.9", ["perf: faster"]) == "0.9.10"


def test_changelog():
    msgs = ["feat(rag): hybrid retrieval", "fix: ignore invalid citations", "docs: typo",
            "refactor!: rename Run.log", "perf(api): cache responses", "not conventional"]
    assert changelog("2.0.0", "2026-10-10", msgs) == (
        "## 2.0.0 (2026-10-10)\n\n"
        "### Breaking changes\n\n- rename Run.log\n\n"
        "### Features\n\n- **rag:** hybrid retrieval\n\n"
        "### Bug fixes\n\n- ignore invalid citations\n- **api:** cache responses\n")
    assert changelog("1.0.1", "2026-01-01", ["fix: x"]) == "## 1.0.1 (2026-01-01)\n\n### Bug fixes\n\n- x\n"


def test_bug_report_and_review():
    report = bug_report("split crashes on one class", "It raises ValueError.", ["Install 1.5", "Run the code"],
                        "A split", "ValueError: least populated class...", {"scikit-learn": "1.5.0", "python": "3.12"},
                        code="from sklearn.model_selection import train_test_split\n")
    assert report.startswith("# split crashes on one class\n\n## Description\nIt raises ValueError.\n\n## Steps to reproduce\n1. Install 1.5\n2. Run the code\n")
    assert "```python\nfrom sklearn.model_selection import train_test_split\n```" in report
    assert report.endswith("## Environment\n- scikit-learn: 1.5.0\n- python: 3.12\n")
    assert review_issue(report) == []
    no_code = bug_report("t", "d", ["s"], "e", "a", {})
    assert review_issue(no_code) == ["no code example"]
    assert review_issue("# It doesn't work\nplease fix") == [f"missing section: {s}" for s in ISSUE_SECTIONS] + ["no code example"]


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    for args in (["config", "user.email", "you@example.com"], ["config", "user.name", "You"], ["config", "commit.gpgsign", "false"]):
        git(tmp_path, *args)
    (tmp_path / "README.md").write_text("# project\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "chore: initial commit")
    return tmp_path


def test_contribute(repo):
    with pytest.raises(ValueError):
        contribute(repo, "fix/x", "a.txt", "x", "fixed it")
    assert git(repo, "branch", "--list", "fix/x") == ""
    main_head = git(repo, "rev-parse", "main")
    sha = contribute(repo, "docs/usage", "docs/usage.md", "How to use it.\n", "docs: add usage guide")
    assert sha == git(repo, "rev-parse", "HEAD") and git(repo, "branch", "--show-current") == "docs/usage"
    assert git(repo, "log", "-1", "--format=%s") == "docs: add usage guide"
    assert (repo / "docs" / "usage.md").read_text() == "How to use it.\n"
    assert git(repo, "rev-parse", "main") == main_head, "main is untouched"
    second = contribute(repo, "fix/typo", "README.md", "# Project\n", "fix: capitalize title")
    assert git(repo, "rev-parse", "fix/typo^") == main_head, "each branch starts from main"
    assert second != sha
