import shutil
import subprocess

import pytest

from exercises import (
    changed_files,
    commit_all,
    commit_on_branch,
    file_at,
    first_commit_with,
    history,
    init_repo,
    is_clean,
    run_git,
    try_merge,
    undo_last_commit,
    write_file,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "project"
    init_repo(path, "Asha Learner", "asha@example.com")
    git(path, "config", "commit.gpgsign", "false")   # don't depend on your global signing setup
    return path


@pytest.fixture
def repo_with_commit(repo):
    write_file(repo, "train.py", "lr = 0.001\n")
    commit_all(repo, "Add training script")
    return repo


def test_init_repo(repo):
    assert (repo / ".git").is_dir()
    assert git(repo, "config", "--local", "user.name") == "Asha Learner"
    assert git(repo, "config", "--local", "user.email") == "asha@example.com"
    assert git(repo, "symbolic-ref", "--short", "HEAD") == "main"


def test_run_git(repo):
    assert run_git(repo, "status", "--porcelain") == ""
    with pytest.raises(RuntimeError, match="not-a-real-command|is not a git command"):
        run_git(repo, "not-a-real-command")


def test_write_file_creates_folders(repo):
    write_file(repo, "src/model/layers.py", "x = 1\n")
    assert (repo / "src" / "model" / "layers.py").read_text(encoding="utf-8") == "x = 1\n"


def test_commit_all_and_history(repo):
    write_file(repo, "a.txt", "1")
    h1 = commit_all(repo, "First commit")
    write_file(repo, "a.txt", "2")
    write_file(repo, "b.txt", "new")
    h2 = commit_all(repo, "Second commit")
    assert len(h1) == 40 and h1 != h2
    assert h2 == git(repo, "rev-parse", "HEAD")
    assert history(repo) == ["Second commit", "First commit"]


def test_commit_all_includes_deletions(repo_with_commit):
    (repo_with_commit / "train.py").unlink()
    commit_all(repo_with_commit, "Remove script")
    assert git(repo_with_commit, "ls-files") == ""


def test_is_clean_and_changed_files(repo_with_commit):
    assert is_clean(repo_with_commit)
    assert changed_files(repo_with_commit) == []
    write_file(repo_with_commit, "train.py", "lr = 0.01\n")
    write_file(repo_with_commit, "notes.md", "hi")
    assert not is_clean(repo_with_commit)
    assert changed_files(repo_with_commit) == ["notes.md", "train.py"]


def test_commit_on_branch_returns_to_start(repo_with_commit):
    commit_on_branch(repo_with_commit, "experiment", "aug.py", "flip = True\n", "Add augmentation")
    assert git(repo_with_commit, "branch", "--show-current") == "main"
    assert not (repo_with_commit / "aug.py").exists()
    assert "Add augmentation" in git(repo_with_commit, "log", "experiment", "--format=%s")


def test_clean_merge(repo_with_commit):
    commit_on_branch(repo_with_commit, "feature", "aug.py", "flip = True\n", "Add augmentation")
    assert try_merge(repo_with_commit, "feature") is True
    assert (repo_with_commit / "aug.py").exists()
    assert history(repo_with_commit)[0] == "Add augmentation"


def test_conflicting_merge_is_aborted(repo_with_commit):
    commit_on_branch(repo_with_commit, "experiment", "train.py", "lr = 0.1\n", "Try a bigger lr")
    write_file(repo_with_commit, "train.py", "lr = 0.0001\n")
    commit_all(repo_with_commit, "Try a smaller lr")
    assert try_merge(repo_with_commit, "experiment") is False
    assert is_clean(repo_with_commit), "abort the merge so the repo is clean again"
    assert (repo_with_commit / "train.py").read_text(encoding="utf-8") == "lr = 0.0001\n"


def test_file_at(repo_with_commit):
    first = git(repo_with_commit, "rev-parse", "HEAD")
    write_file(repo_with_commit, "train.py", "lr = 0.5\n")
    commit_all(repo_with_commit, "Change lr")
    assert file_at(repo_with_commit, first, "train.py") == "lr = 0.001"
    assert file_at(repo_with_commit, "HEAD", "train.py") == "lr = 0.5"


def test_undo_last_commit_keeps_changes(repo_with_commit):
    write_file(repo_with_commit, "oops.py", "secret = 1\n")
    commit_all(repo_with_commit, "Oops")
    undo_last_commit(repo_with_commit)
    assert history(repo_with_commit) == ["Add training script"]
    assert (repo_with_commit / "oops.py").exists()
    assert "A  oops.py" in git(repo_with_commit, "status", "--porcelain"), "changes should stay staged"


def test_first_commit_with(repo):
    write_file(repo, "cfg.py", "lr = 1\n")
    commit_all(repo, "Initial config")
    write_file(repo, "cfg.py", "lr = 1\ndropout = 0.1\n")
    commit_all(repo, "Add dropout")
    write_file(repo, "cfg.py", "lr = 2\ndropout = 0.1\n")
    commit_all(repo, "Tune lr")
    assert first_commit_with(repo, "dropout") == "Add dropout"
    assert first_commit_with(repo, "lr = 2") == "Tune lr"
    assert first_commit_with(repo, "momentum") is None
