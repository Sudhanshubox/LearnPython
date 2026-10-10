import json
import subprocess

import pytest

from exercises import Run, best_run, config_diff, git_info, load_runs, metric_history, runs_table


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1
        return self.t


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    git(code, "init", "-q")
    git(code, "config", "user.email", "you@example.com")
    git(code, "config", "user.name", "You")
    git(code, "config", "commit.gpgsign", "false")
    (code / "train.py").write_text("print('train')\n")
    git(code, "add", ".")
    git(code, "commit", "-q", "-m", "first")
    return code


def test_git_info(repo, tmp_path):
    info = git_info(repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert info == {"commit": head, "dirty": False}
    (repo / "train.py").write_text("print('changed')\n")
    assert git_info(repo)["dirty"] is True
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    assert git_info(plain) is None


def test_run_lifecycle(tmp_path, repo):
    clock = Clock()
    with Run(tmp_path / "runs", "digits", {"lr": 0.01, "seed": 0}, code_dir=repo, clock=clock) as run:
        assert run.dir == tmp_path / "runs" / "digits" / "run-0001"
        meta = json.loads((run.dir / "metadata.json").read_text())
        assert meta["status"] == "running" and meta["start_time"] == 1001.0 and meta["end_time"] is None
        assert meta["git"]["dirty"] is False and len(meta["git"]["commit"]) == 40
        assert {"python", "platform", "argv"} <= set(meta)
        run.log({"loss": 2.0, "acc": 0.5}, step=0)
        run.log({"loss": 1.0}, step=10)
        path = run.save_artifact("preds.csv", "id,pred\n1,7\n")
        run.save_artifact("model.bin", b"\x00\x01")
    assert path == run.dir / "artifacts" / "preds.csv" and path.read_text() == "id,pred\n1,7\n"
    assert (run.dir / "artifacts" / "model.bin").read_bytes() == b"\x00\x01"
    assert json.loads((run.dir / "config.json").read_text()) == {"lr": 0.01, "seed": 0}
    meta = json.loads((run.dir / "metadata.json").read_text())
    assert meta["status"] == "finished" and meta["end_time"] == 1002.0
    assert json.loads((run.dir / "summary.json").read_text()) == {"loss": 1.0, "acc": 0.5}
    lines = (run.dir / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[0]) == {"step": 0, "name": "loss", "value": 2.0} and len(lines) == 3
    assert metric_history(run.dir, "loss") == [(0, 2.0), (10, 1.0)]
    assert metric_history(run.dir, "nope") == []


def test_failed_run_and_names(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with Run(tmp_path, "p", {"x": 1}, code_dir=tmp_path) as run:
            run.log({"loss": 3.0}, step=0)
            1 / 0
    assert json.loads((run.dir / "metadata.json").read_text())["status"] == "failed"
    assert json.loads((run.dir / "metadata.json").read_text())["git"] is None
    second = Run(tmp_path, "p", {"x": 2}, code_dir=tmp_path)
    assert second.dir.name == "run-0002"
    named = Run(tmp_path, "p", {"x": 3}, name="baseline", code_dir=tmp_path)
    assert named.dir.name == "baseline"
    with pytest.raises(FileExistsError):
        Run(tmp_path, "p", {"x": 4}, name="baseline", code_dir=tmp_path)
    assert metric_history(second.dir, "loss") == []


@pytest.fixture
def project(tmp_path):
    results = [({"lr": 0.1, "scale": True}, 0.91, "finished"), ({"lr": 0.01, "scale": True}, 0.97, "finished"),
               ({"lr": 0.001, "scale": False}, 0.99, "failed"), ({"lr": 0.01, "scale": False}, 0.95, "finished")]
    for config, acc, status in results:
        run = Run(tmp_path, "digits", config, code_dir=tmp_path)
        run.log({"val_acc": acc}, step=1)
        run.finish(status)
    Run(tmp_path, "digits", {"lr": 1.0}, code_dir=tmp_path)          # still running, no metrics
    return load_runs(tmp_path, "digits")


def test_load_runs(project):
    assert [r["name"] for r in project] == ["run-0001", "run-0002", "run-0003", "run-0004", "run-0005"]
    assert project[1]["config"] == {"lr": 0.01, "scale": True} and project[1]["summary"] == {"val_acc": 0.97}
    assert [r["status"] for r in project] == ["finished", "finished", "failed", "finished", "running"]
    assert project[4]["summary"] == {}


def test_best_run(project):
    assert best_run(project, "val_acc")["name"] == "run-0002", "failed runs don't count"
    assert best_run(project, "val_acc", mode="min")["name"] == "run-0001"
    assert best_run(project, "loss") is None


def test_config_diff():
    assert config_diff({"lr": 0.1, "seed": 0, "a": 1}, {"lr": 0.01, "seed": 0, "b": 2}) == \
        {"a": (1, None), "b": (None, 2), "lr": (0.1, 0.01)}
    assert config_diff({"x": 1}, {"x": 1}) == {}


def test_runs_table(project):
    table = runs_table(project, "val_acc", ["lr", "scale"])
    lines = table.splitlines()
    assert lines[0] == "| run | lr | scale | val_acc | status |"
    assert lines[1] == "|---|---|---|---|---|"
    assert lines[2] == "| run-0003 | 0.001 | False | 0.9900 | failed |"
    assert lines[3] == "| run-0002 | 0.01 | True | 0.9700 | finished |"
    assert len(lines) == 6
    assert runs_table(project, "val_acc", ["lr"], mode="min").splitlines()[2].startswith("| run-0001 |")
