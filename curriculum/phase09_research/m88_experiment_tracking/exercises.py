"""m88 exercises: a small experiment tracker."""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path


# 1. The code version at `path`: run `git rev-parse HEAD` and `git status --porcelain` there
#    (subprocess.run with capture_output=True, text=True, check=True). Return
#    {"commit": the hash, "dirty": True if status printed anything}, or None if git fails or
#    isn't installed (CalledProcessError or FileNotFoundError).
def git_info(path="."):
    raise NotImplementedError


# 2. A tracked run (README sections 1-2).
#    __init__: the run folder is ROOT/PROJECT/NAME; NAME defaults to "run-NNNN" where NNNN is
#      (number of existing run folders in the project) + 1, zero-padded to 4 digits. Create the
#      project folder if needed; raise FileExistsError if the run folder already exists.
#      Write config.json (a copy of config) and metadata.json with {"name", "status": "running",
#      "start_time": clock(), "end_time": None, "python": platform.python_version(),
#      "platform": platform.platform(), "argv": sys.argv, "git": git_info(code_dir)}.
#      Keep self.dir, self.config, self.metadata and self.summary = {}. Write JSON files with
#      json.dumps(data, indent=2, sort_keys=True).
#    log(metrics, step): append one line {"step", "name", "value"} per metric to metrics.jsonl,
#      and remember the latest value of each metric in self.summary.
#    save_artifact(filename, content): write text or bytes to artifacts/FILENAME; return the path.
#    finish(status): set the status and "end_time" (clock()) in metadata.json; write summary.json.
#    As a context manager, it returns itself and finishes with "failed" if an exception escaped
#    (without swallowing it), else "finished".
class Run:
    def __init__(self, root, project, config, name=None, code_dir=".", clock=time.time):
        raise NotImplementedError

    def log(self, metrics, step):
        raise NotImplementedError

    def save_artifact(self, filename, content):
        raise NotImplementedError

    def finish(self, status="finished"):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb):
        raise NotImplementedError


# 3. [(step, value), ...] for one metric of a run folder, in logged order ([] if no metrics file).
def metric_history(run_dir, name):
    raise NotImplementedError


# 4. Every run folder of a project, sorted by folder name, as dicts {"name", "dir" (a Path),
#    "config", "summary" ({} if summary.json is missing), "status" (from metadata.json, or
#    "unknown"), "git"}.
def load_runs(root, project):
    raise NotImplementedError


# 5. The finished run with the highest (mode="max") or lowest (mode="min") summary value of
#    metric, ignoring runs that didn't finish or lack the metric; None if there is none.
def best_run(runs, metric, mode="max"):
    raise NotImplementedError


# 6. {key: (value in a, value in b)} for every key whose value differs (a missing key counts as
#    None), with keys in sorted order.
def config_diff(a, b):
    raise NotImplementedError


# 7. A Markdown table of the runs that have the metric, best first:
#      | run | KEY1 | KEY2 | METRIC | status |
#      |---|---|---|---|---|
#      | run-0002 | 0.01 | True | 0.9712 | finished |
#    (config values with str(), "" if missing; the metric with 4 decimals).
def runs_table(runs, metric, config_keys, mode="max"):
    raise NotImplementedError
