"""Finds modules in curriculum/ and runs their tests.

Layout: curriculum/phaseNN_<name>/mNN_<name>/ containing
  README.md          the lesson
  exercises.py       stubs the learner fills in
  test_exercises.py  pytest checks for the exercises
Some modules (packages, projects) have more .py files for the learner to edit.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config

TEST_TIMEOUT = 120


@dataclass
class Module:
    id: str
    phase: str
    path: Path

    @property
    def lesson(self):
        readme = self.path / "README.md"
        return readme.read_text(encoding="utf-8") if readme.exists() else ""

    @property
    def title(self):
        first = self.lesson.split("\n", 1)[0].lstrip("# ").strip()
        return first.split("·", 1)[-1].strip() or self.path.name

    @property
    def exercises(self):
        file = self.path / "exercises.py"
        return file.read_text(encoding="utf-8") if file.exists() else ""

    @property
    def files(self):
        """Python files the learner edits (not tests), relative to the module, exercises.py first."""
        found = sorted(
            p.relative_to(self.path).as_posix()
            for p in self.path.rglob("*.py")
            if not p.name.startswith("test_")
            and p.name != "conftest.py"
            and not any(part.startswith((".", "__pycache__")) for part in p.relative_to(self.path).parts)
        )
        return sorted(found, key=lambda f: f != "exercises.py")

    def read_file(self, rel):
        if rel not in self.files:
            raise KeyError(rel)
        return (self.path / rel).read_text(encoding="utf-8")

    def write_file(self, rel, code):
        if rel not in self.files:
            raise KeyError(rel)
        (self.path / rel).write_text(code, encoding="utf-8")

    def code_snapshot(self):
        """All of the learner's files as one markdown string, for the mentor."""
        return "\n\n".join(f"{rel}:\n```python\n{self.read_file(rel)}\n```" for rel in self.files)


def all_modules():
    modules = []
    for phase in sorted(p for p in config.CURRICULUM_DIR.glob("phase*") if p.is_dir()):
        for mod in sorted(m for m in phase.glob("m*") if m.is_dir()):
            modules.append(Module(id=mod.name.split("_")[0], phase=phase.name, path=mod))
    return modules


def get(module_id):
    """Look up a module by id ("m01") or full folder name ("m01_hello_python"); None if unknown."""
    for mod in all_modules():
        if module_id in (mod.id, mod.path.name):
            return mod
    return None


def find(module_id):
    """Like get(), but exits with a helpful message for unknown modules (for the CLI)."""
    mod = get(module_id)
    if mod is None:
        known = ", ".join(m.id for m in all_modules())
        raise SystemExit(f"Unknown module {module_id!r}. Available: {known}")
    return mod


def next_module(completed):
    for mod in all_modules():
        if mod.id not in completed:
            return mod
    return None


def run_tests(mod):
    """Run the module's tests. Returns (passed, output)."""
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest", str(mod.path), "-q", "--no-header", "--tb=short",
                # Stop any single test after 5s so an infinite loop shows up as a clear failure.
                "--timeout=5", "-p", "no:cacheprovider",
            ],
            capture_output=True,
            text=True,
            cwd=mod.path,
            timeout=TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"Tests were stopped after {TEST_TIMEOUT} seconds. Is there an infinite loop?"
    return result.returncode == 0, result.stdout + result.stderr
