"""Finds modules in curriculum/ and runs their tests.

Layout: curriculum/phaseNN_<name>/mNN_<name>/ containing
  README.md          the lesson
  exercises.py       stubs the learner fills in
  test_exercises.py  pytest checks for the exercises
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Module:
    id: str
    phase: str
    path: Path

    @property
    def lesson(self):
        readme = self.path / "README.md"
        return readme.read_text() if readme.exists() else ""

    @property
    def exercises(self):
        file = self.path / "exercises.py"
        return file.read_text() if file.exists() else ""


def all_modules():
    modules = []
    for phase in sorted(p for p in config.CURRICULUM_DIR.glob("phase*") if p.is_dir()):
        for mod in sorted(m for m in phase.glob("m*") if m.is_dir()):
            modules.append(Module(id=mod.name.split("_")[0], phase=phase.name, path=mod))
    return modules


def find(module_id):
    """Look up a module by id ("m01") or full folder name ("m01_hello_python")."""
    for mod in all_modules():
        if module_id in (mod.id, mod.path.name):
            return mod
    known = ", ".join(m.id for m in all_modules())
    raise SystemExit(f"Unknown module {module_id!r}. Available: {known}")


def next_module(completed):
    for mod in all_modules():
        if mod.id not in completed:
            return mod
    return None


def run_tests(mod):
    """Run the module's tests. Returns (passed, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(mod.path), "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=mod.path,
    )
    return result.returncode == 0, result.stdout + result.stderr
