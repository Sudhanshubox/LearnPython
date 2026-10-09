import ast
import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from exercises import check_environment, parse_requirement, public_names, satisfies

HERE = Path(__file__).parent


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "textstats", *args], capture_output=True, text=True, cwd=HERE, timeout=30,
    )


# ---------------------------------------------------------------- the package


def test_tokens():
    from textstats.tokens import sentences, words

    assert words("It's 2 o'clock, Asha!") == ["it's", "2", "o'clock", "asha"]
    assert words("") == []
    assert sentences("Hi there. How are you?  Fine!") == ["Hi there.", "How are you?", "Fine!"]
    assert sentences("No punctuation at the end") == ["No punctuation at the end"]
    assert sentences("   ") == []


def test_stats():
    from textstats.stats import avg_sentence_length, top_words, word_count

    text = (HERE / "sample.txt").read_text(encoding="utf-8")
    assert word_count(text) == 17
    assert top_words(text) == [("the", 4), ("cat", 2), ("sat", 2)]
    assert top_words(text, 1) == [("the", 4)]
    assert avg_sentence_length(text) == 4.25
    assert avg_sentence_length("") == 0.0


def test_stats_uses_relative_import():
    tree = ast.parse((HERE / "textstats" / "stats.py").read_text(encoding="utf-8"))
    relative = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "tokens"]
    assert relative, "import from the sibling module with `from .tokens import ...`"


def test_public_api():
    import textstats

    assert textstats.__version__ == "0.1.0"
    assert sorted(textstats.__all__) == ["avg_sentence_length", "top_words", "word_count"]
    assert textstats.word_count("one two") == 2


def test_cli():
    result = run_cli("sample.txt")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["words: 17", "sentences: 4", "top: the, cat, sat"]


def test_cli_missing_file():
    result = run_cli("nope.txt")
    assert result.returncode == 1
    assert "error: no such file: nope.txt" in result.stdout + result.stderr


def test_main_importable_without_running():
    module = importlib.import_module("textstats.__main__")
    assert callable(module.main)


# ---------------------------------------------------------------- pyproject.toml


def test_pyproject():
    data = tomllib.loads((HERE / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert project["name"] == "textstats"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.10"
    assert isinstance(project.get("description"), str) and project["description"].strip()
    assert project["dependencies"] == []
    assert project["scripts"]["textstats"] == "textstats.__main__:main"


# ---------------------------------------------------------------- exercises.py


def test_public_names():
    import textstats

    assert public_names(textstats) == ["avg_sentence_length", "top_words", "word_count"]
    import types
    m = types.ModuleType("demo")
    m.a, m._hidden, m.b = 1, 2, 3
    assert public_names(m) == ["a", "b"]


@pytest.mark.parametrize("line, expected", [
    ("NumPy >= 1.26, <3  # arrays", ("numpy", ">=1.26,<3")),
    ("requests", ("requests", "")),
    ("torch==2.4.1", ("torch", "==2.4.1")),
    ("  # just a comment", None),
    ("", None),
    ("scikit-learn~=1.5", ("scikit-learn", "~=1.5")),
])
def test_parse_requirement(line, expected):
    assert parse_requirement(line) == expected


@pytest.mark.parametrize("version, spec, expected", [
    ("1.26.4", ">=1.26,<3", True),
    ("3.0", ">=1.26,<3", False),
    ("2", "==2.0.0", True),
    ("1.9", ">1.10", False),
    ("1.10", ">1.9", True),
    ("2.4.1", "!=2.4.1", False),
    ("0.1", "", True),
    ("1.0", "<=1", True),
])
def test_satisfies(version, spec, expected):
    assert satisfies(version, spec) == expected


def test_check_environment():
    installed = {"numpy": "2.1.0", "requests": "2.32.3", "pandas": "2.2.0"}
    lines = ["numpy<2", "torch", "# comment", "Requests>=2.0", "pandas>=2,<3"]
    assert check_environment(installed, lines) == ["conflict: numpy 2.1.0 (needs <2)", "missing: torch"]
    assert check_environment({}, []) == []
