"""Grades YOUR tests (in exercises.py) by mutation testing.

Spoiler warning: the list of planted bugs is below. Try to write thorough tests
before reading it.
"""

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SHOP = (HERE / "shop.py").read_text(encoding="utf-8")
SHOP_SHA256 = "15e201baa7e68b57ad1d43d5d3ca9a403aee8408b036b33f20181f8f30b07243"

MUTANTS = {
    "discount_not_rounded": (
        "return round(price * (100 - percent) / 100, 2)",
        "return price * (100 - percent) / 100",
    ),
    "discount_no_upper_limit": (
        "if not 0 <= percent <= 100:",
        "if percent < 0:",
    ),
    "discount_rejects_100_percent": (
        "if not 0 <= percent <= 100:",
        "if not 0 <= percent < 100:",
    ),
    "split_extra_cents_to_last_people": (
        "base + 1 if i < extra else base",
        "base + 1 if i >= people - extra else base",
    ),
    "split_no_people_check": (
        'if people < 1:\n        raise ValueError("need at least one person")\n    ',
        "",
    ),
    "duration_case_sensitive": (
        'text.replace(" ", "").lower()',
        'text.replace(" ", "")',
    ),
    "duration_accepts_empty": (
        "if not cleaned or not match:",
        "if not match:",
    ),
    "remove_cannot_empty_stock": (
        "if quantity > self._stock.get(item, 0):",
        "if quantity >= self._stock.get(item, 0):",
    ),
    "remove_keeps_zero_items": (
        "        if self._stock[item] == 0:\n            del self._stock[item]\n",
        "",
    ),
    "save_drops_single_items": (
        "json.dump(self._stock, f)",
        "json.dump({k: v for k, v in self._stock.items() if v > 1}, f)",
    ),
    "standardize_sample_std": (
        "std = statistics.pstdev(values)",
        "std = statistics.stdev(values) if len(values) > 1 else 0",
    ),
    "standardize_divides_by_zero": (
        "    if std == 0:\n        return [0.0] * len(values)\n",
        "",
    ),
}


def run_my_tests(tmp_path, shop_source):
    (tmp_path / "shop.py").write_text(shop_source, encoding="utf-8")
    (tmp_path / "exercises.py").write_text((HERE / "exercises.py").read_text(encoding="utf-8"), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "exercises.py", "-q", "-x", "-p", "no:cacheprovider", "--timeout=10"],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )


def test_shop_py_is_unchanged():
    assert hashlib.sha256(SHOP.encode()).hexdigest() == SHOP_SHA256, (
        "shop.py was modified. Restore it (git checkout) and write your tests in exercises.py."
    )


@pytest.mark.timeout(60)
def test_your_tests_pass_on_the_real_code(tmp_path):
    result = run_my_tests(tmp_path, SHOP)
    assert result.returncode == 0, "Your tests fail on the CORRECT shop.py:\n" + result.stdout[-3000:]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_your_tests_catch_bug(tmp_path, name):
    old, new = MUTANTS[name]
    assert old in SHOP, "internal error: mutant doesn't apply"
    result = run_my_tests(tmp_path, SHOP.replace(old, new, 1))
    assert result.returncode != 0, (
        f"A buggy version of shop.py ({name.replace('_', ' ')}) passed all your tests. "
        "Add a test that would catch it."
    )


def test_uses_pytest_features():
    source = (HERE / "exercises.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    tests = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    fixtures = [
        n for n in tree.body if isinstance(n, ast.FunctionDef)
        and any("fixture" in ast.unparse(d) for d in n.decorator_list)
    ]
    uses_tmp_path = any("tmp_path" in [a.arg for a in t.args.args] for t in tests)
    assert len(tests) >= 12, f"write at least 12 test functions (found {len(tests)})"
    assert "pytest.mark.parametrize" in source, "use @pytest.mark.parametrize"
    assert fixtures, "define at least one fixture with @pytest.fixture"
    assert "pytest.raises" in source, "use pytest.raises"
    assert "pytest.approx" in source, "use pytest.approx"
    assert uses_tmp_path, "use the tmp_path fixture in a test"
