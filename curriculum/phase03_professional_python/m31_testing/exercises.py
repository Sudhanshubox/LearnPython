"""m31: write tests for shop.py here.

Your tests must:
- all PASS on the real shop.py, and
- catch every one of 12 hidden buggy versions of shop.py (each must make at least
  one of your tests fail).
Also use: @pytest.mark.parametrize, a @pytest.fixture of your own, pytest.raises,
pytest.approx, and the built-in tmp_path fixture.

Run them yourself with:  pytest exercises.py -v
Two example tests are written for you. Add many more!
"""

import pytest

from shop import Inventory, apply_discount, parse_duration, split_bill, standardize


def test_discount_example():
    assert apply_discount(200, 10) == 180


def test_split_bill_example():
    assert split_bill(1000, 3) == [334, 333, 333]


# Your tests below: apply_discount, split_bill, parse_duration, Inventory
# (including save/load), and standardize.
