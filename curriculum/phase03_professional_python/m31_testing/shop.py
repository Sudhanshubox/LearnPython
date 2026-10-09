"""A small shopping library. This code is CORRECT: your job is to test it.

Don't change this file; write your tests in exercises.py.
"""

import json
import re
import statistics


def apply_discount(price, percent):
    """Return price reduced by `percent` percent, rounded to 2 decimal places.

    Raises ValueError if price is negative or percent is outside 0..100 (inclusive).
    apply_discount(200, 10) -> 180.0
    """
    if price < 0:
        raise ValueError("price must not be negative")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (100 - percent) / 100, 2)


def split_bill(total_cents, people):
    """Split a bill (in integer cents) between `people` as evenly as possible.

    Returns a list of `people` integer amounts that add up exactly to total_cents.
    When it doesn't divide evenly, the FIRST people each pay one extra cent.
    split_bill(1000, 3) -> [334, 333, 333]
    Raises ValueError if people < 1 or total_cents < 0.
    """
    if people < 1:
        raise ValueError("need at least one person")
    if total_cents < 0:
        raise ValueError("total must not be negative")
    base, extra = divmod(total_cents, people)
    return [base + 1 if i < extra else base for i in range(people)]


def parse_duration(text):
    """Parse durations like "1h30m", "45m", "2h" into minutes.

    Spaces are ignored and letters are case-insensitive ("1H 5M" -> 65).
    Raises ValueError for empty or malformed text (like "", "h", "1x", "30m1h").
    """
    cleaned = text.replace(" ", "").lower()
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?", cleaned)
    if not cleaned or not match:
        raise ValueError(f"bad duration: {text!r}")
    hours, minutes = match.groups()
    return int(hours or 0) * 60 + int(minutes or 0)


class Inventory:
    """Item quantities. Quantities are never negative."""

    def __init__(self):
        self._stock = {}

    def add(self, item, quantity):
        """Add stock. quantity must be > 0 (ValueError)."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._stock[item] = self._stock.get(item, 0) + quantity

    def remove(self, item, quantity):
        """Remove stock. ValueError if quantity <= 0 or more than is in stock.
        Items that reach 0 are deleted from the inventory."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self._stock.get(item, 0):
            raise ValueError("not enough stock")
        self._stock[item] -= quantity
        if self._stock[item] == 0:
            del self._stock[item]

    def quantity(self, item):
        """Current quantity (0 for unknown items)."""
        return self._stock.get(item, 0)

    def items(self):
        """Sorted list of item names in stock."""
        return sorted(self._stock)

    def save(self, path):
        """Write the inventory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._stock, f)

    @classmethod
    def load(cls, path):
        """Read an inventory written by save()."""
        inv = cls()
        with open(path, encoding="utf-8") as f:
            inv._stock = {k: int(v) for k, v in json.load(f).items()}
        return inv


def standardize(values):
    """Rescale values to mean 0 and standard deviation 1 (population std, like NumPy's default).

    If all values are equal, return a list of 0.0s. Raises ValueError for an empty list.
    standardize([1, 2, 3]) -> [-1.2247..., 0.0, 1.2247...]
    """
    if not values:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]
