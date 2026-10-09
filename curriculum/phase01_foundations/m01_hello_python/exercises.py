"""m01 exercises. Replace each `raise NotImplementedError` with your solution.

Check your work: press "Run tests" (or run `python -m mentor check m01`).
Stuck? Press "Hint" in the mentor panel.
"""


# 1. Convert a temperature from Celsius to Fahrenheit.
#    Formula: F = C * 9/5 + 32
def celsius_to_fahrenheit(celsius):
    raise NotImplementedError


# 2. Split a number of seconds into (hours, minutes, seconds).
#    seconds_to_hms(3725) -> (1, 2, 5)
#    Hint: divmod() is your friend.
def seconds_to_hms(total_seconds):
    raise NotImplementedError


# 3. Build a receipt line. The total must have exactly 2 decimal places.
#    receipt_line("Notebook", 45.5, 3) -> "Notebook x3 = Rs 136.50"
def receipt_line(item, price, quantity):
    raise NotImplementedError


# 4. Return True if two floats are equal within `tolerance`.
#    Don't use math.isclose: write the comparison yourself.
#    nearly_equal(0.1 + 0.2, 0.3) -> True
def nearly_equal(a, b, tolerance=1e-9):
    raise NotImplementedError


# 5. Sum the digits of a non-negative integer.
#    digit_sum(9045) -> 18
#    Challenge: solve it once with str() and once using only arithmetic (% and //).
def digit_sum(n):
    raise NotImplementedError


# 6. Return the last digit of base ** exponent.
#    last_digit_of_power(7, 3) -> 3   (7**3 = 343)
#    It must be fast even for exponent = 10**18, where base ** exponent would
#    have more digits than atoms in the universe. Work small examples by hand first!
def last_digit_of_power(base, exponent):
    raise NotImplementedError
