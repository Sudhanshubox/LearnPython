"""m03 exercises: conditionals. Replace each `raise NotImplementedError` with your solution."""


# 1. Convert a score (0-100) to a letter grade:
#    90 and above -> "A", 80-89 -> "B", 70-79 -> "C", 60-69 -> "D", below 60 -> "F".
#    Scores outside 0-100 -> "invalid".
def grade(score):
    raise NotImplementedError


# 2. Is `year` a leap year? Rules: divisible by 4, except centuries,
#    except centuries divisible by 400. (2000 yes, 1900 no, 2024 yes, 2023 no)
def is_leap_year(year):
    raise NotImplementedError


# 3. Return "Fizz" for multiples of 3, "Buzz" for multiples of 5,
#    "FizzBuzz" for multiples of both, otherwise the number as a string.
def fizzbuzz(n):
    raise NotImplementedError


# 4. Classify a triangle from its three side lengths:
#    "equilateral" (all equal), "isosceles" (exactly two equal), "scalene" (all different),
#    or "invalid" if any side is <= 0 or the sides can't form a triangle
#    (each side must be shorter than the sum of the other two).
def triangle_type(a, b, c):
    raise NotImplementedError


# 5. A spam classifier predicted `predicted` (True = spam) and the truth is `actual`.
#    Return "TP" (true positive), "FP" (false positive), "TN" or "FN".
def outcome(predicted, actual):
    raise NotImplementedError


# 6. Return the largest of three numbers WITHOUT using max() or sorting.
def largest(a, b, c):
    raise NotImplementedError


# 7. Use a `match` statement. Given a day name in any case, return
#    "weekday", "weekend", or "invalid".
#    day_type("Saturday") -> "weekend", day_type("MONDAY") -> "weekday"
def day_type(day):
    raise NotImplementedError
