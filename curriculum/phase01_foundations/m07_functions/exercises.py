"""m07 exercises: functions. Replace each `raise NotImplementedError` with your solution.

For exercises 1-3 you also need to write the right function SIGNATURE (parameters),
so read the descriptions carefully.
"""


# 1. average(...) takes ANY number of numbers as separate arguments and returns
#    their mean, or 0.0 if there are none.
#    average(1, 2, 3) -> 2.0, average() -> 0.0
def average():
    raise NotImplementedError


# 2. format_price(amount, currency="Rs", *, decimals=2) -> "Rs 12.50"
#    `decimals` must be keyword-only.
#    format_price(12.5) -> "Rs 12.50"
#    format_price(3, "$", decimals=0) -> "$ 3"
def format_price():
    raise NotImplementedError


# 3. This function has the mutable-default bug. Fix it so each call without
#    a `todo` list starts with a fresh empty list.
def add_task(task, todo=[]):
    todo.append(task)
    return todo


# 4. Return a NEW function that computes f(g(x)).
#    inc = lambda x: x + 1; dbl = lambda x: x * 2
#    compose(inc, dbl)(5) -> 11
def compose(f, g):
    raise NotImplementedError


# 5. Return a function that counts how many times it has been called (a closure).
#    counter = make_counter()
#    counter() -> 1, counter() -> 2, counter() -> 3
#    Each make_counter() call starts a separate count.
def make_counter():
    raise NotImplementedError


# 6. Factorial, written RECURSIVELY. factorial(0) -> 1, factorial(5) -> 120
def factorial(n):
    raise NotImplementedError


# 7. The n-th Fibonacci number (fib(0) = 0, fib(1) = 1), written RECURSIVELY
#    but memoized so fib(300) returns instantly.
def fib(n):
    raise NotImplementedError


# 8. Flatten a list that can be nested to any depth, RECURSIVELY.
#    flatten([1, [2, [3, [4]], 5]]) -> [1, 2, 3, 4, 5]
def flatten(nested):
    raise NotImplementedError


# 9. All orderings of the characters in s, as a sorted list without duplicates.
#    Use recursion, not itertools.
#    permutations("abc") -> ["abc", "acb", "bac", "bca", "cab", "cba"]
#    permutations("aab") -> ["aab", "aba", "baa"]
def permutations(s):
    raise NotImplementedError


# 10. Numerical derivative of f at x using the central difference:
#     (f(x + h) - f(x - h)) / (2 * h)
#     derivative(lambda x: x ** 2, 3) -> about 6.0
def derivative(f, x, h=1e-5):
    raise NotImplementedError
