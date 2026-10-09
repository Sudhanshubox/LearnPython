"""m04 exercises: loops. Replace each `raise NotImplementedError` with your solution."""


# 1. Sum all numbers below `limit` that are multiples of 3 or 5.
#    sum_of_multiples(10) -> 23   (3 + 5 + 6 + 9)
def sum_of_multiples(limit):
    raise NotImplementedError


# 2. Collatz: if n is even, halve it; if odd, make it 3n + 1. Count the steps to reach 1.
#    collatz_steps(1) -> 0, collatz_steps(6) -> 8
def collatz_steps(n):
    raise NotImplementedError


# 3. Is n a prime number? (Numbers below 2 are not prime.)
#    Make it fast: you only need to check divisors up to the square root of n.
def is_prime(n):
    raise NotImplementedError


# 4. Return a list of all primes <= n, in increasing order.
#    primes_up_to(20) -> [2, 3, 5, 7, 11, 13, 17, 19]
#    Must handle n = 200_000 in well under a second (look up the Sieve of Eratosthenes).
def primes_up_to(n):
    raise NotImplementedError


# 5. Reverse the digits of a non-negative integer using only arithmetic (no str()).
#    reverse_digits(1234) -> 4321, reverse_digits(1200) -> 21
def reverse_digits(n):
    raise NotImplementedError


# 6. Greatest common divisor using Euclid's algorithm:
#    gcd(a, b) = gcd(b, a % b), and gcd(a, 0) = a. Use a loop, not recursion or math.gcd.
#    gcd(48, 18) -> 6
def gcd(a, b):
    raise NotImplementedError


# 7. Square root by Newton's method. Start with guess = x (or 1.0 if x < 1),
#    then repeat: new_guess = (guess + x / guess) / 2, and stop once the guess
#    barely changes: abs(new_guess - guess) <= tolerance * new_guess.
#    Return the final guess. For x == 0 return 0.0. Don't use ** 0.5 or math.sqrt.
#    (Why a *relative* tolerance? Try an absolute one with x = 1e10 and see what happens.)
def newton_sqrt(x, tolerance=1e-12):
    raise NotImplementedError


# 8. Return a string with a right-aligned staircase of `n` steps made of '#',
#    lines separated by "\n" (no trailing newline).
#    staircase(3) -> "  #\n ##\n###"
def staircase(n):
    raise NotImplementedError
