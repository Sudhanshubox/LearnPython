# m07 · Functions, scope and recursion

**By the end you can:** design clean functions with flexible parameters, avoid the mutable-default trap, pass functions around like data, and solve problems recursively (and make recursion fast with memoization).

**Why it matters for AI:** a neural network is a big composition of functions, `f(g(h(x)))`, and training needs the *derivative* of that composition. In exercise 10 you'll compute derivatives numerically by passing a function into another function, which is how you'll check your own backpropagation code in Phase 6.

---

## 1. Defining and calling

```python
def area(width, height):
    """Return the area of a rectangle."""     # docstring: shown by help(area)
    return width * height

area(3, 4)                   # positional arguments
area(height=4, width=3)      # keyword arguments: order doesn't matter, and it reads clearly
```

A function without `return` returns `None`. A function can return several values as a tuple:

```python
def min_max(nums):
    return min(nums), max(nums)

lo, hi = min_max([3, 1, 4])
```

## 2. Parameters: defaults, *args, **kwargs, keyword-only

```python
def greet(name, greeting="Hello"):         # default value
    return f"{greeting}, {name}"

def total(*nums):                           # any number of positional args, as a tuple
    return sum(nums)
total(1, 2, 3)                              # 6

def config(**options):                      # any keyword args, as a dict
    return options
config(lr=0.01, epochs=10)                  # {'lr': 0.01, 'epochs': 10}

def train(model, *, lr=0.01, epochs=10):    # everything after * must be passed by keyword
    ...
train(m, lr=0.1)                            # OK
train(m, 0.1)                               # TypeError: protects you from mixing up arguments
```

You can also unpack when calling: `area(*[3, 4])`, `config(**settings)`.

## 3. The mutable default trap

```python
def add_item(item, items=[]):     # BUG: the [] is created ONCE, when the def runs
    items.append(item)
    return items

add_item(1)   # [1]
add_item(2)   # [1, 2]   ← surprise!

def add_item(item, items=None):   # the standard fix
    if items is None:
        items = []
    items.append(item)
    return items
```

## 4. Scope: where names live

Python looks names up in this order (**LEGB**): **L**ocal (inside the function) → **E**nclosing functions → **G**lobal (the module) → **B**uilt-in (`len`, `print`...).

```python
rate = 0.1                  # global

def interest(amount):
    bonus = 5               # local: only exists while the function runs
    return amount * rate + bonus
```

Assigning to a name inside a function makes it local. `global` and `nonlocal` let you rebind outer names, but needing them is usually a sign to pass values in and return them out instead.

**Tip:** prefer **pure functions**: output depends only on the inputs, and nothing outside is changed. They're easy to test, reason about and reuse.

## 5. Functions are values

```python
def shout(s):
    return s.upper() + "!"

f = shout                  # no parentheses: the function itself, not a call
f("hi")                    # 'HI!'

def apply_twice(func, x):
    return func(func(x))

apply_twice(shout, "hi")   # 'HI!!'

square = lambda x: x * x   # small anonymous function
sorted(words, key=lambda w: w[-1])
```

A function that takes or returns functions is called a **higher-order function**. A function created inside another function remembers the variables around it; that's a **closure**:

```python
def make_multiplier(n):
    def multiply(x):
        return x * n        # n is remembered
    return multiply

double = make_multiplier(2)
double(21)                  # 42
```

## 6. Recursion

A recursive function calls itself on a smaller version of the problem. Every recursive function needs:

1. a **base case** that's answered directly, and
2. a **recursive case** that moves toward the base case.

```python
def countdown(n):
    if n == 0:              # base case
        print("Liftoff!")
        return
    print(n)
    countdown(n - 1)        # smaller problem
```

Each call gets its own frame on the **call stack**. Python limits the depth to about 1000 by default (`RecursionError`), so very deep recursion should become a loop.

## 7. Memoization: remembering answers

```python
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)     # fib(40) takes ages: the same values are recomputed millions of times
```

Remember each answer the first time you compute it:

```python
def fib(n, memo={}):          # (a deliberate, documented use of a shared default)
    if n < 2:
        return n
    if n not in memo:
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]

from functools import cache   # the standard-library way
@cache
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
```

This turns exponential time into linear time. It's the core idea of **dynamic programming** (Phase 2).

---

## Problem-solving habit #7: trust the recursion

To write a recursive function, don't trace every call. Assume the function already works for smaller inputs, and ask: "how do I build the answer for `n` from the answer for a smaller input?" Then handle the base case. For permutations (exercise 9): if you could get all permutations of the *rest* of the string, how would you get permutations of the whole?

## Go deeper (optional, research-level)

1. The central difference `(f(x+h) - f(x-h)) / 2h` is much more accurate than the forward difference `(f(x+h) - f(x)) / h`. Use a Taylor expansion to show its error shrinks like h² instead of h. Then try h = 1e-1, 1e-5, 1e-10, 1e-15 on `f(x) = x**3` at x = 2. Why does the error get *worse* for tiny h? (Hint: floating-point, m01.)
2. Automatic differentiation (what PyTorch does) computes exact derivatives without choosing h. Skim the micrograd repository by Andrej Karpathy; you'll build one in Phase 6.
3. Why doesn't Python optimize tail calls? Read Guido van Rossum's 2009 blog post "Tail Recursion Elimination".

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
