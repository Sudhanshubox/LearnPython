# m04 · Loops and iteration patterns

**By the end you can:** repeat work with `for` and `while`, control loops with `break`/`continue`, recognize the common loop patterns, and estimate how much work a loop does.

**Why it matters for AI:** training a model *is* a loop: `for epoch in range(epochs): for batch in data: update the weights`. Many algorithms (gradient descent, Newton's method) are loops that repeat until the answer stops changing. You'll write one in exercise 7.

---

## 1. for loops

`for` walks over any **iterable**: strings, lists, ranges, files, and more.

```python
for ch in "abc":
    print(ch)

for i in range(5):        # 0 1 2 3 4
for i in range(2, 10, 3): # 2 5 8      (start, stop, step)
for i in range(5, 0, -1): # 5 4 3 2 1
```

## 2. enumerate and zip

```python
names = ["Asha", "Ben", "Chen"]
scores = [91, 78, 85]

for i, name in enumerate(names, start=1):
    print(i, name)              # 1 Asha, 2 Ben, 3 Chen

for name, score in zip(names, scores):
    print(f"{name}: {score}")   # pairs items up; stops at the shorter one
```

**Tip:** if you write `for i in range(len(items)):` and then use `items[i]`, you almost always want `enumerate` or a direct `for item in items` instead.

## 3. while loops

Use `while` when you don't know in advance how many times to repeat.

```python
n = 27
steps = 0
while n != 1:
    n = n // 2 if n % 2 == 0 else 3 * n + 1
    steps += 1
```

Every `while` loop needs something that eventually makes the condition false. If not, it runs forever (press `Ctrl+C` to stop it).

## 4. break, continue and loop-else

```python
for n in numbers:
    if n < 0:
        continue          # skip to the next item
    if n == 0:
        break             # leave the loop completely
    print(n)

for d in range(2, n):
    if n % d == 0:
        print("not prime")
        break
else:                     # runs only if the loop did NOT break
    print("prime")
```

## 5. The five patterns behind most loops

```python
# Accumulate
total = 0
for x in xs:
    total += x

# Count
count = 0
for x in xs:
    if x > 0:
        count += 1

# Find (search and stop)
found = None
for x in xs:
    if is_good(x):
        found = x
        break

# Best so far
best = xs[0]
for x in xs[1:]:
    if x > best:
        best = x

# Build a new collection
squares = []
for x in xs:
    squares.append(x * x)
```

When you face a new problem, ask: "which pattern is this?"

## 6. Nested loops and how much work they do

```python
for i in range(n):
    for j in range(n):
        ...               # runs n * n times
```

If `n` is 1,000, that's a million steps; if `n` is 100,000, it's 10 billion (far too slow). Counting how the work grows with the input size is called **complexity analysis**. We write it as O(n) for one loop and O(n²) for two nested loops. You'll study this properly in Phase 2.

## 7. Loop invariants: how to know your loop is correct

A **loop invariant** is something that's true before and after every iteration. For the "best so far" pattern: *"`best` is the largest of the items seen so far"*. If it's true at the start and each step keeps it true, it's true at the end, so your answer is correct. This is how computer scientists *prove* algorithms correct.

---

## Problem-solving habit #4: trace by hand

When a loop gives a wrong answer, don't guess. Make a table with one column per variable and one row per iteration, and fill it in for a tiny input. The bug almost always shows up by row 3.

## Go deeper (optional, research-level)

1. The Collatz conjecture (exercise 2) says every positive integer eventually reaches 1. It's unproven! Which number below 10,000 takes the most steps?
2. Your `primes_up_to` with trial division does roughly how many operations for n = 1,000,000? Look up the Sieve of Eratosthenes, and explain why it's O(n log log n).
3. Newton's method (exercise 7) roughly *doubles* the number of correct digits each step ("quadratic convergence"). Print the error after each iteration and check this yourself. Gradient descent, which trains neural networks, usually converges much more slowly. Why might people still prefer it?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
