# m03 · Conditionals and boolean logic

**By the end you can:** make programs choose between paths, combine conditions correctly, use Python's "truthiness", and spot the classic logic bugs.

**Why it matters for AI:** a classifier turns a probability into a decision with a condition (`if p >= 0.5`). Evaluating models means counting true/false positives and negatives, which is pure boolean logic. You'll write exactly that in exercise 5.

---

## 1. Comparisons produce booleans

```python
3 < 5          # True
3 == 3.0       # True   (== compares values)
"a" < "b"      # True   (strings compare by code point, letter by letter)
"Z" < "a"      # True!  uppercase letters come before lowercase
1 <= x < 10    # chained: same as 1 <= x and x < 10
x != y         # not equal
```

**Tip:** `==` asks "same value?", `is` asks "same object?". Use `is` only for `None`: `if result is None:`.

## 2. if / elif / else

```python
if temperature > 30:
    print("hot")
elif temperature > 15:      # only checked if the first condition was False
    print("pleasant")
else:
    print("cold")
```

Python uses **indentation** (4 spaces) to mark blocks. Order matters: the first true branch wins, so put the most specific conditions first.

## 3. and, or, not

```python
age >= 18 and has_id          # both must be True
is_admin or is_owner          # at least one
not is_banned
```

They **short-circuit**: `and` stops at the first false value, `or` at the first true value. That's useful for guarding:

```python
if items and items[0] == "x":   # items[0] is never evaluated when items is empty
    ...
```

## 4. Truthiness

Any value can be used as a condition. These are **falsy**: `False`, `None`, `0`, `0.0`, `""`, `[]`, `{}`, `set()`. Everything else is truthy.

```python
name = input("Name? ")
if not name:
    print("You didn't type anything")
```

`and`/`or` return one of their operands, not necessarily `True`/`False`:

```python
username = given_name or "guest"   # "guest" if given_name is empty
```

## 5. The conditional expression

```python
label = "even" if n % 2 == 0 else "odd"
```

Use it for simple either/or values. If it needs explaining, use a normal `if`.

## 6. match (Python 3.10+)

```python
match command.split():
    case ["go", direction]:
        move(direction)
    case ["quit" | "exit"]:
        stop()
    case _:
        print("Unknown command")
```

`match` compares *structure*, not just values. It shines when parsing commands or data shapes.

## 7. Classic bugs

```python
if x == 1 or 2:          # ALWAYS True: means (x == 1) or (2), and 2 is truthy
if x == 1 or x == 2:     # correct
if x in (1, 2):          # better

if 0.1 + 0.2 == 0.3:     # False (remember m01): compare floats with a tolerance
```

**De Morgan's laws** help you simplify negations:
`not (a and b)` equals `(not a) or (not b)`, and `not (a or b)` equals `(not a) and (not b)`.

---

## Problem-solving habit #3: enumerate the cases

Before writing branches, list every case in a small table, including edge cases (boundaries like exactly 90, zero, negatives). Then check that each row maps to exactly one branch. Exercise 4 is much easier this way.

## Go deeper (optional, research-level)

1. In NumPy, `if np.array([1, 2]):` raises *"The truth value of an array with more than one element is ambiguous"*. Why did the designers choose to raise instead of picking a rule? (You'll meet `a.any()` and `a.all()` in Phase 4.)
2. Write `not (a or b)` and `(not a) and (not b)` as truth tables to prove De Morgan's law for all four input combinations.
3. Precision and recall are built from the TP/FP/FN counts in exercise 5. Why is accuracy misleading when 99% of emails are not spam?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
