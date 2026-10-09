# m01 · Hello Python: values, names and types

**By the end you can:** run Python, store and transform data with variables, explain what Python does when you write `x = 5`, and avoid the floating-point bug that catches almost every beginner (and plenty of ML engineers).

**Why it matters for AI:** every tensor in PyTorch is numbers of a specific type (`float32`, `bfloat16`, `int64`). Knowing how numbers are stored helps you understand training instability, memory use and why models are "quantized".

---

## 1. Running Python

Two ways you'll use all the time:

```bash
python3              # interactive REPL: type code, see results immediately
python3 script.py    # run a file
```

Use the REPL to experiment. **Tip:** when you're unsure what something does, try it in the REPL first rather than guessing.

## 2. Values and types

Every value has a **type**, which decides what you can do with it.

```python
>>> type(42)
<class 'int'>
>>> type(3.14)
<class 'float'>
>>> type("hi")
<class 'str'>
>>> type(True)
<class 'bool'>
>>> type(None)
<class 'NoneType'>
```

- `int`: whole numbers of **any size**. `2 ** 1000` works fine (many languages would overflow).
- `float`: decimal numbers stored in 64-bit binary (IEEE 754). Fast, but approximate.
- `str`: text. Immutable: you can't change a string, only make a new one.
- `bool`: `True`/`False`. A subtype of `int` (`True + True == 2`).
- `None`: "no value". Functions without a `return` give back `None`.

## 3. Variables are names, not boxes

```python
a = [1, 2, 3]
b = a          # b is a second name for the SAME list
b.append(4)
print(a)       # [1, 2, 3, 4]  <- surprised?
```

`x = 5` means "make the name `x` refer to the object `5`". Assignment never copies. You can check whether two names point at the same object with `is` or `id()`:

```python
>>> a is b
True
```

This matters a lot later: NumPy arrays and PyTorch tensors behave the same way, and accidental sharing causes real bugs in data pipelines.

## 4. Arithmetic

```python
7 / 2     # 3.5   true division, always a float
7 // 2    # 3     floor division
7 % 2     # 1     remainder
2 ** 10   # 1024  power
divmod(7, 2)  # (3, 1)  quotient and remainder together
-7 // 2   # -4    floors toward negative infinity, not toward zero!
```

## 5. The floating-point trap

```python
>>> 0.1 + 0.2
0.30000000000000004
>>> 0.1 + 0.2 == 0.3
False
```

`0.1` can't be stored exactly in binary, just like 1/3 can't be written exactly in decimal. **Never compare floats with `==`.** Compare with a tolerance instead: `abs(a - b) <= tol`.

## 6. Strings and f-strings

```python
name = "Sudhanshu"
score = 0.98765
print(f"{name} scored {score:.2%}")   # Sudhanshu scored 98.77%
print(f"{1234567.891:,.2f}")          # 1,234,567.89
print(f"{score=}")                    # score=0.98765   (great for debugging)
```

## 7. Input and type conversion

`input()` always returns a `str`. Convert explicitly:

```python
age = int(input("Age? "))
height = float("1.75")
str(42)      # "42"
int("3.5")   # ValueError! Use int(float("3.5")) if you mean it.
```

---

## Problem-solving habit #1: work small examples by hand

Before coding, solve the problem on paper for 2–3 tiny inputs, including an edge case (0, empty, negative). The pattern you notice becomes your algorithm. Use this on exercise 6.

## Go deeper (optional, research-level)

1. Why is `0.1 + 0.2` exactly `0.30000000000000004`? Look at `(0.1).hex()` and read [the Python tutorial on floating point](https://docs.python.org/3/tutorial/floatingpoint.html).
2. Python's `int` has unlimited size. How is it stored in memory? (Hint: look up "CPython long object digits".) Why is `sys.getsizeof(2**1000)` bigger than `sys.getsizeof(1)`?
3. Deep learning often trains in `bfloat16`, which has only 8 bits of mantissa. What's the smallest number you can add to `1.0` in bfloat16 and get something different? Why might that break training?

## Your turn

Open `exercises.py`, fill in each function, then run:

```bash
python -m mentor check m01
```
