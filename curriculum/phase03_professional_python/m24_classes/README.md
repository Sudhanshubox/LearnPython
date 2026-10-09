# m24 · Classes and objects

**By the end you can:** design classes that bundle data with the behaviour that belongs to it, protect an object's invariants with properties, write alternative constructors, and know when a class is (and isn't) the right tool.

**Why it matters for AI:** in PyTorch every model is a class (`class MyModel(nn.Module)`) that holds its **state** (the weights) and its **behaviour** (`forward`). Optimizers, datasets, tokenizers and data loaders are all classes. In exercise 7 you'll write a tiny model class and train it with gradient descent: the same loop that trains GPT, at toy size.

---

## 1. Why classes?

So far, data lived in dicts and lists, and functions operated on them. That works, but nothing stops code from putting an account into an invalid state:

```python
account = {"owner": "Asha", "balance": 100}
account["balance"] -= 500            # oops: negative balance, nobody checked
```

A **class** keeps data and the rules for changing it together, so the rules can't be bypassed by accident.

## 2. Defining a class

```python
class Account:
    def __init__(self, owner, balance=0):       # runs when you create an Account
        self.owner = owner                      # instance attributes
        self.balance = balance

    def deposit(self, amount):                  # a method: a function that belongs to the class
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount

a = Account("Asha")       # create an object (an instance)
a.deposit(50)             # Python calls Account.deposit(a, 50)
a.balance                 # 50
b = Account("Ben", 10)    # a separate object with its own attributes
```

`self` is the object the method was called on. Python passes it automatically.

## 3. Class attributes vs instance attributes

```python
class Account:
    bank_name = "PyBank"          # class attribute: shared by ALL accounts
    def __init__(self, owner):
        self.owner = owner        # instance attribute: one per object
```

**Classic bug:** a mutable class attribute is shared by every instance.

```python
class Cart:
    items = []                     # BUG: one list for every cart in the program
    def add(self, x):
        self.items.append(x)

class Cart:
    def __init__(self):
        self.items = []            # correct: each cart gets its own list
```

## 4. Readable objects: __repr__

```python
class Account:
    ...
    def __repr__(self):
        return f"Account(owner={self.owner!r}, balance={self.balance})"
```

Now `print(a)` and the debugger show something useful instead of `<Account object at 0x10f...>`. (m26 covers all the "dunder" methods.)

## 5. Properties: attributes with rules

A **property** looks like an attribute but runs code:

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius               # goes through the setter below

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self):                    # computed, always in sync
        return self._celsius * 9 / 5 + 32
```

A leading underscore (`_celsius`) is a convention meaning "internal: don't touch from outside". Python doesn't enforce it; it's a promise between programmers.

A property **without** a setter is read-only: `t.fahrenheit = 5` raises `AttributeError`.

## 6. Class methods and static methods

```python
class Rectangle:
    def __init__(self, width, height):
        self.width, self.height = width, height

    @classmethod
    def square(cls, side):           # alternative constructor; cls is the class itself
        return cls(side, side)

    @staticmethod
    def is_valid(size):              # a plain function that lives in the class's namespace
        return size > 0

Rectangle.square(3)
```

## 7. Invariants: the real reason classes exist

An **invariant** is a rule that must always be true for an object, like "balance is never negative" or "a temperature is never below absolute zero". A good class makes it *impossible* to break its invariants from outside: check them in `__init__` and in every method that changes state. If the rules are enforced in one place, the rest of your program can trust every object it receives.

## 8. When *not* to write a class

If you only need to group a few values, use a tuple, dict or dataclass (m27). If you only need behaviour without state, a function is simpler. A class with just `__init__` and one other method is usually a function in disguise.

---

## Problem-solving habit #22: nouns and verbs

When modelling a problem, the important **nouns** often become classes and attributes (account, balance, transaction), and the **verbs** become methods (deposit, withdraw, transfer). Then ask: what must always be true about each noun? Those are your invariants.

## Go deeper (optional, research-level)

1. Every Python object has a `__dict__`. Create an object, inspect `obj.__dict__` and `type(obj).__dict__`, and explain where attribute lookup finds `obj.method` vs `obj.attribute`.
2. How does `@property` actually work? Read the "Descriptor HowTo Guide" in the Python docs; properties are built from descriptors.
3. Look at PyTorch's `nn.Linear` source code. Which attributes are the learnable parameters, and what does `forward` compute? Compare with your `LinearModel`.

## Your turn

Open the **Exercises** tab, solve each class, then press **Run tests**.
