# m25 · Inheritance, composition and interfaces

**By the end you can:** reuse and extend classes with inheritance and `super()`, define interfaces with abstract base classes, combine behaviour with mixins, and (most importantly) choose composition over inheritance when it fits better.

**Why it matters for AI:** PyTorch is designed around these ideas. Every layer **inherits** from `nn.Module` and **overrides** `forward`. A model is a **composition** of layers (`nn.Sequential(nn.Linear(...), nn.ReLU(), ...)`), and a `Sequential` is itself a Module, so models nest inside models. scikit-learn's estimators all share one interface (`fit` / `predict` / `transform`), which is why any of them can be dropped into a `Pipeline`. In exercise 4 you'll build that design yourself.

---

## 1. Inheritance: "is a"

```python
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def annual_pay(self):
        return self.salary * 12

class Manager(Employee):                    # a Manager IS AN Employee
    def __init__(self, name, salary, bonus):
        super().__init__(name, salary)      # let the parent set up its part
        self.bonus = bonus

    def annual_pay(self):                   # override...
        return super().annual_pay() + self.bonus   # ...and extend the parent's version
```

A `Manager` gets every method of `Employee` for free, can **override** methods, and can call the parent's version with `super()`.

```python
isinstance(m, Employee)   # True: a Manager is an Employee
issubclass(Manager, Employee)
```

## 2. Polymorphism

Code written for the parent works with any child:

```python
def payroll(staff):
    return sum(person.annual_pay() for person in staff)   # each object uses its own annual_pay
```

`payroll` doesn't know or care which kinds of employee exist. Add a `Contractor` class next year and `payroll` still works unchanged.

## 3. Abstract base classes: interfaces with teeth

An **abstract** class defines what methods subclasses *must* provide, and can't be instantiated itself:

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        ...

    def describe(self):                      # concrete methods can use abstract ones
        return f"{type(self).__name__} with area {self.area():.2f}"

class Circle(Shape):
    def __init__(self, r):
        self.r = r
    def area(self):
        return 3.141592653589793 * self.r ** 2

Shape()      # TypeError: Can't instantiate abstract class Shape
```

Forget to implement `area` in a subclass, and you get an error the moment you create one, not much later when something calls `area()`.

## 4. Duck typing and protocols

"If it walks like a duck and quacks like a duck, it's a duck." Python doesn't require inheritance for polymorphism: any object with an `area()` method works in `sum(s.area() for s in shapes)`. ABCs are for when you want the interface written down and enforced; duck typing is for when you don't. (`typing.Protocol`, in m27, lets type checkers verify duck typing.)

## 5. Mixins and multiple inheritance

A **mixin** is a small class that adds one capability, meant to be combined with others:

```python
class ReprMixin:
    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({fields})"

class Point(ReprMixin):
    def __init__(self, x, y):
        self.x, self.y = x, y

Point(1, 2)        # Point(x=1, y=2)
```

With several parents, Python searches them in the **method resolution order** (`Point.__mro__`). Keep mixins small and independent so the order doesn't matter.

## 6. Composition: "has a"

```python
class Engine:
    def start(self):
        return "vroom"

class Car:
    def __init__(self):
        self.engine = Engine()       # a Car HAS AN Engine
    def start(self):
        return self.engine.start()   # delegate
```

**Prefer composition over inheritance** when the relationship is "has a" or "uses a". Inheritance creates tight coupling: a change in the parent can break every child, and deep hierarchies become hard to follow. Composition lets you swap parts (a different Engine) without touching the Car.

A powerful pattern combining both: the **composite**. A `Sequential` of layers is itself a layer, so it can contain other `Sequential`s. That's how PyTorch models nest blocks inside blocks.

## 7. Liskov substitution principle

A subclass should work anywhere its parent works. The classic violation: `Square(Rectangle)`. If code sets a rectangle's width without touching its height, a square either breaks that expectation or stops being a square. When a child can't keep the parent's promises, it shouldn't be a child.

---

## Problem-solving habit #23: design the interface first

Before writing classes, write the code that *uses* them: `model = Sequential(Scale(2), ReLU()); model.forward([1, -3])`. If the calling code reads clearly, the design is probably right. Then implement the classes to make that code work.

## Go deeper (optional, research-level)

1. Print `bool.__mro__` and `type(True).__mro__`. Then look up how Python computes the MRO with **C3 linearization**, and work it out by hand for a "diamond" (`class D(B, C)` where `B` and `C` both inherit from `A`).
2. Read the scikit-learn paper *API design for machine learning software: experiences from the scikit-learn project* (Buitinck et al., 2013). Which design principles from this module do they describe?
3. Read the source of PyTorch's `nn.Sequential` (it's short). How does it register its children, and how does `forward` use them?

## Your turn

Open the **Exercises** tab, solve each part, then press **Run tests**.
