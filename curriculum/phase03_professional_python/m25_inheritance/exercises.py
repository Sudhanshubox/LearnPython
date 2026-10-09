"""m25 exercises: inheritance, composition and interfaces."""

import json
import math
from abc import ABC, abstractmethod


# 1. Shapes.
#    - Shape is an ABSTRACT class with abstract methods area() and perimeter(),
#      and a concrete describe() returning e.g. "Circle with area 3.14" (2 decimals,
#      using the class's name).
#    - Circle(radius), Rectangle(width, height) and Square(side) implement it.
#      Square must inherit from Rectangle and reuse its __init__ via super().
#    - Shape() itself must raise TypeError.
class Shape(ABC):
    pass


class Circle(Shape):
    pass


class Rectangle(Shape):
    pass


class Square(Rectangle):
    pass


# 2. Total area of ANY objects that have an area() method (duck typing: they don't
#    have to be Shapes).
def total_area(shapes):
    raise NotImplementedError


# 3. Employees.
#    - Employee(name, monthly_salary) with annual_pay() = 12 * monthly_salary.
#    - Manager(name, monthly_salary, bonus) inherits from Employee; annual_pay() adds the
#      bonus to the parent's annual_pay() (call it with super()). A manager also has a
#      `reports` list and add_report(employee).
#    - team_cost(manager) = the manager's annual pay plus that of every direct report,
#      and of THEIR reports if they are managers too (recursively).
class Employee:
    pass


class Manager(Employee):
    pass


def team_cost(manager):
    raise NotImplementedError


# 4. Neural-network-style layers (like PyTorch modules).
#    - Layer is abstract with an abstract forward(xs) that takes a list of numbers
#      and returns a NEW list of numbers. Calling a layer like a function,
#      layer(xs), must call forward (implement __call__ once, in Layer).
#    - Scale(factor): multiplies each number.  Shift(amount): adds to each number.
#      ReLU(): replaces negatives with 0.
#    - Sequential(*layers) is ALSO a Layer: forward passes the input through each layer
#      in order. add(layer) appends a layer and returns the Sequential itself (so calls
#      can be chained). len(seq) is the number of direct layers.
#    Sequential(Scale(2), Shift(-1), ReLU())([0, 1, 2]) -> [0, 1, 3]
class Layer(ABC):
    pass


class Scale(Layer):
    pass


class Shift(Layer):
    pass


class ReLU(Layer):
    pass


class Sequential(Layer):
    pass


# 5. Mixins.
#    - ReprMixin.__repr__ -> "ClassName(field=value, ...)" from the instance's attributes,
#      in the order they were set (use vars(self)), values shown with repr().
#    - JsonMixin.to_json() -> json.dumps of the instance's attributes (sort_keys=True),
#      and a classmethod from_json(text) that creates an instance by passing the
#      decoded fields as keyword arguments to the class.
#    - Point(x, y) uses BOTH mixins and defines only __init__.
class ReprMixin:
    pass


class JsonMixin:
    pass


class Point:
    pass
