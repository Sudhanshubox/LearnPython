import inspect
import math

import pytest

from exercises import (
    Circle,
    Employee,
    JsonMixin,
    Layer,
    Manager,
    Point,
    Rectangle,
    ReLU,
    ReprMixin,
    Scale,
    Sequential,
    Shape,
    Shift,
    Square,
    team_cost,
    total_area,
)

# ---------------------------------------------------------------- shapes


def test_shape_is_abstract():
    with pytest.raises(TypeError):
        Shape()
    assert inspect.isabstract(Shape)


def test_incomplete_subclass_cannot_be_created():
    class Blob(Shape):
        def area(self):
            return 1

    with pytest.raises(TypeError):
        Blob()


def test_shapes():
    c = Circle(1)
    assert c.area() == pytest.approx(math.pi)
    assert c.perimeter() == pytest.approx(2 * math.pi)
    assert c.describe() == "Circle with area 3.14"
    r = Rectangle(2, 3)
    assert (r.area(), r.perimeter()) == (6, 10)
    assert r.describe() == "Rectangle with area 6.00"
    s = Square(4)
    assert isinstance(s, Rectangle) and isinstance(s, Shape)
    assert (s.area(), s.perimeter()) == (16, 16)
    assert s.describe() == "Square with area 16.00"


def test_square_uses_super():
    assert "super" in inspect.getsource(Square)


def test_total_area_duck_typing():
    class Blob:
        def area(self):
            return 10

    assert total_area([Square(2), Rectangle(1, 3), Blob()]) == 17
    assert total_area([]) == 0


# ---------------------------------------------------------------- employees


def test_employees():
    e = Employee("Asha", 1000)
    m = Manager("Ben", 2000, bonus=5000)
    assert e.annual_pay() == 12_000
    assert m.annual_pay() == 29_000
    assert isinstance(m, Employee)
    assert m.reports == []
    assert "super" in inspect.getsource(Manager)


def test_team_cost():
    ceo = Manager("CEO", 10_000, 50_000)
    cto = Manager("CTO", 8000, 20_000)
    dev1, dev2, sales = Employee("Dev1", 5000), Employee("Dev2", 5000), Employee("Sales", 4000)
    cto.add_report(dev1)
    cto.add_report(dev2)
    ceo.add_report(cto)
    ceo.add_report(sales)
    assert team_cost(cto) == 116_000 + 60_000 + 60_000
    assert team_cost(ceo) == 170_000 + 236_000 + 48_000


def test_reports_not_shared():
    a, b = Manager("A", 1, 0), Manager("B", 1, 0)
    a.add_report(Employee("x", 1))
    assert b.reports == []


# ---------------------------------------------------------------- layers


def test_layer_is_abstract():
    with pytest.raises(TypeError):
        Layer()


def test_basic_layers():
    assert Scale(2).forward([1, -2]) == [2, -4]
    assert Shift(1)([1, -2]) == [2, -1]
    assert ReLU()([-1, 0, 3]) == [0, 0, 3]


def test_layers_return_new_lists():
    xs = [1, -1]
    ReLU()(xs)
    Scale(3)(xs)
    assert xs == [1, -1]


def test_sequential():
    model = Sequential(Scale(2), Shift(-1), ReLU())
    assert isinstance(model, Layer)
    assert model([0, 1, 2]) == [0, 1, 3]
    assert len(model) == 3


def test_sequential_add_chains_and_nests():
    inner = Sequential().add(Scale(10)).add(Shift(1))
    assert len(inner) == 2
    outer = Sequential(inner, ReLU(), inner)
    assert outer([-1, 1]) == [1, 111]     # [-1, 1] -> [-9, 11] -> [0, 11] -> [1, 111]
    assert Sequential()([5]) == [5]


def test_layer_call_defined_once():
    assert "__call__" in vars(Layer)
    for cls in (Scale, Shift, ReLU, Sequential):
        assert "__call__" not in vars(cls), f"{cls.__name__} should inherit __call__ from Layer"


# ---------------------------------------------------------------- mixins


def test_point_mixins():
    p = Point(1, 2)
    assert isinstance(p, ReprMixin) and isinstance(p, JsonMixin)
    assert repr(p) == "Point(x=1, y=2)"
    assert p.to_json() == '{"x": 1, "y": 2}'
    q = Point.from_json('{"x": 5, "y": -3}')
    assert isinstance(q, Point) and (q.x, q.y) == (5, -3)


def test_point_defines_only_init():
    own = {k for k in vars(Point) if not k.startswith("__") or k == "__repr__"}
    assert own == set() and "__init__" in vars(Point)


def test_mixins_work_for_other_classes():
    class User(ReprMixin, JsonMixin):
        def __init__(self, name, tags):
            self.name = name
            self.tags = tags

    u = User("asha", ["admin"])
    assert repr(u) == "User(name='asha', tags=['admin'])"
    assert User.from_json(u.to_json()).tags == ["admin"]
