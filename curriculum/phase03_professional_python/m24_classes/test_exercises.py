import pytest

from exercises import Account, Cart, LinearModel, Rectangle, Temperature

# ---------------------------------------------------------------- Account


def test_account_basics():
    a = Account("Asha", 50)
    a.deposit(25)
    a.withdraw(10)
    assert a.balance == 65
    assert a.owner == "Asha"
    assert a.history() == [("deposit", 25), ("withdraw", 10)]
    assert repr(Account("Asha", 50)) == "Account(owner='Asha', balance=50)"


def test_account_rules():
    a = Account("Ben")
    assert a.balance == 0
    with pytest.raises(ValueError):
        a.deposit(0)
    with pytest.raises(ValueError):
        a.withdraw(-5)
    with pytest.raises(ValueError):
        a.withdraw(1)
    with pytest.raises(ValueError):
        Account("Neg", -1)


def test_account_balance_is_read_only():
    a = Account("Asha", 10)
    with pytest.raises(AttributeError):
        a.balance = 1_000_000


def test_account_history_is_protected():
    a = Account("Asha")
    a.deposit(5)
    a.history().append(("deposit", 999))
    assert a.history() == [("deposit", 5)]


def test_accounts_are_independent():
    a, b = Account("A"), Account("B")
    a.deposit(5)
    assert b.history() == [] and b.balance == 0


# ---------------------------------------------------------------- Temperature


def test_temperature():
    t = Temperature(100)
    assert t.fahrenheit == pytest.approx(212)
    t.fahrenheit = 32
    assert t.celsius == pytest.approx(0)
    t.celsius = -40
    assert t.fahrenheit == pytest.approx(-40)


def test_temperature_absolute_zero():
    with pytest.raises(ValueError):
        Temperature(-300)
    t = Temperature(0)
    with pytest.raises(ValueError):
        t.celsius = -274
    with pytest.raises(ValueError):
        t.fahrenheit = -500
    assert t.celsius == 0


# ---------------------------------------------------------------- Rectangle


def test_rectangle():
    r = Rectangle(3, 4)
    assert r.area == 12
    assert r.perimeter == 14
    assert not r.is_square()
    s = Rectangle.square(5)
    assert isinstance(s, Rectangle) and s.is_square() and s.area == 25


def test_rectangle_scale_returns_new():
    r = Rectangle(2, 3)
    big = r.scale(2)
    assert (big.width, big.height) == (4, 6)
    assert (r.width, r.height) == (2, 3)


def test_rectangle_rules():
    with pytest.raises(ValueError):
        Rectangle(0, 5)
    with pytest.raises(ValueError):
        Rectangle(5, -1)
    with pytest.raises(AttributeError):
        Rectangle(1, 1).area = 10


# ---------------------------------------------------------------- Cart


def test_cart():
    c = Cart()
    c.add("pen", 10)
    c.add("book", 250, 2)
    c.add("pen", 12, 3)          # now 4 pens at 12
    assert c.total() == 4 * 12 + 2 * 250
    c.remove("book")
    assert c.total() == 48
    with pytest.raises(KeyError):
        c.remove("laptop")


def test_carts_do_not_share_items():
    a, b = Cart(), Cart()
    a.add("tea", 5)
    assert b.total() == 0


def test_cart_count():
    before = Cart.count
    Cart()
    Cart()
    assert Cart.count == before + 2


# ---------------------------------------------------------------- LinearModel


def test_linear_model_predict_and_mse():
    m = LinearModel(2, 1)
    assert m.predict(3) == 7
    assert m.mse([0, 1], [1, 5]) == pytest.approx((0 + 4) / 2)


def test_linear_model_single_step():
    m = LinearModel()
    m.step([1, 2], [2, 4], lr=0.1)
    # predictions 0, errors -2 and -4: dw = (2/2)(-2*1 + -4*2) = -10, db = (2/2)(-6) = -6
    assert m.weight == pytest.approx(1.0)
    assert m.bias == pytest.approx(0.6)


def test_linear_model_learns_a_line():
    xs = [x / 10 for x in range(-20, 21)]
    ys = [3 * x + 2 for x in xs]
    m = LinearModel()
    losses = m.fit(xs, ys, lr=0.1, epochs=500)
    assert len(losses) == 500
    assert losses[0] > losses[-1]
    assert m.weight == pytest.approx(3, abs=1e-3)
    assert m.bias == pytest.approx(2, abs=1e-3)
