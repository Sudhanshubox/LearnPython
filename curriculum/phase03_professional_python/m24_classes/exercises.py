"""m24 exercises: classes and objects. Replace each `raise NotImplementedError`."""


# 1. A bank account.
#    - Account(owner, balance=0); a negative starting balance raises ValueError.
#    - `balance` is a READ-ONLY property (no setter).
#    - deposit(amount) and withdraw(amount): amount must be > 0 (ValueError);
#      withdrawing more than the balance raises ValueError.
#    - history() returns a list of ("deposit"/"withdraw", amount) tuples, oldest first.
#      Returning it must not let callers change the account's real history.
#    - repr(Account("Asha", 50)) == "Account(owner='Asha', balance=50)"
class Account:
    def __init__(self, owner, balance=0):
        raise NotImplementedError


# 2. A temperature with two views.
#    - Temperature(celsius); `celsius` is a property whose setter rejects values
#      below -273.15 with ValueError.
#    - `fahrenheit` is a property with BOTH a getter and a setter (F = C * 9/5 + 32).
#      Setting fahrenheit updates celsius (and applies the same check).
class Temperature:
    def __init__(self, celsius):
        raise NotImplementedError


# 3. A rectangle.
#    - Rectangle(width, height); both must be > 0 (ValueError).
#    - `area` and `perimeter` are read-only properties.
#    - Rectangle.square(side) is a classmethod that returns a Rectangle.
#    - scale(factor) returns a NEW Rectangle; it doesn't change this one.
#    - is_square() returns True if width == height.
class Rectangle:
    def __init__(self, width, height):
        raise NotImplementedError


# 4. A shopping cart. (Watch out for shared mutable state between carts!)
#    - Cart() starts empty.
#    - add(name, price, quantity=1): adding an existing name increases its quantity
#      (and keeps the newest price).
#    - remove(name): removes the item entirely; KeyError if absent.
#    - total(): sum of price * quantity.
#    - Cart.count is a CLASS attribute: how many carts have been created so far.
class Cart:
    count = 0

    def __init__(self):
        raise NotImplementedError


# 5. A tiny linear model y = weight * x + bias, trained by gradient descent.
#    - LinearModel(weight=0.0, bias=0.0)
#    - predict(x) -> weight * x + bias
#    - mse(xs, ys) -> mean of (predict(x) - y)² over the data
#    - step(xs, ys, lr): ONE gradient-descent update using
#          dw = (2 / n) * sum((predict(x) - y) * x)
#          db = (2 / n) * sum(predict(x) - y)
#          weight -= lr * dw;  bias -= lr * db
#      (compute BOTH gradients before changing either parameter!)
#    - fit(xs, ys, lr=0.01, epochs=1000): call step `epochs` times and return the list
#      of mse values, one recorded BEFORE each step.
#    Training on points from y = 3x + 2 should find weight ≈ 3, bias ≈ 2.
class LinearModel:
    def __init__(self, weight=0.0, bias=0.0):
        raise NotImplementedError
