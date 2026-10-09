"""m56 exercises: a scalar autograd engine and a tiny neural network library.

Use only the standard library (math, random). No NumPy, no PyTorch.
"""

import math
import random


# 1. The Value class. __init__ and __repr__ are written for you.
#    Every operation returns a NEW Value whose _prev holds the operands and whose
#    _backward pushes out.grad into the operands' .grad with += (see the README table).
#    Wrap plain numbers in Value so that Value(2) * 3 and 3 * Value(2) both work.
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    # 1a. Primitives: implement these with their own _backward.
    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __pow__(self, exponent):          # exponent is a plain int or float, not a Value
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    # 1b. Everything else, built from the primitives above (no new _backward needed).
    def __neg__(self):                    # -self
        raise NotImplementedError

    def __sub__(self, other):             # self - other
        raise NotImplementedError

    def __truediv__(self, other):         # self / other
        raise NotImplementedError

    def __radd__(self, other):            # other + self
        raise NotImplementedError

    def __rmul__(self, other):            # other * self
        raise NotImplementedError

    def __rsub__(self, other):            # other - self
        raise NotImplementedError

    def __rtruediv__(self, other):        # other / self
        raise NotImplementedError

    # 3. Backpropagation: set self.grad = 1.0, then call _backward on every node in
    #    reversed topological order (see topological_order below).
    def backward(self):
        raise NotImplementedError


# 2. Every node reachable from root through _prev, each exactly once, with every node
#    appearing AFTER all of its children (root is last).
#    Do it ITERATIVELY with an explicit stack: the tests build a chain 5,000 nodes deep,
#    which would overflow Python's recursion limit.
def topological_order(root):
    raise NotImplementedError


# 4. Set .grad = 0.0 on every Value in params.
def zero_grad(params):
    raise NotImplementedError


# 5. A neuron with nin inputs: weights Value(rng.uniform(-1, 1)) (drawn in order, one per
#    input) and bias Value(0.0). Calling it on a list of nin numbers or Values returns
#    sum(w_i * x_i) + b, passed through tanh if nonlin is True.
#    parameters() returns the weights followed by the bias.
class Neuron:
    def __init__(self, nin, nonlin=True, rng=None):
        rng = rng or random.Random(0)
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


# 6. A layer of nout neurons (created in order, sharing rng), all seeing the same inputs.
#    Calling it returns a list of nout Values. parameters() concatenates the neurons'.
class Layer:
    def __init__(self, nin, nout, nonlin=True, rng=None):
        rng = rng or random.Random(0)
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


# 7. An MLP: MLP(2, [8, 8, 1]) has layers 2->8, 8->8, 8->1. Create ONE random.Random(seed)
#    and pass it to every layer. Every layer uses tanh except the last, which is linear.
#    Calling it returns the last layer's list of Values, or a single Value if that layer
#    has one neuron.
class MLP:
    def __init__(self, nin, nouts, seed=0):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


# 8. Cross-entropy for one example: logits is a list of Values, target an int class index.
#    Return logsumexp(logits) - logits[target] as a Value, subtracting the max .data (a plain
#    float) before exponentiating so that logits like [1000, 0] don't overflow.
def cross_entropy(logits, target):
    raise NotImplementedError


# 9. Train a binary classifier with labels y in {-1, +1} using the mean squared error
#    loss = sum((model(x) - y)^2 for each example) / len(X) and plain gradient descent.
#    Each step: compute the loss, zero_grad, backward, update every parameter by -lr * grad.
#    Return the list of loss values (floats), one per step, computed BEFORE that step's update.
def train(model, X, y, steps=100, lr=0.05):
    raise NotImplementedError
