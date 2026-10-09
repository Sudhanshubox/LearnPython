"""m14 exercises: stacks and queues. Replace each `raise NotImplementedError` with your solution."""

from collections import deque


# 1. Are all brackets (), [], {} balanced and properly nested? Ignore other characters.
#    balanced("f(a[1]) {ok}") -> True, balanced("(]") -> False, balanced("((") -> False
def balanced(text):
    raise NotImplementedError


# 2. Simplify a Unix-style absolute path.
#    "." means the current folder, ".." the parent (staying at "/" if already at root),
#    and repeated slashes count as one. The result starts with "/" and has no trailing "/".
#    simplify_path("/a/./b/../../c/") -> "/c"
#    simplify_path("/../") -> "/"
#    simplify_path("/home//user/") -> "/home/user"
def simplify_path(path):
    raise NotImplementedError


# 3. Evaluate an expression in Reverse Polish Notation. Tokens are integers or
#    + - * /. Division truncates toward zero (use int(a / b)).
#    eval_rpn(["2", "1", "+", "3", "*"]) -> 9
#    eval_rpn(["4", "13", "5", "/", "+"]) -> 6
def eval_rpn(tokens):
    raise NotImplementedError


# 4. Shunting-yard: convert an infix expression to RPN tokens.
#    Input tokens: integers, + - * /, and parentheses. * and / bind tighter than + and -.
#    All operators are left-associative.
#    to_rpn(["3", "+", "4", "*", "2"]) -> ["3", "4", "2", "*", "+"]
#    to_rpn(["(", "1", "+", "2", ")", "*", "3"]) -> ["1", "2", "+", "3", "*"]
#    Then calculate(expression) should work for strings like "2 * (3 + 4) - 5"
#    (tokens separated by spaces): tokenize with split(), to_rpn, eval_rpn.
def to_rpn(tokens):
    raise NotImplementedError


def calculate(expression):
    raise NotImplementedError


# 5. For each day, how many days until a warmer temperature? 0 if never. O(n).
#    days_until_warmer([73, 74, 75, 71, 69, 72, 76, 73]) -> [1, 1, 4, 2, 1, 1, 0, 0]
def days_until_warmer(temps):
    raise NotImplementedError


# 6. A stack that also returns its minimum in O(1). Every method must be O(1).
class MinStack:
    def __init__(self):
        raise NotImplementedError

    def push(self, value):
        raise NotImplementedError

    def pop(self):
        """Remove and return the top value."""
        raise NotImplementedError

    def top(self):
        raise NotImplementedError

    def get_min(self):
        raise NotImplementedError


# 7. A FIFO queue built from TWO lists used only as stacks (append / pop() at the end).
#    No deque, no pop(0), no insert(0, ...). enqueue is O(1); dequeue is O(1) amortized.
class TwoStackQueue:
    def __init__(self):
        raise NotImplementedError

    def enqueue(self, value):
        raise NotImplementedError

    def dequeue(self):
        """Remove and return the oldest value."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# 8. Hot potato: n players (names) stand in a circle. Starting from the first, pass the
#    potato `k` times; whoever holds it is eliminated. Continue from the next player.
#    Return the names in the order they're eliminated. Use a deque.
#    hot_potato(["A", "B", "C", "D"], 1) -> ["B", "D", "C", "A"]
def hot_potato(names, k):
    raise NotImplementedError
