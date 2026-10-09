"""Helpers for tests that check HOW an exercise was solved, not just the answer.

They read the function's syntax tree, so comments and strings never count.
"""

import ast
import inspect
import textwrap


def _tree(func):
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def names_used(func):
    """Every variable, function and attribute name the function's code refers to."""
    names = set()
    for node in ast.walk(_tree(func)):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def uses_any(func, *names):
    return bool(names_used(func) & set(names))


def contains(func, node_type):
    """True if the function contains a given kind of syntax, e.g. ast.While or ast.Match."""
    return any(isinstance(node, node_type) for node in ast.walk(_tree(func)))


def is_recursive(func):
    """True if the function calls itself."""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func.__name__
        for node in ast.walk(_tree(func))
    )


def uses_power_half(func):
    """True if the code computes something like x ** 0.5."""
    return any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
        and isinstance(node.right, ast.Constant) and node.right.value == 0.5
        for node in ast.walk(_tree(func))
    )
