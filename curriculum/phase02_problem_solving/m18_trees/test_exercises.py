import random

import pytest

from exercises import (
    TreeNode,
    balanced_bst,
    bst_contains,
    bst_insert,
    build,
    deserialize,
    diameter,
    height,
    inorder,
    is_valid_bst,
    lca_bst,
    level_order,
    postorder,
    predict,
    preorder,
    serialize,
)

rng = random.Random(18)
LESSON_TREE = [8, 3, 10, 1, 6, None, 14, None, None, 4, 7]


def same_tree(a, b):
    if a is None or b is None:
        return a is b
    return a.value == b.value and same_tree(a.left, b.left) and same_tree(a.right, b.right)


def random_tree(n):
    root = None
    for v in rng.sample(range(-50, 50), n):
        root = bst_insert(root, v)
    return root


@pytest.mark.parametrize("values, expected", [([], -1), ([1], 0), ([1, 2], 1), (LESSON_TREE, 3), ([1, 2, None, 3, None, 4], 3)])
def test_height(values, expected):
    assert height(build(values)) == expected


def test_traversals():
    root = build(LESSON_TREE)
    assert preorder(root) == [8, 3, 1, 6, 4, 7, 10, 14]
    assert inorder(root) == [1, 3, 4, 6, 7, 8, 10, 14]
    assert postorder(root) == [1, 4, 7, 6, 3, 14, 10, 8]
    assert preorder(None) == inorder(None) == postorder(None) == []


def test_level_order():
    assert level_order(build([3, 9, 20, None, None, 15, 7])) == [[3], [9, 20], [15, 7]]
    assert level_order(build(LESSON_TREE)) == [[8], [3, 10], [1, 6, 14], [4, 7]]
    assert level_order(None) == []


def test_bst_insert_and_contains():
    root = None
    values = [50, 30, 70, 20, 40, 60, 80, 30]
    for v in values:
        root = bst_insert(root, v)
    assert inorder(root) == [20, 30, 40, 50, 60, 70, 80]
    assert bst_contains(root, 60)
    assert not bst_contains(root, 65)
    assert not bst_contains(None, 1)


def test_bst_random():
    for _ in range(20):
        values = rng.sample(range(1000), 50)
        root = None
        for v in values:
            root = bst_insert(root, v)
        assert inorder(root) == sorted(values)
        assert all(bst_contains(root, v) for v in values)


def test_is_valid_bst():
    assert is_valid_bst(build(LESSON_TREE))
    assert is_valid_bst(None)
    assert not is_valid_bst(build([5, 1, 4, None, None, 3, 6]))
    # 6 is a right child of 3 but sits in 5's LEFT subtree: invalid even though each parent/child pair looks fine
    assert not is_valid_bst(TreeNode(5, TreeNode(3, None, TreeNode(6)), TreeNode(8)))
    assert not is_valid_bst(TreeNode(2, TreeNode(2)))


@pytest.mark.parametrize("a, b, expected", [(1, 7, 3), (4, 7, 6), (14, 4, 8), (3, 4, 3), (10, 14, 10)])
def test_lca_bst(a, b, expected):
    assert lca_bst(build(LESSON_TREE), a, b) == expected


@pytest.mark.parametrize("values, expected", [
    ([1, 2, 3, 4, 5], 3), ([1], 0), ([], 0), ([1, 2], 1),
    ([1, 2, None, 3, 4, None, None, 5, None, None, 6, 7, None, None, 8], 6),
])
def test_diameter(values, expected):
    assert diameter(build(values)) == expected


def test_balanced_bst():
    for n in [0, 1, 2, 7, 8, 100, 1000]:
        values = list(range(n))
        root = balanced_bst(values)
        assert inorder(root) == values
        assert height(root) == (n.bit_length() - 1 if n else -1)


def test_serialize_round_trip():
    for values in [[], [1], LESSON_TREE, [1, None, 2, None, 3], [-5, 10, -20]]:
        tree = build(values)
        assert same_tree(deserialize(serialize(tree)), tree)
    for _ in range(20):
        tree = random_tree(rng.randint(0, 30))
        text = serialize(tree)
        assert isinstance(text, str)
        assert same_tree(deserialize(text), tree)


IRIS_TREE = {
    "feature": "petal_length", "threshold": 2.45,
    "left": {"label": "setosa"},
    "right": {
        "feature": "petal_width", "threshold": 1.75,
        "left": {"label": "versicolor"},
        "right": {"label": "virginica"},
    },
}


@pytest.mark.parametrize("sample, expected", [
    ({"petal_length": 1.4, "petal_width": 0.2}, "setosa"),
    ({"petal_length": 2.45, "petal_width": 3.0}, "setosa"),
    ({"petal_length": 4.5, "petal_width": 1.5}, "versicolor"),
    ({"petal_length": 5.8, "petal_width": 2.2}, "virginica"),
])
def test_predict(sample, expected):
    assert predict(IRIS_TREE, sample) == expected


def test_predict_leaf_only():
    assert predict({"label": 1}, {}) == 1
