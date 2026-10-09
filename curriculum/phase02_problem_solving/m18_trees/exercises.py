"""m18 exercises: trees. Replace each `raise NotImplementedError` with your solution."""

from collections import deque


class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.value!r})"


def build(values):
    """Build a tree from a level-order list, with None for missing nodes.
    build([8, 3, 10, 1, 6, None, 14]) gives the tree drawn in the lesson (without 4 and 7)."""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        for side in ("left", "right"):
            if i < len(values) and values[i] is not None:
                child = TreeNode(values[i])
                setattr(node, side, child)
                queue.append(child)
            i += 1
    return root


# 1. Height: edges on the longest root-to-leaf path. Empty tree -> -1, one node -> 0.
def height(root):
    raise NotImplementedError


# 2. The three depth-first traversals, each returning a list of values.
def preorder(root):
    raise NotImplementedError


def inorder(root):
    raise NotImplementedError


def postorder(root):
    raise NotImplementedError


# 3. Level-order traversal: a list of levels, each a list of values left to right.
#    level_order(build([3, 9, 20, None, None, 15, 7])) -> [[3], [9, 20], [15, 7]]
def level_order(root):
    raise NotImplementedError


# 4. Insert a value into a BST (ignore duplicates) and return the root.
#    Then bst_contains returns whether the value is in the BST, in O(height).
def bst_insert(root, value):
    raise NotImplementedError


def bst_contains(root, value):
    raise NotImplementedError


# 5. Is this a valid BST? Every value in a node's left subtree must be smaller than
#    the node, and every value in its right subtree larger (not just the children!).
def is_valid_bst(root):
    raise NotImplementedError


# 6. In a BST, the lowest common ancestor of values a and b (both present):
#    the deepest node that has both in its subtree (a node counts as its own ancestor).
#    Return its value. Use the BST property: O(height).
def lca_bst(root, a, b):
    raise NotImplementedError


# 7. Diameter: the number of edges on the longest path between ANY two nodes
#    (the path doesn't have to pass through the root). O(n).
#    diameter(build([1, 2, 3, 4, 5])) -> 3   (4 → 2 → 1 → 3)
def diameter(root):
    raise NotImplementedError


# 8. Build a height-balanced BST from a SORTED list of distinct values
#    (the middle value becomes the root). Return the root.
def balanced_bst(sorted_values):
    raise NotImplementedError


# 9. Turn a tree into a string and back. Any format works as long as
#    deserialize(serialize(t)) rebuilds the same tree. Values are ints.
#    Hint: preorder with a marker like "#" for None, comma-separated.
def serialize(root):
    raise NotImplementedError


def deserialize(text):
    raise NotImplementedError


# 10. Run a decision tree classifier. Internal nodes are dicts
#     {"feature": name, "threshold": number, "left": subtree, "right": subtree}
#     and leaves are {"label": value}. Go left if sample[feature] <= threshold,
#     otherwise right. Return the label you reach.
def predict(tree, sample):
    raise NotImplementedError
