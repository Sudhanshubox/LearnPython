# m18 · Trees and binary search trees

**By the end you can:** represent trees, traverse them recursively and level by level, use and validate binary search trees, and solve tree problems by thinking about "what does each subtree return?"

**Why it matters for AI:** decision trees, random forests and gradient-boosted trees (XGBoost, LightGBM) still win most competitions on tabular data. Parsers turn code and sentences into trees. Hierarchical clustering produces trees. Game-playing and planning algorithms search trees. In exercise 10 you'll run a decision tree classifier by hand.

---

## 1. Vocabulary

```
            8          ← root
          /   \
         3     10      ← 3 is the parent of 1 and 6
        / \      \
       1   6      14   ← nodes with no children are leaves
          / \
         4   7
```

- **Binary tree:** each node has at most two children, `left` and `right`.
- **Depth** of a node: edges from the root. **Height** of a tree: the depth of its deepest node (here 3; the tree with a single node has height 0, and an empty tree has height -1).
- Every node is the root of its own **subtree**. This is why trees and recursion fit together perfectly.

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
```

## 2. Thinking recursively about trees

Almost every tree function has the same shape:

```python
def solve(node):
    if node is None:
        return <answer for an empty tree>
    left = solve(node.left)
    right = solve(node.right)
    return <combine node.value, left, right>
```

```python
def size(node):
    if node is None:
        return 0
    return 1 + size(node.left) + size(node.right)

def height(node):
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))
```

Ask: *"if I already had the answers for the left and right subtrees, how would I get the answer for this node?"*

## 3. Depth-first traversals

The three orders differ only in *when* you visit the node itself:

| Order | Visit | Example (tree above) | Typical use |
|---|---|---|---|
| preorder | node, left, right | 8 3 1 6 4 7 10 14 | copying / serializing a tree |
| inorder | left, node, right | 1 3 4 6 7 8 10 14 | BST → sorted order |
| postorder | left, right, node | 1 4 7 6 3 14 10 8 | deleting, evaluating expression trees, backprop |

## 4. Breadth-first (level order)

Visit level by level using a queue (m14):

```python
from collections import deque

def level_order(root):
    if root is None:
        return []
    levels, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):        # exactly the nodes of this level
            node = queue.popleft()
            level.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        levels.append(level)
    return levels
```

## 5. Binary search trees

A **BST** keeps everything in a node's left subtree smaller than it, and everything in the right subtree larger. Search is binary search on a tree:

```python
def contains(node, target):
    while node:
        if target == node.value:
            return True
        node = node.left if target < node.value else node.right
    return False
```

This takes O(height) steps. A **balanced** tree has height about log₂ n, so search, insert and delete are O(log n). But inserting sorted data (1, 2, 3, …) makes a "tree" that's really a linked list: height n, O(n) per operation. Self-balancing trees (AVL, red-black) fix this by rotating nodes. Databases use a cousin, the B-tree.

**Classic bug:** checking only `node.left.value < node.value < node.right.value` is *not* enough to validate a BST. Every node in the left subtree must be smaller, not just the direct child. Pass down the allowed (low, high) range instead.

## 6. Decision trees: a tree that makes predictions

```
               petal_length <= 2.45 ?
                /                 \
             yes                   no
          "setosa"         petal_width <= 1.75 ?
                              /            \
                        "versicolor"    "virginica"
```

To classify a flower, start at the root, answer each question, and follow the branch until you reach a leaf. Training a decision tree (Phase 5) means choosing the questions; using one is just walking down a tree.

---

## Problem-solving habit #17: define what the recursion returns

For harder tree problems (like the diameter, exercise 7), the answer you want isn't what each subtree should return. Define a helper that returns something easier (here, the height), and update the real answer on the side as you go. Getting this split right is the key insight in most hard tree problems.

## Go deeper (optional, research-level)

1. How many distinct BST shapes can hold the values 1..n? Compute the numbers for n = 1..6 and look up the **Catalan numbers**. (They also count balanced parenthesis strings from m15!)
2. Read about **B-trees**. Why do databases and file systems use wide trees with hundreds of children per node instead of binary trees? (Think about disk reads.)
3. Gradient-boosted trees often beat neural networks on tabular data. Read the abstract of *Why do tree-based models still outperform deep learning on tabular data?* (Grinsztajn et al., 2022). What properties of tabular data do they blame?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. `TreeNode` and a `build` helper (from a level-order list with `None` for gaps) are provided.
