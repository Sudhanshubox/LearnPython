# m17 · Linked lists

**By the end you can:** build a linked list from scratch with classes, manipulate pointers confidently (reverse, merge, detect cycles), and explain when a linked structure beats an array.

**Why it matters for AI:** you'll rarely use a linked list directly in ML code, but the skill it trains, **reasoning about references between objects**, is exactly what you need for computation graphs (how PyTorch tracks operations for backpropagation), trees, graphs and caches. An LRU cache, used for KV-caches and memoizing expensive model calls, is a linked list plus a dict.

---

## 1. Nodes and links

A linked list is a chain of **nodes**. Each node holds a value and a reference to the next node:

```
head → [3 | •] → [7 | •] → [1 | None]
```

In Python:

```python
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

head = Node(3, Node(7, Node(1)))
head.value            # 3
head.next.value       # 7
head.next.next.next   # None: the end
```

`self` is the object being created or used. `self.value = value` stores the value *on that object*. (Phase 3 covers classes in depth.)

## 2. Walking the list

```python
def to_list(head):
    out = []
    node = head
    while node is not None:
        out.append(node.value)
        node = node.next          # move along the chain
    return out
```

Almost every linked-list function is this loop plus some pointer changes.

## 3. Arrays vs linked lists

| Operation | Python list (array) | Linked list |
|---|---|---|
| access item i | O(1) | O(i): walk from the head |
| insert/delete at the front | O(n): shift everything | O(1): change one pointer |
| insert/delete after a known node | O(n) | O(1) |
| memory | compact, cache-friendly | one object per item, scattered |

In Python, lists win almost always because they're compact and implemented in C. Linked structures win when you insert/delete in the middle a lot *and* already hold a reference to the spot, as in an LRU cache.

## 4. Drawing pointer changes

Inserting `5` after the node `a`:

```
before:   a → b
step 1:   new = Node(5, a.next)      new → b
step 2:   a.next = new               a → new → b
```

The order matters: if you did step 2 first, you'd lose your only reference to `b`. **Always draw the boxes and arrows before writing pointer code.**

## 5. The dummy head trick

Functions that might change the *first* node (deleting it, merging two lists) need special cases for the head. A **dummy** node in front removes them:

```python
def remove_all(head, target):
    dummy = Node(None, head)
    node = dummy
    while node.next:
        if node.next.value == target:
            node.next = node.next.next     # skip it
        else:
            node = node.next
    return dummy.next                       # the real head, possibly changed
```

## 6. Fast and slow pointers

Move one pointer two steps at a time and another one step:

- When `fast` reaches the end, `slow` is at the **middle**.
- If there's a **cycle**, `fast` eventually laps `slow` and they meet (Floyd's algorithm). O(n) time, O(1) memory.

## 7. Reversing a list

The classic interview question. Walk the list, flipping each arrow to point backwards. You need three references: `prev`, `current` and `next` (saved before you overwrite `current.next`).

---

## Problem-solving habit #16: check the edges of a data structure

For every linked-list function, test: the empty list (`None`), a single node, two nodes, and the operation at the head and at the tail. Most pointer bugs only appear at these edges.

## Go deeper (optional, research-level)

1. PyTorch builds a graph of `grad_fn` nodes as you compute. Run `y = (x * 2).sum()` with `x = torch.ones(3, requires_grad=True)` (in Phase 6) and follow `y.grad_fn.next_functions`. It's a linked structure pointing backwards through your computation.
2. Python's `collections.OrderedDict` was historically implemented with a doubly linked list. Read why it was needed and what changed when normal dicts became ordered in Python 3.7.
3. Prove Floyd's cycle detection: why must the fast pointer meet the slow one inside the cycle, and how can you then find where the cycle *starts*?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. The `Node` class and the `from_list` / `to_list` helpers are written for you.
