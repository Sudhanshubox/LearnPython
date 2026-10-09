"""m17 exercises: linked lists.

Work only with Node objects and their .next pointers: don't convert the linked
list into a Python list to solve the problem (except in the provided helpers).
"""


class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next

    def __repr__(self):
        return f"Node({self.value!r})"


def from_list(values):
    """Build a linked list from a Python list; returns the head (or None)."""
    head = None
    for v in reversed(values):
        head = Node(v, head)
    return head


def to_list(head):
    """Convert a linked list back to a Python list."""
    out = []
    while head is not None:
        out.append(head.value)
        head = head.next
    return out


# 1. Number of nodes. length(from_list([4, 5, 6])) -> 3, length(None) -> 0
def length(head):
    raise NotImplementedError


# 2. Return a new head with `value` inserted at position `index` (0 = front).
#    If index >= length, append at the end.
def insert_at(head, index, value):
    raise NotImplementedError


# 3. Remove every node whose value equals target. Return the (possibly new) head.
#    Hint: a dummy node in front of the head avoids special cases.
def remove_all(head, target):
    raise NotImplementedError


# 4. Reverse the list IN PLACE (change the .next pointers; don't create new nodes).
#    Return the new head.
def reverse(head):
    raise NotImplementedError


# 5. The middle node (for even length, the second of the two middles), using
#    fast and slow pointers in ONE pass. middle(from_list([1, 2, 3, 4])).value -> 3
def middle(head):
    raise NotImplementedError


# 6. Does the list contain a cycle? O(1) extra memory (no sets): Floyd's algorithm.
def has_cycle(head):
    raise NotImplementedError


# 7. Merge two sorted linked lists into one sorted list by relinking the existing
#    nodes. Return the head.
def merge_sorted(a, b):
    raise NotImplementedError


# 8. Remove the n-th node from the END in one pass (n is between 1 and the length).
#    remove_nth_from_end(from_list([1, 2, 3, 4, 5]), 2) -> 1 → 2 → 3 → 5
def remove_nth_from_end(head, n):
    raise NotImplementedError


# 9. An LRU (least recently used) cache with a fixed capacity. get() and put()
#    must both be O(1). When full, put() evicts the least recently used key.
#    Using get() or put() on a key makes it the most recently used.
#    You may use collections.OrderedDict OR build a doubly linked list + dict
#    yourself (better practice!).
class LRUCache:
    def __init__(self, capacity):
        raise NotImplementedError

    def get(self, key):
        """Return the value for key, or None if it isn't cached."""
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError
