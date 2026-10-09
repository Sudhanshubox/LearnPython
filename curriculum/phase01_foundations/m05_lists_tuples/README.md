# m05 · Lists and tuples

**By the end you can:** store and transform sequences of data, write list comprehensions, avoid the aliasing and copying bugs, and know which list operations are fast or slow.

**Why it matters for AI:** a dataset is a sequence of examples, a batch is a slice of it, and a matrix is a list of rows. Before NumPy and PyTorch do this for you at high speed, you'll build mini-batching, matrix multiplication and moving averages by hand so you know exactly what those libraries do.

---

## 1. Lists: ordered and changeable

```python
nums = [3, 1, 4, 1, 5]
mixed = [1, "two", 3.0, [4]]    # any types, even other lists
empty = []

nums[0]        # 3
nums[-1]       # 5
nums[1:3]      # [1, 4]
len(nums)      # 5
4 in nums      # True
nums[0] = 10   # lists are mutable: [10, 1, 4, 1, 5]
```

## 2. Changing a list

```python
nums.append(9)        # add one item at the end
nums.extend([2, 6])   # add several items
nums.insert(0, 7)     # insert at a position
nums.pop()            # remove and return the last item
nums.pop(0)           # remove and return the item at index 0
nums.remove(1)        # remove the first 1 (ValueError if absent)
del nums[1:3]         # delete a slice
nums.index(4)         # position of the first 4
nums.count(1)         # how many 1s
nums.reverse()        # reverse in place
```

## 3. Sorting

```python
nums.sort()                     # sorts IN PLACE, returns None
new = sorted(nums)              # returns a NEW sorted list, original unchanged
sorted(nums, reverse=True)

words = ["banana", "Apple", "cherry"]
sorted(words, key=str.lower)    # sort by a computed key: ['Apple', 'banana', 'cherry']
sorted(words, key=len)          # by length

students = [("Asha", 91), ("Ben", 78), ("Chen", 91)]
sorted(students, key=lambda s: (-s[1], s[0]))   # score high→low, then name A→Z
```

**Classic bug:** `nums = nums.sort()` sets `nums` to `None`.

## 4. List comprehensions

A compact way to build a list from another iterable:

```python
squares = [x * x for x in range(10)]
evens = [x for x in nums if x % 2 == 0]
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]
labels = ["pos" if s > 0 else "neg" for s in scores]
```

Read them as "**give me** `x * x` **for each** `x` **in** `range(10)`". Use them for simple transformations; use a normal loop when the logic needs several steps.

## 5. Tuples: fixed sequences

```python
point = (3, 4)
single = (5,)          # the comma makes the tuple, not the parentheses
x, y = point           # unpacking
a, b = b, a            # swap without a temporary variable
first, *rest = [1, 2, 3, 4]   # first = 1, rest = [2, 3, 4]
```

Tuples are **immutable**. Use them for records with a fixed shape (a point, an RGB colour, a database row) and for returning several values from a function.

## 6. Aliasing and copying

Remember from m01: assignment never copies.

```python
a = [1, 2, 3]
b = a            # same list, two names
c = a[:]         # shallow copy (also a.copy() or list(a))

grid = [[0] * 3] * 3      # BUG: three references to the SAME row
grid[0][0] = 1
print(grid)               # [[1, 0, 0], [1, 0, 0], [1, 0, 0]]

grid = [[0] * 3 for _ in range(3)]   # correct: three separate rows
```

A **shallow copy** copies the outer list, but inner lists are still shared. For nested data, use `copy.deepcopy()`.

## 7. What's fast and what's slow

| Operation | Cost | Why |
|---|---|---|
| `lst[i]`, `lst[i] = x`, `len(lst)` | O(1) | a list is an array of pointers |
| `lst.append(x)`, `lst.pop()` | O(1) on average | extra space is reserved at the end |
| `lst.insert(0, x)`, `lst.pop(0)` | O(n) | every item after it must shift |
| `x in lst`, `lst.index(x)` | O(n) | checks items one by one |
| `lst.sort()` | O(n log n) | Timsort |

**Tip:** if you need fast `in` checks, use a set (next module). If you need to add/remove at both ends, use `collections.deque`.

---

## Problem-solving habit #5: two pointers

Many list problems become easy with two indices moving through the data, for example one at each end moving inward, or one in each of two sorted lists. Exercise 9 (merging sorted lists) is the classic example, and it's the heart of merge sort.

## Go deeper (optional, research-level)

1. `lst.append` is "amortized O(1)". Measure `sys.getsizeof(lst)` as you append 100 items, one at a time. When does the size jump, and by how much? Find the growth formula in CPython's `listobject.c`.
2. Your `matmul` does about n³ multiplications for two n×n matrices. NumPy can be 100–1000× faster for the same work. List three reasons (hint: memory layout, C loops, SIMD/BLAS).
3. Why is Timsort stable, and why does stability matter when you sort by several keys one after another?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
