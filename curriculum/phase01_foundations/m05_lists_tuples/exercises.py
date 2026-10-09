"""m05 exercises: lists and tuples. Replace each `raise NotImplementedError` with your solution.

None of these functions should change the lists passed in: return new lists.
"""


# 1. Return the second-largest DISTINCT value, or None if there isn't one.
#    second_largest([3, 1, 4, 4, 2]) -> 3, second_largest([5, 5]) -> None
def second_largest(nums):
    raise NotImplementedError


# 2. Remove duplicates but keep the first occurrence of each item, in order.
#    dedupe([3, 1, 3, 2, 1]) -> [3, 1, 2]
def dedupe(items):
    raise NotImplementedError


# 3. Rotate a list right by k places. k can be larger than the list or negative.
#    rotate([1, 2, 3, 4, 5], 2) -> [4, 5, 1, 2, 3]
#    rotate([1, 2, 3], -1) -> [2, 3, 1]
def rotate(items, k):
    raise NotImplementedError


# 4. Split items into batches of `size`. The last batch may be smaller.
#    batches([1, 2, 3, 4, 5], 2) -> [[1, 2], [3, 4], [5]]
#    (This is exactly how training data is split into mini-batches.)
def batches(items, size):
    raise NotImplementedError


# 5. Transpose a matrix (list of rows): rows become columns.
#    transpose([[1, 2, 3], [4, 5, 6]]) -> [[1, 4], [2, 5], [3, 6]]
def transpose(matrix):
    raise NotImplementedError


# 6. Multiply two matrices using loops (no libraries).
#    a is n x m, b is m x p, and the result is n x p, where
#    result[i][j] = sum of a[i][k] * b[k][j] over k.
#    Return None if the shapes don't match (a's column count != b's row count).
def matmul(a, b):
    raise NotImplementedError


# 7. Moving average with a window of size `window`: the average of each run of
#    `window` consecutive values. Return [] if there are fewer values than the window.
#    moving_average([1, 2, 3, 4, 5], 3) -> [2.0, 3.0, 4.0]
#    Challenge: make it O(n) by updating a running sum instead of re-adding each window.
def moving_average(values, window):
    raise NotImplementedError


# 8. Sort (name, score) pairs by score from high to low; break ties by name A-Z.
#    rank([("Ben", 78), ("Asha", 91), ("Chen", 91)])
#    -> [("Asha", 91), ("Chen", 91), ("Ben", 78)]
def rank(results):
    raise NotImplementedError


# 9. Merge two already-sorted lists into one sorted list in O(len(a) + len(b)),
#    using two pointers. Don't call sort() or sorted().
#    merge_sorted([1, 4, 9], [2, 3, 10]) -> [1, 2, 3, 4, 9, 10]
def merge_sorted(a, b):
    raise NotImplementedError
