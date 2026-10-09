"""m21 exercises: dynamic programming.

For each one, first write down (in a comment) the STATE, RECURRENCE and BASE CASES.
Your mentor can review your reasoning, not just your code.
"""


# 1. Ways to climb n stairs taking 1, 2 or 3 steps at a time.
#    climb(3) -> 4 (1+1+1, 1+2, 2+1, 3), climb(0) -> 1
def climb(n):
    raise NotImplementedError


# 2. House robber: houses in a row hold money; you can't rob two adjacent houses.
#    Return the maximum you can rob. rob([2, 7, 9, 3, 1]) -> 12 (2 + 9 + 1)
def rob(houses):
    raise NotImplementedError


# 3. Fewest coins that add up to `amount` (unlimited coins of each value), or -1.
#    coin_change([1, 5, 6], 10) -> 2,  coin_change([2], 3) -> -1
def coin_change(coins, amount):
    raise NotImplementedError


# 4. Number of different ways to make `amount` from the coins (order doesn't matter).
#    count_ways([1, 2, 5], 5) -> 4   (5, 2+2+1, 2+1+1+1, 1+1+1+1+1)
def count_ways(coins, amount):
    raise NotImplementedError


# 5. Length of the longest strictly increasing subsequence (not necessarily contiguous).
#    lis([10, 9, 2, 5, 3, 7, 101, 18]) -> 4   ([2, 3, 7, 101])
#    O(n²) DP passes; an O(n log n) version exists (think binary search, m16).
def lis(nums):
    raise NotImplementedError


# 6. Edit distance (Levenshtein): fewest insertions, deletions and substitutions
#    to turn a into b. edit_distance("kitten", "sitting") -> 3
def edit_distance(a, b):
    raise NotImplementedError


# 7. Longest common subsequence: return the subsequence ITSELF (a string).
#    If several have the maximum length, any one is accepted.
#    lcs("ABCBDAB", "BDCABA") -> a string of length 4, e.g. "BCBA"
def lcs(a, b):
    raise NotImplementedError


# 8. 0/1 knapsack: items are (weight, value); each can be taken at most once.
#    Return the maximum total value with total weight <= capacity.
#    knapsack([(1, 1), (3, 4), (4, 5), (5, 7)], 7) -> 9   (weights 3 + 4)
def knapsack(items, capacity):
    raise NotImplementedError


# 9. Minimum path sum in a grid of non-negative numbers, moving only right or down
#    from the top-left to the bottom-right cell (both included).
#    min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) -> 7   (1 → 3 → 1 → 1 → 1)
def min_path_sum(grid):
    raise NotImplementedError


# 10. Word break: can `text` be split into a sequence of words from `vocab`?
#     word_break("applepenapple", ["apple", "pen"]) -> True
#     (Tokenizers solve a close cousin of this problem.)
def word_break(text, vocab):
    raise NotImplementedError
