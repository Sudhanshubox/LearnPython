"""m15 exercises: recursion and backtracking.

Don't use itertools: the point is to build the search yourself.
Results may be returned in any order unless the exercise says otherwise (the tests sort them).
"""


# 1. All subsets of a list of DISTINCT numbers (including [] and the full list).
#    subsets([1, 2]) -> [[], [1], [2], [1, 2]]  (any order)
def subsets(nums):
    raise NotImplementedError


# 2. All subsets of a list that MAY contain duplicates, without duplicate subsets.
#    Each subset sorted. subsets_with_dups([2, 1, 2]) -> [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
def subsets_with_dups(nums):
    raise NotImplementedError


# 3. All ways to choose k numbers from 1..n, each combination in increasing order.
#    combinations(4, 2) -> [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
def combinations(n, k):
    raise NotImplementedError


# 4. All combinations of `candidates` (distinct positive ints) that add up to `target`,
#    where each candidate may be used ANY number of times. Each combination sorted.
#    combination_sum([2, 3, 6, 7], 7) -> [[2, 2, 3], [7]]
#    Prune: stop exploring once the running sum is over target.
def combination_sum(candidates, target):
    raise NotImplementedError


# 5. All strings of n pairs of correctly matched parentheses.
#    parentheses(3) -> ["((()))", "(()())", "(())()", "()(())", "()()()"]
#    Hint: you may add "(" if you still have some left, and ")" only if it would
#    close an open one.
def parentheses(n):
    raise NotImplementedError


# 6. Can `word` be traced in the grid by moving up/down/left/right between
#    neighbouring cells, using each cell at most once?
#    grid = [list("ABCE"), list("SFCS"), list("ADEE")]
#    word_search(grid, "ABCCED") -> True, word_search(grid, "ABCB") -> False
def word_search(grid, word):
    raise NotImplementedError


# 7. N-Queens: count the ways to place n queens on an n x n board so that no two
#    attack each other (same row, column or diagonal). n_queens(8) -> 92
#    Must count n = 9 in well under a second: track used columns and diagonals in sets.
def n_queens(n):
    raise NotImplementedError


# 8. Sudoku solver. `board` is a list of 9 lists of 9 ints, 0 meaning empty.
#    Fill it IN PLACE so every row, column and 3x3 box contains 1-9, and return True;
#    return False (leaving the board as it was) if there's no solution, including
#    when the given digits already break the rules (check them first).
#    Speed-up: fill the empty cell with the fewest possible digits first.
def solve_sudoku(board):
    raise NotImplementedError
