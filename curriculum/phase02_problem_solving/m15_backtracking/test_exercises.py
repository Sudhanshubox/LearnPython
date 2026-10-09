import copy
import time

import pytest

import exercises
from code_checks import uses_any
from exercises import (
    combination_sum,
    combinations,
    n_queens,
    parentheses,
    solve_sudoku,
    subsets,
    subsets_with_dups,
    word_search,
)


def canon(lists):
    return sorted(sorted(x) for x in lists)


@pytest.mark.parametrize("name", [
    "subsets", "subsets_with_dups", "combinations", "combination_sum", "parentheses", "n_queens",
])
def test_no_itertools(name):
    assert not uses_any(getattr(exercises, name), "itertools", "permutations", "product")


def test_subsets():
    assert canon(subsets([1, 2])) == canon([[], [1], [2], [1, 2]])
    assert canon(subsets([])) == [[]]
    result = subsets([1, 2, 3, 4, 5])
    assert len(result) == 32
    assert len({tuple(sorted(s)) for s in result}) == 32


def test_subsets_with_dups():
    assert canon(subsets_with_dups([2, 1, 2])) == [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
    assert canon(subsets_with_dups([0])) == [[], [0]]
    assert len(subsets_with_dups([1, 1, 1, 1])) == 5


def test_combinations():
    assert sorted(combinations(4, 2)) == [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    assert combinations(3, 3) == [[1, 2, 3]]
    assert len(combinations(10, 3)) == 120
    assert all(c == sorted(c) for c in combinations(6, 4))


def test_combination_sum():
    assert canon(combination_sum([2, 3, 6, 7], 7)) == [[2, 2, 3], [7]]
    assert canon(combination_sum([2, 3, 5], 8)) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
    assert combination_sum([2], 1) == []


def test_combination_sum_is_pruned():
    start = time.perf_counter()
    result = combination_sum([3, 4, 5, 7, 9, 11, 13], 45)
    assert time.perf_counter() - start < 2.0, "prune branches whose sum is already too big"
    assert all(sum(c) == 45 for c in result)


def test_parentheses():
    assert sorted(parentheses(3)) == ["((()))", "(()())", "(())()", "()(())", "()()()"]
    assert parentheses(1) == ["()"]
    assert len(parentheses(6)) == 132


GRID = [list("ABCE"), list("SFCS"), list("ADEE")]


@pytest.mark.parametrize("word, expected", [
    ("ABCCED", True), ("SEE", True), ("ABCB", False), ("ASADFBCCEESE", True), ("Z", False), ("ABCESCFSADEE", True),
])
def test_word_search(word, expected):
    grid = copy.deepcopy(GRID)
    assert word_search(grid, word) == expected
    assert grid == GRID, "restore any cells you mark while searching"


@pytest.mark.parametrize("n, expected", [(1, 1), (2, 0), (3, 0), (4, 2), (6, 4), (8, 92)])
def test_n_queens(n, expected):
    assert n_queens(n) == expected


def test_n_queens_is_fast():
    start = time.perf_counter()
    assert n_queens(9) == 352
    assert time.perf_counter() - start < 1.0, "use sets for columns and both diagonals"


PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


def is_valid_solution(board, puzzle):
    digits = set(range(1, 10))
    rows = all(set(r) == digits for r in board)
    cols = all({board[r][c] for r in range(9)} == digits for c in range(9))
    boxes = all(
        {board[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)} == digits
        for br in (0, 3, 6) for bc in (0, 3, 6)
    )
    kept = all(puzzle[r][c] in (0, board[r][c]) for r in range(9) for c in range(9))
    return rows and cols and boxes and kept


def test_solve_sudoku():
    board = copy.deepcopy(PUZZLE)
    start = time.perf_counter()
    assert solve_sudoku(board) is True
    assert time.perf_counter() - start < 3.0
    assert is_valid_solution(board, PUZZLE)


def test_unsolvable_sudoku():
    puzzle = copy.deepcopy(PUZZLE)
    puzzle[0][2] = 5          # duplicate 5 in the top row: impossible
    puzzle[0][0] = 0
    puzzle[0][1] = 5
    board = copy.deepcopy(puzzle)
    assert solve_sudoku(board) is False
    assert board == puzzle
