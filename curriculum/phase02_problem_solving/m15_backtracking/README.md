# m15 · Recursion and backtracking

**By the end you can:** systematically generate every subset, permutation or combination, solve constraint puzzles (N-Queens, Sudoku) by backtracking, and prune the search so it finishes in reasonable time.

**Why it matters for AI:** search is one of the oldest ideas in AI. Game-playing programs (from chess engines to AlphaGo's tree search), planners, constraint solvers and theorem provers all explore a tree of choices and back up from dead ends. Modern LLM "reasoning" methods like tree-of-thought search are the same idea with a neural network scoring the branches.

---

## 1. The decision tree

Many problems are a sequence of **choices**. Generating all subsets of `[1, 2, 3]` means deciding, for each number, *include it or not*:

```
                     []
            /                   \
       take 1                skip 1
        [1]                    []
      /     \               /      \
   [1,2]    [1]           [2]      []
   /  \     /  \         /  \     /  \
[1,2,3][1,2][1,3][1]  [2,3] [2]  [3]  []
```

Every leaf is one answer. **Backtracking** walks this tree depth-first: make a choice, explore everything below it, **undo** the choice, then try the next one.

## 2. The template

```python
def backtrack(state, choices):
    if is_complete(state):
        results.append(copy_of(state))
        return
    for choice in choices_available(state):
        if not is_valid(state, choice):
            continue                 # prune: skip branches that can't work
        make(choice, state)          # choose
        backtrack(state, choices)    # explore
        undo(choice, state)          # un-choose
```

Concrete version for subsets:

```python
def subsets(nums):
    results, current = [], []

    def explore(i):
        if i == len(nums):
            results.append(current[:])     # copy! current keeps changing
            return
        current.append(nums[i])            # choose: take nums[i]
        explore(i + 1)
        current.pop()                      # un-choose
        explore(i + 1)                     # skip nums[i]

    explore(0)
    return results
```

**Classic bug:** `results.append(current)` without copying. Every entry ends up pointing to the same (finally empty) list. (Aliasing again: m01, m05.)

## 3. How big is the tree?

| Problem | Number of answers |
|---|---|
| subsets of n items | 2ⁿ |
| permutations of n items | n! |
| combinations, choose k of n | n! / (k!(n−k)!) |

20 items → 2²⁰ ≈ 1 million subsets (fine). 20! ≈ 2.4×10¹⁸ permutations (impossible). Backtracking is for small n, or for problems where **pruning** cuts the tree down drastically.

## 4. Pruning

Pruning means detecting early that a branch can't lead to a valid answer and not exploring it at all.

- **Combination sum:** if the numbers are sorted and the next one already overshoots the target, every later one will too: `break`.
- **N-Queens:** don't place a queen on a square attacked by an earlier queen. Keep sets of used columns and diagonals for O(1) checks.
- **Sudoku:** only try digits not already in the row, column or box. Try the cell with the *fewest* options first.

Good pruning can make an "impossible" exponential search finish in milliseconds.

## 5. Avoiding duplicate answers

If the input has duplicates (`[1, 2, 2]`), naive generation repeats answers. Sort first, then at each level skip a choice equal to the previous choice *at the same level*:

```python
for i in range(start, len(nums)):
    if i > start and nums[i] == nums[i - 1]:
        continue
```

## 6. Recursion depth and the call stack

Each level of recursion is a frame on the call stack (m07, m14). Backtracking depth equals the number of decisions (n), which is usually small. But for a recursion 10,000 levels deep you'd hit Python's `RecursionError`; rewrite with an explicit stack.

---

## Problem-solving habit #14: draw the tree for n = 3

Before coding a backtracking solution, draw the decision tree for a tiny input. What's the choice at each level? What makes a leaf? Where can you cut a branch? The code follows directly from the drawing.

## Go deeper (optional, research-level)

1. Sudoku is a **constraint satisfaction problem**. Read about *constraint propagation* (Peter Norvig's essay "Solving Every Sudoku Puzzle"). How much faster is it than plain backtracking on the hardest puzzles?
2. AlphaGo used **Monte Carlo Tree Search** instead of exhaustive search. Read about UCB1 / UCT: how does it decide which branch deserves more exploration?
3. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., 2023) applies BFS/DFS with backtracking over an LLM's intermediate "thoughts". Which parts of your template correspond to the LLM, and which to the search algorithm?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
