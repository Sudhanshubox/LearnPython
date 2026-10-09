# m21 · Dynamic programming

**By the end you can:** recognize problems with overlapping subproblems, define a DP state and recurrence, implement it top-down (memoization) or bottom-up (a table), reconstruct the actual solution, and reduce memory.

**Why it matters for AI:** dynamic programming is everywhere in ML. **Edit distance** measures speech-recognition and OCR error rates (WER/CER). The **Viterbi** algorithm decodes hidden Markov models and CRFs. **CTC loss**, used to train speech models, is a DP over alignments. Reinforcement learning is built on the **Bellman equation**, which *is* dynamic programming. Even sequence alignment of DNA and protein (AlphaFold's input pipeline) is DP.

---

## 1. The idea

DP = recursion + remembering answers (m07's memoized Fibonacci). It works when:

1. **Optimal substructure:** the best answer for a problem can be built from best answers to smaller subproblems.
2. **Overlapping subproblems:** the same smaller subproblems come up again and again.

Plain recursion recomputes them exponentially many times; DP computes each **once**.

## 2. The recipe

1. **State:** what *exactly* does `dp[i]` (or `dp[i][j]`) mean? Write it as a sentence. This is the hardest and most important step.
2. **Recurrence:** how is `dp[i]` computed from smaller states?
3. **Base cases:** the smallest states, answered directly.
4. **Order:** compute states so everything a state needs is ready before it.
5. **Answer:** which state holds the final answer?

## 3. Example: climbing stairs

You can climb 1 or 2 steps at a time. How many ways to reach step n?

- **State:** `ways[i]` = number of ways to reach step i.
- **Recurrence:** the last move was either 1 step (from i−1) or 2 steps (from i−2): `ways[i] = ways[i-1] + ways[i-2]`.
- **Base cases:** `ways[0] = 1` (standing still), `ways[1] = 1`.

**Top-down** (memoized recursion, natural to write):

```python
from functools import cache

@cache
def ways(i):
    if i <= 1:
        return 1
    return ways(i - 1) + ways(i - 2)
```

**Bottom-up** (a table filled in order, no recursion limit):

```python
def ways(n):
    dp = [1] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

Each state only needs the previous two, so you can keep two variables instead of a table: O(1) memory.

## 4. Example: coin change (fewest coins)

Coins `[1, 5, 6]`, amount 10. Greedy (take the biggest coin first) gives 6+1+1+1+1 = 5 coins; the best is 5+5 = 2 coins. Greedy fails, DP doesn't:

- **State:** `dp[a]` = fewest coins that make amount a.
- **Recurrence:** `dp[a] = 1 + min(dp[a - c] for each coin c ≤ a)`.
- **Base:** `dp[0] = 0`.

## 5. Two-dimensional DP: edit distance

The fewest single-character insertions, deletions or substitutions to turn word `a` into word `b` ("kitten" → "sitting" = 3).

- **State:** `dp[i][j]` = edit distance between the first i characters of `a` and the first j characters of `b`.
- **Recurrence:** if `a[i-1] == b[j-1]`, then `dp[i][j] = dp[i-1][j-1]`. Otherwise, 1 + the minimum of:
  - `dp[i-1][j]` (delete from a),
  - `dp[i][j-1]` (insert into a),
  - `dp[i-1][j-1]` (substitute).
- **Base:** `dp[i][0] = i`, `dp[0][j] = j`.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Draw the table for small examples. It makes 2-D DP much easier to reason about.

## 6. Reconstructing the solution

The table gives the *value* of the best answer. To get the answer itself (which coins? which subsequence?), either store the choice made at each state, or walk back through the table from the final state, re-checking which option produced each value. Exercise 7 asks for this.

## 7. Common DP families

| Family | State looks like | Examples |
|---|---|---|
| 1-D sequence | `dp[i]` = best for the first i items | stairs, house robber, LIS |
| two sequences | `dp[i][j]` for prefixes of both | edit distance, LCS |
| knapsack | `dp[i][capacity]` | 0/1 knapsack, subset sum |
| grid | `dp[r][c]` | unique paths, min path sum |
| intervals | `dp[i][j]` for a range | matrix-chain, palindromes |

---

## Problem-solving habit #20: start from the brute-force recursion

Don't try to invent the table directly. Write the plain recursive solution first ("try every choice for the first step, recurse on the rest"). Then notice the repeated arguments, add memoization, and you have DP. Converting to bottom-up is a mechanical last step.

## Go deeper (optional, research-level)

1. Speech recognition is scored by **word error rate**: edit distance between word sequences, divided by the number of reference words. Implement WER using your edit distance on lists of words, and compute it for "the cat sat on the mat" vs "the cat sat on mat".
2. Read about the **Viterbi algorithm** for HMMs. Write its recurrence as a DP state and transition. How is it like your coin change, with `max`/`+` replaced by `max`/`×` of probabilities?
3. The Bellman equation in reinforcement learning, V(s) = maxₐ [r(s, a) + γ V(s')], defines a DP over states. Read chapter 4 ("Dynamic Programming") of Sutton & Barto's *Reinforcement Learning: An Introduction* (free online).

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Each test checks speed too, so plain exponential recursion won't pass.
