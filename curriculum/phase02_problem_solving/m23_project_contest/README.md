# m23 · Project: problem-solving contest

**Phase 2 finale.** Ten problems, in no particular order of technique. Unlike the earlier modules, nothing tells you *which* idea to use. Recognizing the right technique is the real skill that interviews, competitions and research all test.

Everything you need is in m12–m22: hashing, two pointers, sliding windows, stacks, binary search, heaps, graphs, DP, greedy.

---

## Contest rules (for yourself)

1. **Time yourself.** Aim for 25–40 minutes per problem. If you're stuck after 40 minutes, ask your mentor for a *hint*, not a solution.
2. **Use the framework from m12** on every problem: understand → examples → brute force → optimize → code → test → analyze.
3. **Write your reasoning** as a comment above each solution: the technique, *why* it applies, and the time/space complexity. Ask the mentor to review the comment as well as the code.
4. **Keep a technique journal.** After each problem, add one line to a file of your own: "Problem → clue in the statement → technique". After 50 problems this journal is worth more than any book.

## Problems

| # | Problem | Difficulty |
|---|---|---|
| 1 | Longest consecutive run | ★★ |
| 2 | Trapping rain water | ★★★ |
| 3 | Ways to decode a message | ★★ |
| 4 | Can you finish all courses? | ★★ |
| 5 | Shipping within D days | ★★★ |
| 6 | k-th largest in a stream | ★★ |
| 7 | Network delay | ★★★ |
| 8 | Largest rectangle in a histogram | ★★★★ |
| 9 | Minimum window substring | ★★★★ |
| 10 | Equal-sum partition | ★★★ |

Full statements are in `exercises.py`. Every test also checks speed, so brute force won't pass the large cases (but writing brute force first is still a great way to check your answers).

## After the contest

- **Review** every solution with the mentor (press **Review**): is there a simpler or faster approach?
- **Practise more** on LeetCode, Codeforces or AtCoder. A sustainable pace is 3–5 problems a week, alongside the next phases.
- **Revisit** any technique you didn't recognize. That's what the journal is for.

## Go deeper (optional, research-level)

1. Competitive programming is now a benchmark for AI systems (AlphaCode, *Competition-Level Code Generation with AlphaCode*, Li et al., 2022). Read how they generated and *filtered* millions of candidate programs using the example tests. What does that say about the value of tests?
2. Pick one problem you solved and prove your solution correct, in writing, using a loop invariant (m04), an exchange argument (m22) or a DP recurrence (m21).
