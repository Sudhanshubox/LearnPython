# m22 · Greedy algorithms and intervals

**By the end you can:** solve problems by repeatedly making the locally best choice, *prove* (or disprove) that greedy works for a given problem, and handle interval scheduling problems with sorting and sweeping.

**Why it matters for AI:** greedy decoding (always pick the most likely next token) is the simplest way to generate text, and understanding when greedy fails explains why beam search and sampling exist. **Huffman coding** (exercise 7) is the foundation of compression and is directly tied to **entropy** and **cross-entropy**, the loss function you'll use to train almost every classifier and language model. Interval scheduling is how clusters pack GPU jobs.

---

## 1. The greedy idea

At each step, make the choice that looks best right now, never reconsider, and hope it adds up to the best overall answer.

- **When it works**, it's usually the simplest and fastest solution (often just "sort, then one pass").
- **When it doesn't**, it's confidently wrong. Coin change with coins `[1, 5, 6]` and amount 10: greedy takes 6 + 1 + 1 + 1 + 1 (5 coins); the best is 5 + 5 (2 coins). That's why m21 needed DP.

The hard part of greedy is never the code. It's knowing whether it's correct.

## 2. Proving greedy correct: the exchange argument

Take any optimal solution. Show you can **swap** one of its choices for the greedy choice without making it worse. Repeat until the optimal solution *is* the greedy one. So greedy is optimal too.

**Activity selection:** choose the most non-overlapping meetings from a list.
Greedy: **sort by end time**, and take each meeting that starts after the last one you took ended.

Why? Let `g` be the meeting that ends first. Any optimal schedule starts with some meeting `o`. Since `g` ends no later than `o`, replacing `o` by `g` can't create a conflict, and the count is the same. Repeat for the rest of the schedule.

(Sorting by *start* time or by *shortest duration* both fail. Find counterexamples; it's good practice.)

## 3. Intervals: sort, then sweep

Most interval problems start by sorting:

```python
def merge(intervals):
    intervals = sorted(intervals)                  # by start
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:      # overlaps the last one
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

**Sweep line:** to find the most meetings happening at once (the number of rooms needed), turn each interval into two events, `(start, +1)` and `(end, -1)`, sort the events, and walk through them keeping a running count. The maximum count is the answer. (Process an end before a start at the same time if a room freed at 10:00 can be reused at 10:00.)

## 4. Huffman coding

To compress text, give frequent characters short bit codes and rare characters long ones. Huffman's greedy algorithm builds the optimal prefix code:

1. Put every symbol in a min-heap (m19) keyed by its frequency.
2. Repeatedly pop the two least frequent, merge them into one node whose frequency is their sum, and push it back.
3. When one node is left, it's the root of the code tree. Each symbol's code length is its depth in the tree.

The total cost (sum of frequency × code length) is the minimum possible for any prefix code.

**The link to AI:** Shannon proved the best possible average code length is the **entropy** H = −Σ p log₂ p bits per symbol, and Huffman gets within 1 bit of it. When you train a language model with **cross-entropy loss**, you are literally minimizing how many bits the model would need to compress the training text. Better language models are better compressors.

## 5. A checklist for greedy

1. What's the greedy choice? (Often: sort by some key, then take what fits.)
2. Try to break it: small examples, ties, edge cases. Check against brute force.
3. If you can't break it, try the exchange argument.
4. If you *can* break it, the problem probably needs DP (m21) or search (m15).

---

## Problem-solving habit #21: test greedy against brute force

Whenever you have a greedy idea, write a tiny brute-force solver too, and compare them on hundreds of random small inputs. It takes five minutes and catches wrong greedy ideas immediately. Several tests in this module do exactly that, and exercise 8 asks you to automate it.

## Go deeper (optional, research-level)

1. Compute the entropy of the character frequencies in a paragraph of English, and compare it with your Huffman code's average bits per character. How close is it?
2. Read *Language Modeling Is Compression* (Delétang et al., 2023). How can a language model be turned into a compressor, and how did LLMs compare with gzip on images and audio?
3. For which coin systems is greedy change-making always optimal? These are called **canonical** coin systems. Use exercise 8 to test the Indian rupee system (1, 2, 5, 10, 20, 50, 100, 200, 500, 2000) and the old British system (1, 3, 6, 12, 24, 30).

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**.
