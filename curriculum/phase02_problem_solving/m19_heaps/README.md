# m19 · Heaps and priority queues

**By the end you can:** explain how a binary heap works inside, implement one from scratch, and use Python's `heapq` to solve top-k, merging and scheduling problems efficiently.

**Why it matters for AI:** when an LLM generates text with **top-k sampling**, it needs the k most likely next tokens out of ~100,000. Beam search keeps the best k candidate sequences. Nearest-neighbour search in vector databases (the "R" in RAG) keeps the k closest vectors. Dijkstra's shortest paths (next module) and job schedulers on GPU clusters use priority queues. All of these are heaps.

---

## 1. The problem a heap solves

You need to repeatedly **get the smallest item** while items keep arriving.

| Structure | insert | get & remove min |
|---|---|---|
| unsorted list | O(1) | O(n) |
| sorted list | O(n) | O(1) |
| **binary heap** | **O(log n)** | **O(log n)** |

## 2. What a heap is

A **min-heap** is a complete binary tree where every parent is ≤ its children. So the smallest item is always at the root. Siblings are in no particular order: a heap is much less organized than a sorted list, which is exactly why it's cheaper to maintain.

The clever part: a complete tree fits perfectly in a plain list, with no pointers:

```
              1                index:  0  1  2  3  4  5
            /   \              value: [1, 3, 2, 7, 4, 5]
           3     2
          / \   /              children of i:  2i + 1 and 2i + 2
         7   4 5               parent of i:    (i - 1) // 2
```

## 3. The two operations

**Push** (insert): append the item at the end, then **sift up**: while it's smaller than its parent, swap them.

**Pop** (remove the min): the answer is `heap[0]`. Move the *last* item to the root, then **sift down**: while it's bigger than its smaller child, swap with that child.

Each sift moves along one path from root to leaf: the tree's height, which is O(log n).

**Heapify:** turning a list of n items into a heap can be done in O(n) (not O(n log n)!) by sifting down every non-leaf, starting from the last one.

## 4. heapq in practice

```python
import heapq

h = []
heapq.heappush(h, 5)
heapq.heappush(h, 1)
heapq.heappush(h, 3)
h[0]                      # 1: peek at the smallest
heapq.heappop(h)          # 1

nums = [5, 1, 8, 3]
heapq.heapify(nums)       # in place, O(n)

heapq.nlargest(3, nums)   # top 3
heapq.nsmallest(2, nums)
```

`heapq` only provides a **min**-heap. For a max-heap, push negated values (`-x`). For items with priorities, push tuples: `(priority, counter, item)`. The counter breaks ties, so Python never tries to compare two items directly (which fails for dicts or custom objects).

## 5. The top-k pattern

To keep the k **largest** of a stream, keep a **min**-heap of size k. Its root is the smallest of your current top-k, the one to kick out when something bigger arrives:

```python
def top_k(stream, k):
    heap = []
    for x in stream:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)     # pop the min and push x, in one step
    return sorted(heap, reverse=True)
```

O(n log k) time and only O(k) memory: it works on streams too large to sort or even store.

## 6. Merging sorted streams

To merge k sorted lists, push the first item of each list into a heap; repeatedly pop the smallest and push the next item from the same list. O(N log k) for N items in total. This is how databases and external sorting merge sorted files too big for memory.

## 7. Two heaps: running median

To track the median of a stream: a **max**-heap for the smaller half and a **min**-heap for the larger half, kept the same size (± 1). The median is at the top of one or both. Each new number costs O(log n).

---

## Problem-solving habit #18: "repeatedly the best" → heap

If an algorithm repeatedly needs the smallest/largest/most urgent item from a changing collection, reach for a heap. If you only need it once, `min()` or sorting is simpler.

## Go deeper (optional, research-level)

1. Prove that heapify is O(n). (Hint: most nodes are near the bottom and sift down only a little. Sum the work level by level.)
2. Read how top-k and top-p (nucleus) sampling work in *The Curious Case of Neural Text Degeneration* (Holtzman et al., 2019). Why does pure greedy decoding produce repetitive text?
3. Vector databases (FAISS, HNSW) answer "find the k nearest vectors" over millions of embeddings without comparing against all of them. Where in the HNSW algorithm are priority queues used?

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. In exercise 1 you build a heap yourself; after that you may use `heapq`.
