# m20 · Graphs

**By the end you can:** model problems as graphs, explore them with BFS and DFS, find shortest paths (unweighted and weighted), order tasks with dependencies, and compute PageRank.

**Why it matters for AI:** a neural network *is* a graph of operations. Backpropagation visits that graph in reverse **topological order**, exactly the algorithm in exercise 8. Knowledge graphs, social networks, molecules and road maps are graphs, and **graph neural networks** learn directly on them. PageRank, which made Google, is a random walk on a graph.

---

## 1. Vertices and edges

A graph is a set of **vertices** (nodes) connected by **edges**. Edges can be **directed** (one-way, like "A follows B") or **undirected** (two-way, like "A is friends with B"), and **weighted** (a distance, a cost) or not.

The usual Python representation is an **adjacency list**: a dict from each node to its neighbours.

```python
graph = {
    "A": ["B", "C"],
    "B": ["A", "D"],
    "C": ["A", "D"],
    "D": ["B", "C"],
}

weighted = {"A": [("B", 4), ("C", 1)], "C": [("B", 2)], "B": []}
```

For n nodes and m edges it takes O(n + m) memory. (An n×n **adjacency matrix** takes O(n²) but answers "is there an edge u→v?" in O(1). It's what GNN papers write as A.)

## 2. Breadth-first search (BFS)

Explore in rings: first all neighbours, then their neighbours, and so on. Uses a **queue**:

```python
from collections import deque

def bfs(graph, start):
    seen = {start}
    order = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)          # mark when ADDING, not when popping
                queue.append(nxt)
    return order
```

In an unweighted graph, BFS reaches every node by a **shortest path**. To recover the path, remember each node's `parent` and walk back from the goal.

## 3. Depth-first search (DFS)

Go as deep as possible, then backtrack (m15). Recursive or with an explicit **stack**:

```python
def dfs(graph, node, seen):
    seen.add(node)
    for nxt in graph[node]:
        if nxt not in seen:
            dfs(graph, nxt, seen)
```

Use DFS for: connected components, cycle detection, flood fill on grids, and topological sorting.

**Grids are graphs:** each cell is a node and its up/down/left/right neighbours are edges. Counting islands on a map is counting connected components.

## 4. Cycles in directed graphs

Color each node: **white** (unvisited), **gray** (on the current DFS path), **black** (finished). Reaching a gray node means you've looped back onto your own path: a cycle.

## 5. Topological sort: dependencies in order

Given "task A must happen before task B" edges (a **DAG**, directed acyclic graph), list tasks so every task comes after its prerequisites. **Kahn's algorithm**:

1. Count each node's incoming edges (its in-degree).
2. Put all nodes with in-degree 0 in a queue: they're ready now.
3. Pop one, output it, and decrease its neighbours' in-degrees; add any that reach 0.
4. If you output fewer nodes than exist, there was a cycle.

Course prerequisites, build systems (make), package installers (pip resolving dependencies), spreadsheets recalculating cells, and deep-learning frameworks computing gradients all do this.

## 6. Weighted shortest paths: Dijkstra

With non-negative weights, BFS no longer works (a path with more edges can be cheaper). **Dijkstra's algorithm** always expands the closest unfinished node next, using a **heap** (m19):

```python
import heapq

def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue                       # outdated entry, skip
        for nxt, w in graph[node]:
            nd = d + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return dist
```

O((n + m) log n). Negative weights break it (use Bellman-Ford). A* search adds a heuristic "estimated distance to goal" to explore toward the goal first; it's what game AI and route planners use.

## 7. PageRank

Imagine a surfer who keeps clicking random links, and occasionally (probability 1 − d, usually 0.15) jumps to a completely random page. A page's **PageRank** is the fraction of time the surfer spends there. Compute it by repeating:

```
rank[v] = (1 - d) / N  +  d × Σ over pages u linking to v of  rank[u] / outdegree(u)
```

Start with every rank = 1/N and iterate until the numbers stop changing. Pages with no outgoing links ("dangling nodes") spread their rank evenly over all pages. This is a **power iteration** for the leading eigenvector of a matrix, a link to the linear algebra in Phase 4.

---

## Problem-solving habit #19: is it secretly a graph?

Word ladders (change one letter at a time), puzzle states, course schedules, social distance, currency exchange: whenever you have "things" and "moves between things", draw them as nodes and edges. Then ask: shortest path (BFS / Dijkstra)? reachability (DFS)? ordering (topological sort)? groups (components)?

## Go deeper (optional, research-level)

1. Read the original PageRank paper, *The Anatomy of a Large-Scale Hypertextual Web Search Engine* (Brin & Page, 1998), section 2.1. How does their formula differ from ours?
2. Graph neural networks update each node from its neighbours: hᵥ ← f(hᵥ, Σᵤ g(hᵤ)). Read the introduction of *Semi-Supervised Classification with Graph Convolutional Networks* (Kipf & Welling, 2017). How is one GCN layer like one step of PageRank?
3. Why does Dijkstra fail with negative edge weights? Build the smallest counterexample you can.

## Your turn

Open the **Exercises** tab, solve each function, then press **Run tests**. Graphs are dicts mapping each node to a list of neighbours (or of `(neighbour, weight)` pairs); visit neighbours in the order they're listed.
