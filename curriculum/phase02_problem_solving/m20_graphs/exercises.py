"""m20 exercises: graphs.

A graph is a dict: node -> list of neighbours (visit them in the listed order).
A weighted graph is a dict: node -> list of (neighbour, weight) pairs.
"""

import heapq
from collections import deque


# 1. Build an adjacency list from a list of (u, v) edges. Every node that appears
#    in an edge must be a key. For undirected graphs add both directions.
#    Keep neighbours in the order the edges were given.
#    build_graph([(1, 2), (1, 3)]) -> {1: [2, 3], 2: [1], 3: [1]}
def build_graph(edges, directed=False):
    raise NotImplementedError


# 2. The order in which BFS visits nodes, starting at `start`.
def bfs_order(graph, start):
    raise NotImplementedError


# 3. A shortest path (fewest edges) from start to goal as a list of nodes,
#    or None if goal can't be reached. Use BFS and remember each node's parent.
def shortest_path(graph, start, goal):
    raise NotImplementedError


# 4. Number of connected components in an undirected graph with nodes 0..n-1.
#    count_components(5, [(0, 1), (1, 2), (3, 4)]) -> 2
def count_components(n, edges):
    raise NotImplementedError


# 5. Count islands: groups of "1" cells connected up/down/left/right.
#    Don't modify the input grid.
def count_islands(grid):
    raise NotImplementedError


# 6. Does a DIRECTED graph contain a cycle? (DFS with white/gray/black colouring,
#    or Kahn's algorithm.)
def has_cycle(graph):
    raise NotImplementedError


# 7. Topological order of a directed graph using Kahn's algorithm. When several nodes
#    are ready at once, take the SMALLEST first (use a heap) so the answer is unique.
#    Return None if there's a cycle.
#    topo_order({"wake": ["coffee"], "coffee": ["code"], "shower": ["code"], "code": []})
#    -> ["shower", "wake", "coffee", "code"]
def topo_order(graph):
    raise NotImplementedError


# 8. Dijkstra: the shortest distance from start to every reachable node (dict).
def dijkstra(graph, start):
    raise NotImplementedError


# 9. Word ladder: the fewest words in a sequence from `begin` to `end`, changing one
#    letter at a time, where every word after `begin` must be in `words`.
#    Count both ends; return 0 if impossible.
#    word_ladder("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) -> 5
#    (hit → hot → dot → dog → cog)
def word_ladder(begin, end, words):
    raise NotImplementedError


# 10. PageRank with damping d, starting from 1/N for every node and running
#     `iterations` updates of:
#        rank[v] = (1 - d) / N + d * (sum of rank[u] / outdegree(u) over u linking to v
#                                     + (sum of rank of dangling nodes) / N)
#     A dangling node has no outgoing links. Return a dict node -> rank (sums to 1).
def pagerank(graph, d=0.85, iterations=50):
    raise NotImplementedError
