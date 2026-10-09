import copy
import random

import pytest

from exercises import (
    bfs_order,
    build_graph,
    count_components,
    count_islands,
    dijkstra,
    has_cycle,
    pagerank,
    shortest_path,
    topo_order,
    word_ladder,
)

rng = random.Random(20)

SQUARE = {"A": ["B", "C"], "B": ["A", "D"], "C": ["A", "D"], "D": ["B", "C", "E"], "E": ["D"], "Z": []}


def test_build_graph():
    assert build_graph([(1, 2), (1, 3)]) == {1: [2, 3], 2: [1], 3: [1]}
    assert build_graph([(1, 2), (2, 3)], directed=True) == {1: [2], 2: [3], 3: []}
    assert build_graph([]) == {}


def test_bfs_order():
    assert bfs_order(SQUARE, "A") == ["A", "B", "C", "D", "E"]
    assert bfs_order(SQUARE, "Z") == ["Z"]


def test_shortest_path():
    assert shortest_path(SQUARE, "A", "E") == ["A", "B", "D", "E"]
    assert shortest_path(SQUARE, "A", "A") == ["A"]
    assert shortest_path(SQUARE, "A", "Z") is None


def test_shortest_path_is_shortest():
    for _ in range(20):
        n = 30
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(40)]
        g = build_graph(edges)
        nodes = list(g)
        a, b = rng.choice(nodes), rng.choice(nodes)
        path = shortest_path(g, a, b)
        # compare with brute-force BFS distances
        dist = {a: 0}
        frontier = [a]
        while frontier:
            nxt = []
            for u in frontier:
                for v in g[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        if b not in dist:
            assert path is None
        else:
            assert path[0] == a and path[-1] == b and len(path) - 1 == dist[b]
            assert all(path[i + 1] in g[path[i]] for i in range(len(path) - 1))


@pytest.mark.parametrize("n, edges, expected", [
    (5, [(0, 1), (1, 2), (3, 4)], 2), (3, [], 3), (1, [], 1), (4, [(0, 1), (2, 3), (1, 2)], 1),
])
def test_count_components(n, edges, expected):
    assert count_components(n, edges) == expected


def test_count_islands():
    grid = [
        list("11000"),
        list("11000"),
        list("00100"),
        list("00011"),
    ]
    original = copy.deepcopy(grid)
    assert count_islands(grid) == 3
    assert grid == original, "don't modify the input"
    assert count_islands([list("111"), list("010"), list("111")]) == 1
    assert count_islands([list("101"), list("010"), list("101")]) == 5
    assert count_islands([]) == 0


def test_has_cycle():
    assert not has_cycle({"a": ["b"], "b": ["c"], "c": []})
    assert has_cycle({"a": ["b"], "b": ["c"], "c": ["a"]})
    assert has_cycle({"a": ["a"]})
    assert not has_cycle({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}), "a diamond is not a cycle"
    assert has_cycle({"x": [], "a": ["b"], "b": ["c"], "c": ["b"]})


def test_topo_order():
    g = {"wake": ["coffee"], "coffee": ["code"], "shower": ["code"], "code": []}
    assert topo_order(g) == ["shower", "wake", "coffee", "code"]
    assert topo_order({"a": ["b"], "b": ["a"]}) is None
    assert topo_order({3: [], 1: [], 2: []}) == [1, 2, 3]


def test_topo_order_respects_every_edge():
    nodes = list(range(40))
    edges = [(a, b) for a in nodes for b in nodes if a < b and rng.random() < 0.1]
    rng.shuffle(nodes)
    g = {n: [] for n in nodes}
    for a, b in edges:
        g[a].append(b)
    order = topo_order(g)
    position = {n: i for i, n in enumerate(order)}
    assert sorted(order) == sorted(nodes)
    assert all(position[a] < position[b] for a, b in edges)


def test_dijkstra():
    g = {"A": [("B", 4), ("C", 1)], "C": [("B", 2), ("D", 7)], "B": [("D", 1)], "D": [], "X": [("A", 1)]}
    assert dijkstra(g, "A") == {"A": 0, "C": 1, "B": 3, "D": 4}
    assert dijkstra(g, "D") == {"D": 0}


def test_dijkstra_matches_brute_force():
    n = 25
    g = {i: [] for i in range(n)}
    for _ in range(80):
        g[rng.randrange(n)].append((rng.randrange(n), rng.randint(0, 20)))
    dist = dijkstra(g, 0)
    # Bellman-Ford as a reference
    ref = {0: 0}
    for _ in range(n):
        for u in g:
            if u in ref:
                for v, w in g[u]:
                    if ref[u] + w < ref.get(v, float("inf")):
                        ref[v] = ref[u] + w
    assert dist == ref


@pytest.mark.parametrize("begin, end, words, expected", [
    ("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"], 5),
    ("hit", "cog", ["hot", "dot", "dog", "lot", "log"], 0),
    ("a", "c", ["a", "b", "c"], 2),
    ("same", "same", ["same"], 1),
])
def test_word_ladder(begin, end, words, expected):
    assert word_ladder(begin, end, words) == expected


def test_pagerank_simple():
    g = {"A": ["B"], "B": ["A"]}
    ranks = pagerank(g)
    assert ranks["A"] == pytest.approx(0.5)
    assert ranks["B"] == pytest.approx(0.5)


def test_pagerank_known_values():
    g = {"A": ["B", "C"], "B": ["C"], "C": ["A"], "D": ["C"]}
    ranks = pagerank(g, d=0.85, iterations=100)
    assert sum(ranks.values()) == pytest.approx(1.0)
    assert ranks == pytest.approx({"A": 0.37252, "B": 0.19582, "C": 0.39416, "D": 0.0375}, abs=1e-4)


def test_pagerank_dangling_nodes():
    g = {"A": ["B"], "B": [], "C": ["B"]}
    ranks = pagerank(g, iterations=100)
    assert sum(ranks.values()) == pytest.approx(1.0)
    assert ranks["B"] > ranks["A"] > 0
