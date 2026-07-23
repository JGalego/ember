"""Graph coloring greedy baseline (Welsh-Powell style): process nodes in
descending-degree order, assign the smallest color not used by
already-colored neighbors; if all `k` colors conflict (the palette is fixed
-- no "add a new color" escape hatch), pick whichever color causes the
fewest conflicts.
"""

from __future__ import annotations

import time

from kona_ebm.datasets.graph_coloring import Adjacency, GraphColoringDomain
from kona_ebm.training.metrics import SolveRecord


def greedy_graph_coloring(problem: Adjacency, domain: GraphColoringDomain) -> SolveRecord:
    t0 = time.time()
    n = domain.n_nodes
    degree = [sum(problem[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degree[i])
    colors = [-1] * n

    for node in order:
        neighbor_colors = {colors[j] for j in range(n) if problem[node][j] == 1 and colors[j] != -1}
        available = [c for c in range(domain.k_colors) if c not in neighbor_colors]
        if available:
            colors[node] = available[0]
        else:
            conflict_counts = [
                sum(1 for j in range(n) if problem[node][j] == 1 and colors[j] == c)
                for c in range(domain.k_colors)
            ]
            colors[node] = min(range(domain.k_colors), key=lambda c: conflict_counts[c])

    _, violations = domain.verify(problem, colors)
    return SolveRecord(
        solved=violations == 0, violations=violations, n_iters=n, runtime_s=time.time() - t0
    )
