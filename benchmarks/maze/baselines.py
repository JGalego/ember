"""Maze baselines: a myopic greedy walker (always steps to the open neighbor
that most reduces Manhattan distance to the goal, no backtracking -- can get
stuck behind a wall and stop short) and the exact BFS shortest path (the
same algorithm used to generate ground truth, included as an upper-bound
reference point: the EBM solver is not expected to beat an exact polynomial
algorithm on a problem exact algorithms already solve trivially).
"""

from __future__ import annotations

import time

from kona_ebm.datasets.maze import MazeDomain
from kona_ebm.training.metrics import SolveRecord


def greedy_maze(problem: dict, domain: MazeDomain) -> SolveRecord:
    t0 = time.time()
    walls = problem["walls"]
    start, goal = tuple(problem["start"]), tuple(problem["goal"])
    path = [start]
    visited = {start}
    current = start
    n_iters = 0
    max_steps = domain.h * domain.w
    while current != goal and n_iters < max_steps:
        n_iters += 1
        r, c = current
        candidates = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < domain.h
                and 0 <= nc < domain.w
                and walls[nr][nc] == 0
                and (nr, nc) not in visited
            ):
                dist = abs(nr - goal[0]) + abs(nc - goal[1])
                candidates.append((dist, (nr, nc)))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        current = candidates[0][1]
        visited.add(current)
        path.append(current)

    valid, violations = domain.verify(problem, path)
    return SolveRecord(
        solved=valid, violations=violations, n_iters=n_iters, runtime_s=time.time() - t0
    )


def bfs_optimal_maze(problem: dict, domain: MazeDomain) -> SolveRecord:
    t0 = time.time()
    start, goal = tuple(problem["start"]), tuple(problem["goal"])
    path = domain._bfs_shortest_path(problem["walls"], start, goal) or [start]
    valid, violations = domain.verify(problem, path)
    return SolveRecord(
        solved=valid, violations=violations, n_iters=len(path), runtime_s=time.time() - t0
    )
