"""Sudoku-specific greedy baseline: no backtracking, so it's a genuinely weak
baseline that will fail on many puzzles -- exactly the kind of thing the EBM
solver's iterative refinement should outperform.
"""

from __future__ import annotations

import time

from kona_ebm.datasets.sudoku import Grid, _is_valid
from kona_ebm.training.metrics import SolveRecord


def greedy_sudoku(problem: Grid) -> SolveRecord:
    t0 = time.time()
    grid = [row[:] for row in problem]
    n_iters = 0
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                continue
            n_iters += 1
            for val in range(1, 10):
                if _is_valid(grid, r, c, val):
                    grid[r][c] = val
                    break
            else:
                grid[r][c] = 1  # no valid digit left; place a placeholder and keep going

    from kona_ebm.datasets.sudoku import SudokuDomain

    _, violations = SudokuDomain().verify(problem, grid)
    return SolveRecord(
        solved=violations == 0, violations=violations, n_iters=n_iters, runtime_s=time.time() - t0
    )
