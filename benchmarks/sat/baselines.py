"""SAT-specific greedy baseline: assigns each variable, in order, to whichever
value (true/false) currently satisfies more of the clauses it appears in.
A single pass, no backtracking or restarts -- unlike a real SAT solver (e.g.
DPLL/CDCL) or even WalkSAT.
"""

from __future__ import annotations

import time

from kona_ebm.datasets.sat import CNF, SATDomain
from kona_ebm.training.metrics import SolveRecord


def _count_satisfied(clauses: CNF, assignment: list[bool], domain: SATDomain) -> int:
    return sum(1 for c in clauses if domain._satisfied_by(c, assignment))


def greedy_sat(problem: CNF, domain: SATDomain) -> SolveRecord:
    t0 = time.time()
    assignment = [False] * domain.n_vars

    for var in range(domain.n_vars):
        clauses_with_var = [c for c in problem if any(abs(lit) - 1 == var for lit in c)]

        assignment[var] = True
        score_true = _count_satisfied(clauses_with_var, assignment, domain)
        assignment[var] = False
        score_false = _count_satisfied(clauses_with_var, assignment, domain)
        assignment[var] = score_true >= score_false

    _, violations = domain.verify(problem, assignment)
    return SolveRecord(
        solved=violations == 0,
        violations=violations,
        n_iters=domain.n_vars,
        runtime_s=time.time() - t0,
    )
