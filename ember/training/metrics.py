"""Solve-quality metrics shared by training validation and the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SolveRecord:
    """Outcome of a single solve attempt (EBM or baseline)."""

    solved: bool
    violations: int
    n_iters: int
    runtime_s: float


def aggregate(records: list[SolveRecord]) -> dict[str, float]:
    """Summarize a batch of `SolveRecord`s into accuracy / convergence / runtime / violations."""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "mean_violations": 0.0,
            "mean_convergence_steps": 0.0,
            "mean_runtime_s": 0.0,
        }
    return {
        "n": n,
        "accuracy": sum(r.solved for r in records) / n,
        "mean_violations": sum(r.violations for r in records) / n,
        "mean_convergence_steps": sum(r.n_iters for r in records) / n,
        "mean_runtime_s": sum(r.runtime_s for r in records) / n,
    }
