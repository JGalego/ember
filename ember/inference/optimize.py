"""Thin convenience layer over `models/optimizer.py`: closes an energy function
over a fixed (problem, latent) pair so callers only need to think in terms of
"optimize this candidate," not the full `E(problem, latent, candidate)` signature.
"""

from __future__ import annotations

import torch

from ember.models.energy_model import CompositeEnergy
from ember.models.optimizer import (
    EnergyFn,
    LatentOptimizer,
    OptimizationResult,
    multi_start_optimize,
)


def make_energy_fn(
    energy: CompositeEnergy, problem: torch.Tensor, latent: torch.Tensor
) -> EnergyFn:
    def energy_fn(candidate: torch.Tensor) -> torch.Tensor:
        return energy(problem, latent, candidate)

    return energy_fn


def optimize_candidate(
    energy: CompositeEnergy,
    problem: torch.Tensor,
    latent: torch.Tensor,
    init: torch.Tensor,
    method: str = "adam",
    lr: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-4,
    patience: int = 10,
    record_trajectory: bool = False,
) -> OptimizationResult:
    energy_fn = make_energy_fn(energy, problem, latent)
    optimizer = LatentOptimizer(
        method=method,
        lr=lr,
        max_iters=max_iters,
        tol=tol,
        patience=patience,
        record_trajectory=record_trajectory,
    )
    return optimizer.optimize(energy_fn, init)


def optimize_multi_start(
    energy: CompositeEnergy,
    problem: torch.Tensor,
    latent: torch.Tensor,
    init_candidates: list[torch.Tensor],
    **optimizer_kwargs,
) -> tuple[torch.Tensor, list[OptimizationResult]]:
    energy_fn = make_energy_fn(energy, problem, latent)
    return multi_start_optimize(energy_fn, init_candidates, **optimizer_kwargs)
