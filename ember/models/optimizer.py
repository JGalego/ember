"""Latent (candidate-solution) optimization: inference by energy minimization
instead of autoregressive decoding.

    candidate = init(problem)
    repeat:
        energy = E(problem, candidate)
        candidate = candidate - alpha * grad(energy, candidate)
    until convergence

`LatentOptimizer` runs a single optimization trajectory with a chosen
optimizer (SGD / Adam / LBFGS); `multi_start_optimize` runs several
independently-initialized trajectories and keeps, per batch element, whichever
one reached the lowest final energy (Step 10 of the design).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

EnergyFn = Callable[[torch.Tensor], torch.Tensor]  # candidate (B, D) -> energy (B,)


@dataclass
class OptimizationResult:
    candidate: torch.Tensor
    energy_history: list[float] = field(default_factory=list)
    trajectory: list[torch.Tensor] = field(default_factory=list)
    n_iters: int = 0
    converged: bool = False


class LatentOptimizer:
    """Minimizes a scalar energy function w.r.t. a candidate tensor via gradient descent."""

    def __init__(
        self,
        method: str = "adam",
        lr: float = 0.1,
        max_iters: int = 200,
        tol: float = 1e-4,
        patience: int = 10,
        record_trajectory: bool = False,
    ) -> None:
        if method not in ("sgd", "adam", "lbfgs"):
            raise ValueError(f"Unknown optimization method '{method}'. Use sgd, adam, or lbfgs.")
        self.method = method
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.patience = patience
        self.record_trajectory = record_trajectory

    def optimize(self, energy_fn: EnergyFn, init: torch.Tensor) -> OptimizationResult:
        z = init.clone().detach().requires_grad_(True)
        history: list[float] = []
        trajectory: list[torch.Tensor] = []

        if self.method in ("sgd", "adam"):
            opt_cls = torch.optim.SGD if self.method == "sgd" else torch.optim.Adam
            opt = opt_cls([z], lr=self.lr)
            stall = 0
            prev_mean: float | None = None
            n_iters = 0
            for _ in range(self.max_iters):
                n_iters += 1
                opt.zero_grad()
                energy = energy_fn(z)
                energy.sum().backward()
                opt.step()
                mean_energy = energy.detach().mean().item()
                history.append(mean_energy)
                if self.record_trajectory:
                    trajectory.append(z.detach().clone())
                if prev_mean is not None and abs(prev_mean - mean_energy) < self.tol:
                    stall += 1
                    if stall >= self.patience:
                        break
                else:
                    stall = 0
                prev_mean = mean_energy
            converged = stall >= self.patience
        else:  # lbfgs
            opt = torch.optim.LBFGS(
                [z],
                lr=self.lr,
                max_iter=self.max_iters,
                tolerance_change=self.tol,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                energy = energy_fn(z)
                loss = energy.sum()
                loss.backward()
                history.append(energy.detach().mean().item())
                if self.record_trajectory:
                    trajectory.append(z.detach().clone())
                return loss

            opt.step(closure)
            n_iters = len(history)
            converged = True

        return OptimizationResult(
            candidate=z.detach(),
            energy_history=history,
            trajectory=trajectory,
            n_iters=n_iters,
            converged=converged,
        )


def multi_start_optimize(
    energy_fn: EnergyFn,
    init_candidates: list[torch.Tensor],
    optimizer: LatentOptimizer | None = None,
    **optimizer_kwargs,
) -> tuple[torch.Tensor, list[OptimizationResult]]:
    """Run one optimization trajectory per initialization in `init_candidates` and
    return, per batch element, the candidate from whichever start reached the
    lowest final energy -- along with every trajectory's result for inspection.
    """
    if optimizer is None:
        optimizer = LatentOptimizer(**optimizer_kwargs)

    results = [optimizer.optimize(energy_fn, init) for init in init_candidates]

    with torch.no_grad():
        finals = torch.stack([energy_fn(r.candidate) for r in results], dim=0)  # (n_starts, B)
        best_start = finals.argmin(dim=0)  # (B,)
        candidates = torch.stack([r.candidate for r in results], dim=0)  # (n_starts, B, D)
        batch_idx = torch.arange(candidates.shape[1])
        best_candidate = candidates[best_start, batch_idx]

    return best_candidate, results
