"""End-to-end solver: encode -> optimize latent (multi-start) -> decode ->
verify constraints -> self-correct (restart or keep optimizing) -> return.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from kona_ebm.datasets.domain import Domain
from kona_ebm.models.decoder import Decoder
from kona_ebm.models.encoder import Encoder
from kona_ebm.models.energy_model import CompositeEnergy
from kona_ebm.models.optimizer import LatentOptimizer, multi_start_optimize

from .optimize import make_energy_fn


@dataclass
class SolveResult:
    solution: Any
    valid: bool
    violations: int
    final_energy: float
    n_iters: int
    n_restarts: int
    runtime_s: float
    energy_history: list[float] = field(default_factory=list)


def solve(
    domain: Domain,
    problem_raw: Any,
    encoder: Encoder,
    energy: CompositeEnergy,
    decoder: Decoder | None = None,
    method: str = "adam",
    lr: float = 0.1,
    max_iters: int = 200,
    tol: float = 1e-4,
    patience: int = 10,
    n_starts: int = 4,
    max_restarts: int = 3,
    energy_threshold: float = 0.5,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> SolveResult:
    """Solve a single problem instance via latent-optimization inference.

    Self-correction (Step 11): if the decoded candidate fails `domain.verify`
    after one round of (multi-start) optimization, and its energy is still
    above `energy_threshold`, the whole optimization is restarted from fresh
    random initializations (up to `max_restarts` times), keeping the
    least-violating candidate seen across restarts.
    """
    t0 = time.time()
    device_t = torch.device(device)

    problem_t = domain.encode_problem(problem_raw).unsqueeze(0).to(device_t)

    encoder.eval()
    energy.eval()
    if decoder is not None:
        decoder.eval()

    with torch.no_grad():
        latent = encoder(problem_t)

    energy_fn = make_energy_fn(energy, problem_t, latent)
    optimizer = LatentOptimizer(
        method=method, lr=lr, max_iters=max_iters, tol=tol, patience=patience
    )

    total_iters = 0
    best_solution = None
    best_violations: int | None = None
    best_valid = False
    best_final_energy = float("inf")
    best_history: list[float] = []
    restarts_used = 0

    for restart in range(max_restarts):
        restarts_used = restart + 1
        inits = [
            domain.random_solution(problem_raw, generator=generator).unsqueeze(0).to(device_t)
            for _ in range(n_starts)
        ]
        candidate, results = multi_start_optimize(energy_fn, inits, optimizer=optimizer)
        total_iters += sum(r.n_iters for r in results)

        if decoder is not None:
            with torch.no_grad():
                candidate = decoder(candidate, latent)

        final_energy = energy_fn(candidate).item()
        solution_raw = domain.decode_solution(problem_raw, candidate.squeeze(0))
        valid, violations = domain.verify(problem_raw, solution_raw)

        if best_violations is None or violations < best_violations:
            best_solution = solution_raw
            best_violations = violations
            best_valid = valid
            best_final_energy = final_energy
            best_history = min(results, key=lambda r: r.energy_history[-1]).energy_history

        if valid or final_energy < energy_threshold:
            break

    return SolveResult(
        solution=best_solution,
        valid=best_valid,
        violations=best_violations if best_violations is not None else -1,
        final_energy=best_final_energy,
        n_iters=total_iters,
        n_restarts=restarts_used,
        runtime_s=time.time() - t0,
        energy_history=best_history,
    )
