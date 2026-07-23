"""Domain-generic baselines and the benchmark harness.

`simulated_annealing` and `beam_search` operate purely through the `Domain`
interface (`random_solution` / `perturb` / `decode_solution` / `verify`), so
they work unmodified for any registered domain. `TransformerBaseline` is a
single-shot supervised model (problem -> solution in one forward pass, no
iterative energy optimization) used as the "traditional transformer" point of
comparison against the EBM's iterative-refinement inference.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

import torch
from torch import nn

from kona_ebm.datasets.domain import Domain
from kona_ebm.training.metrics import SolveRecord, aggregate


def simulated_annealing(
    domain: Domain,
    problem_raw,
    init_temp: float = 2.0,
    cooling: float = 0.995,
    n_steps: int = 2000,
    seed: int = 0,
) -> SolveRecord:
    t0 = time.time()
    rng = torch.Generator().manual_seed(seed)
    py_rng = random.Random(seed)

    def violations_of(cand: torch.Tensor) -> int:
        raw = domain.decode_solution(problem_raw, cand)
        _, v = domain.verify(problem_raw, raw)
        return v

    candidate = domain.random_solution(problem_raw, generator=rng)
    current_viol = violations_of(candidate)
    best_candidate, best_viol = candidate.clone(), current_viol
    temp = init_temp
    step = 0
    for step in range(n_steps):  # noqa: B007 -- `step` is used after the loop to report n_iters
        proposal = domain.perturb(candidate, noise=0.5, generator=rng)
        proposal_viol = violations_of(proposal)
        delta = proposal_viol - current_viol
        if delta <= 0 or py_rng.random() < math.exp(-delta / max(temp, 1e-6)):
            candidate, current_viol = proposal, proposal_viol
            if current_viol < best_viol:
                best_candidate, best_viol = candidate.clone(), current_viol
        temp *= cooling
        if best_viol == 0:
            break

    solution_raw = domain.decode_solution(problem_raw, best_candidate)
    valid, violations = domain.verify(problem_raw, solution_raw)
    return SolveRecord(
        solved=valid, violations=violations, n_iters=step + 1, runtime_s=time.time() - t0
    )


def beam_search(
    domain: Domain,
    problem_raw,
    beam_width: int = 8,
    branching: int = 4,
    n_steps: int = 200,
    seed: int = 0,
) -> SolveRecord:
    t0 = time.time()
    rng = torch.Generator().manual_seed(seed)

    def violations_of(cand: torch.Tensor) -> int:
        raw = domain.decode_solution(problem_raw, cand)
        _, v = domain.verify(problem_raw, raw)
        return v

    beam = [domain.random_solution(problem_raw, generator=rng) for _ in range(beam_width)]
    beam.sort(key=violations_of)
    best_candidate, best_viol = beam[0], violations_of(beam[0])
    step = 0
    for step in range(n_steps):  # noqa: B007 -- `step` is used after the loop to report n_iters
        candidates = list(beam)
        for c in beam:
            candidates.extend(domain.perturb(c, noise=0.3, generator=rng) for _ in range(branching))
        candidates.sort(key=violations_of)
        beam = candidates[:beam_width]
        if violations_of(beam[0]) < best_viol:
            best_candidate, best_viol = beam[0], violations_of(beam[0])
        if best_viol == 0:
            break

    solution_raw = domain.decode_solution(problem_raw, best_candidate)
    valid, violations = domain.verify(problem_raw, solution_raw)
    return SolveRecord(
        solved=valid, violations=violations, n_iters=step + 1, runtime_s=time.time() - t0
    )


class TransformerBaseline(nn.Module):
    """Single-shot supervised baseline: problem -> solution in one forward pass
    (a small Transformer encoder over tokenized problem features), trained
    with plain regression/classification loss against ground-truth solutions.
    No iterative optimization at inference time -- this is the point of
    comparison the EBM's energy-minimization inference is contrasted against.
    """

    def __init__(
        self,
        problem_dim: int,
        solution_dim: int,
        n_tokens: int = 16,
        token_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.tokenize = nn.Linear(problem_dim, n_tokens * token_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads, dim_feedforward=4 * token_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(n_tokens * token_dim, solution_dim)

    def forward(self, problem: torch.Tensor) -> torch.Tensor:
        batch = problem.shape[0]
        tokens = self.tokenize(problem).view(batch, self.n_tokens, self.token_dim) + self.pos_embed
        encoded = self.transformer(tokens).reshape(batch, -1)
        return self.readout(encoded)


def train_transformer_baseline(
    domain: Domain,
    problems: torch.Tensor,
    solutions: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
) -> TransformerBaseline:
    model = TransformerBaseline(domain.problem_dim, domain.solution_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(problems)
        loss = torch.nn.functional.mse_loss(pred, solutions)
        loss.backward()
        opt.step()
    return model


def solve_with_transformer_baseline(
    model: TransformerBaseline, domain: Domain, problem_raw
) -> SolveRecord:
    t0 = time.time()
    model.eval()
    problem_t = domain.encode_problem(problem_raw).unsqueeze(0)
    with torch.no_grad():
        candidate = model(problem_t).squeeze(0)
    solution_raw = domain.decode_solution(problem_raw, candidate)
    valid, violations = domain.verify(problem_raw, solution_raw)
    return SolveRecord(solved=valid, violations=violations, n_iters=1, runtime_s=time.time() - t0)


def run_benchmark(
    solvers: dict[str, Callable[[], SolveRecord]],
) -> dict[str, SolveRecord]:
    """Run every `name -> zero-arg solver callable` once and return their records."""
    return {name: solver() for name, solver in solvers.items()}


def summarize(records_by_solver: dict[str, list[SolveRecord]]) -> dict[str, dict[str, float]]:
    return {name: aggregate(records) for name, records in records_by_solver.items()}
