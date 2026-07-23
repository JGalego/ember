"""Random 3-SAT domain with a planted satisfying assignment.

Each instance is built by sampling a random assignment first, then generating
clauses that are always satisfied by it (flipping one literal on the rare
occasion a random draw wouldn't be). This is the standard "planted solution"
trick for building guaranteed-satisfiable random CSP datasets and keeps
generation trivial (no SAT solver required), at the cost of a mild bias
toward instances close to the planted assignment -- acceptable for a research
benchmark, not a substitute for a hardness-calibrated SAT competition set.

Clauses are represented as a fixed-size (max_clauses, n_vars) tensor with
entries in {-1, 0, +1}: +1 if the variable appears positively in the clause,
-1 if negated, 0 if absent.
"""

from __future__ import annotations

import random

import torch

from ember.datasets.domain import Domain, ProblemInstance, register_domain

Clause = list[int]
CNF = list[Clause]


@register_domain
class SATDomain(Domain):
    name = "sat"

    def __init__(self, n_vars: int = 20, max_clauses: int = 91, k: int = 3) -> None:
        self.n_vars = n_vars
        self.max_clauses = max_clauses
        self.k = k
        self.problem_dim = max_clauses * n_vars
        self.solution_dim = n_vars

    def generate(self, n: int, seed: int) -> list[ProblemInstance]:
        rng = random.Random(seed)
        instances = []
        for _ in range(n):
            assignment = [rng.random() < 0.5 for _ in range(self.n_vars)]
            clauses: CNF = []
            attempts = 0
            max_attempts = self.max_clauses * 20
            while len(clauses) < self.max_clauses and attempts < max_attempts:
                attempts += 1
                chosen_vars = rng.sample(range(1, self.n_vars + 1), self.k)
                clause = [v if rng.random() < 0.5 else -v for v in chosen_vars]
                if not self._satisfied_by(clause, assignment):
                    flip = rng.randrange(self.k)
                    var = abs(clause[flip]) - 1
                    clause[flip] = (var + 1) if assignment[var] else -(var + 1)
                clauses.append(clause)
            instances.append(
                ProblemInstance(
                    problem=clauses, solution=assignment, meta={"n_clauses": len(clauses)}
                )
            )
        return instances

    @staticmethod
    def _satisfied_by(clause: Clause, assignment: list[bool]) -> bool:
        return any(
            (lit > 0 and assignment[abs(lit) - 1]) or (lit < 0 and not assignment[abs(lit) - 1])
            for lit in clause
        )

    def encode_problem(self, problem: CNF) -> torch.Tensor:
        mat = torch.zeros(self.max_clauses, self.n_vars)
        for i, clause in enumerate(problem[: self.max_clauses]):
            for lit in clause:
                var = abs(lit) - 1
                mat[i, var] = 1.0 if lit > 0 else -1.0
        return mat.reshape(-1)

    def encode_solution(self, solution: list[bool]) -> torch.Tensor:
        return torch.tensor([1.0 if v else -1.0 for v in solution], dtype=torch.float32)

    def decode_solution(self, problem: CNF, tensor: torch.Tensor) -> list[bool]:
        return [bool(x.item() > 0) for x in tensor.reshape(-1)]

    def verify(self, problem: CNF, solution: list[bool]) -> tuple[bool, int]:
        violations = sum(0 if self._satisfied_by(c, solution) else 1 for c in problem if c)
        return violations == 0, violations

    def random_solution(
        self, problem: CNF, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.rand(self.solution_dim, generator=generator) * 2 - 1

    def perturb(
        self,
        solution_tensor: torch.Tensor,
        noise: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = solution_tensor.clone()
        n_flip = max(1, int(0.2 * self.solution_dim))
        idx = torch.randperm(self.solution_dim, generator=generator)[:n_flip]
        x[idx] = -x[idx]
        x = x + noise * 0.2 * torch.randn(x.shape, generator=generator)
        return x.clamp(-3.0, 3.0)
