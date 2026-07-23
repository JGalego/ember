"""Common interface every constraint-satisfaction domain implements.

A `Domain` is a self-contained plugin that knows how to:
  * generate (problem, solution) instances deterministically from a seed,
  * flatten a raw problem/solution into fixed-size tensors the model stack
    can consume (`encode_problem` / `encode_solution`),
  * map an optimized continuous solution tensor back to a raw, checkable
    structure (`decode_solution`),
  * verify whether a raw solution actually satisfies the problem's
    constraints (`verify`), and
  * produce a random point in solution space to seed latent optimization
    (`random_solution`) and a perturbation of a solution for negative
    sampling during contrastive training (`perturb`).

New CSP domains only need to implement this interface to be usable by every
encoder, energy model, decoder, optimizer, solver, and benchmark in the repo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ProblemInstance:
    """A single generated (problem, solution) pair plus bookkeeping metadata."""

    problem: Any
    solution: Any
    meta: dict[str, Any] = field(default_factory=dict)


class Domain(ABC):
    """Abstract base class for a constraint-satisfaction domain."""

    name: str
    problem_dim: int
    solution_dim: int

    @abstractmethod
    def generate(self, n: int, seed: int) -> list[ProblemInstance]:
        """Generate `n` (problem, solution) instances deterministically from `seed`."""

    @abstractmethod
    def encode_problem(self, problem: Any) -> torch.Tensor:
        """Flatten a raw problem into a fixed-size float tensor of shape (problem_dim,)."""

    @abstractmethod
    def encode_solution(self, solution: Any) -> torch.Tensor:
        """Flatten a raw solution into a continuous-relaxation tensor of shape (solution_dim,)."""

    @abstractmethod
    def decode_solution(self, problem: Any, tensor: torch.Tensor) -> Any:
        """Map an optimized continuous solution tensor back to the raw solution structure."""

    @abstractmethod
    def verify(self, problem: Any, solution: Any) -> tuple[bool, int]:
        """Return (is_valid, num_violated_constraints) for a raw candidate solution."""

    @abstractmethod
    def random_solution(
        self, problem: Any, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """A random point in continuous solution space, used to seed latent optimization."""

    def perturb(
        self,
        solution_tensor: torch.Tensor,
        noise: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Default negative-sample perturbation: additive Gaussian noise + clamp.

        Domains with structured solution spaces (e.g. one-hot blocks) may override
        this to produce more informative negatives (e.g. swapping two entries).
        """
        noise_t = torch.randn(solution_tensor.shape, generator=generator)
        return (solution_tensor + noise * noise_t).clamp(-4.0, 4.0)


_REGISTRY: dict[str, type[Domain]] = {}


def register_domain(cls: type[Domain]) -> type[Domain]:
    """Class decorator that registers a Domain implementation under its `name`."""
    _REGISTRY[cls.name] = cls
    return cls


def get_domain(name: str, **kwargs: Any) -> Domain:
    """Instantiate a registered domain by name, e.g. get_domain('sudoku')."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown domain '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_domains() -> list[str]:
    return sorted(_REGISTRY)
