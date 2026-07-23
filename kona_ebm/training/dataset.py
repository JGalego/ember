"""PyTorch Dataset wrapping a domain's generated (problem, solution) instances."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from kona_ebm.datasets.domain import Domain, ProblemInstance


class CSPDataset(Dataset):
    """Wraps pre-generated `ProblemInstance`s and encodes them lazily on access."""

    def __init__(self, domain: Domain, instances: list[ProblemInstance]) -> None:
        self.domain = domain
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inst = self.instances[idx]
        problem = self.domain.encode_problem(inst.problem)
        solution = self.domain.encode_solution(inst.solution)
        return problem, solution


def make_dataset(domain: Domain, n: int, seed: int) -> CSPDataset:
    """Generate `n` instances from `domain` and wrap them in a `CSPDataset`."""
    return CSPDataset(domain, domain.generate(n, seed))
