"""Graph coloring domain: fixed-size random graphs with a planted proper k-coloring.

Nodes are assigned a random color first, then edges are added between
differently-colored node pairs only (with probability `edge_prob`), which
guarantees the planted coloring is valid by construction -- the same
planted-solution trick used for the SAT generator, applied to graphs.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from ember.datasets.domain import Domain, ProblemInstance, register_domain

Adjacency = list[list[int]]
Coloring = list[int]


@register_domain
class GraphColoringDomain(Domain):
    name = "graph_coloring"

    def __init__(self, n_nodes: int = 12, k_colors: int = 3, edge_prob: float = 0.35) -> None:
        self.n_nodes = n_nodes
        self.k_colors = k_colors
        self.edge_prob = edge_prob
        self.problem_dim = n_nodes * n_nodes
        self.solution_dim = n_nodes * k_colors

    def generate(self, n: int, seed: int) -> list[ProblemInstance]:
        rng = random.Random(seed)
        instances = []
        for _ in range(n):
            colors = [rng.randrange(self.k_colors) for _ in range(self.n_nodes)]
            adjacency = [[0] * self.n_nodes for _ in range(self.n_nodes)]
            n_edges = 0
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    if colors[i] != colors[j] and rng.random() < self.edge_prob:
                        adjacency[i][j] = 1
                        adjacency[j][i] = 1
                        n_edges += 1
            instances.append(
                ProblemInstance(problem=adjacency, solution=colors, meta={"n_edges": n_edges})
            )
        return instances

    def encode_problem(self, problem: Adjacency) -> torch.Tensor:
        return torch.tensor(problem, dtype=torch.float32).reshape(-1)

    def encode_solution(self, solution: Coloring) -> torch.Tensor:
        colors = torch.tensor(solution, dtype=torch.long)
        return F.one_hot(colors, num_classes=self.k_colors).float().reshape(-1)

    def decode_solution(self, problem: Adjacency, tensor: torch.Tensor) -> Coloring:
        logits = tensor.reshape(self.n_nodes, self.k_colors)
        return logits.argmax(dim=-1).tolist()

    def verify(self, problem: Adjacency, solution: Coloring) -> tuple[bool, int]:
        violations = 0
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if problem[i][j] == 1 and solution[i] == solution[j]:
                    violations += 1
        return violations == 0, violations

    def random_solution(
        self, problem: Adjacency, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.randn(self.solution_dim, generator=generator)

    def perturb(
        self,
        solution_tensor: torch.Tensor,
        noise: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        x = solution_tensor.clone().reshape(self.n_nodes, self.k_colors)
        n_swap = max(2, int(0.3 * self.n_nodes))
        idx = torch.randperm(self.n_nodes, generator=generator)[:n_swap]
        perm = idx[torch.randperm(len(idx), generator=generator)]
        x[idx] = x[perm].clone()
        x = x + noise * 0.3 * torch.randn(x.shape, generator=generator)
        return x.reshape(-1)
