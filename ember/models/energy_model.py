"""Energy networks: E(problem, candidate_solution) -> scalar, lower is better.

Two complementary kinds of energy are provided, matching Steps 4-5 of the
design:

* `EnergyModel` subclasses are *learned* energy heads (MLP / Transformer /
  Deep Sets) that take an encoded problem latent and a candidate solution
  tensor and output a scalar per batch element. These are what contrastive
  training (see `losses.py`) shapes.
* `ConstraintEnergy` subclasses are *hand-specified, differentiable*
  constraint-violation penalties, one per domain, operating directly on the
  (soft/continuous-relaxed) candidate tensor and the raw problem tensor. They
  give the model an explicit, interpretable signal for violated constraints,
  duplicate values, invalid assignments, and structural inconsistency,
  independent of whatever the learned energy head has picked up.

`CompositeEnergy` combines both via a configurable weighted sum, so a model
can be trained with the learned energy alone, the constraint energy alone, or
any blend of the two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Learned energy heads
# --------------------------------------------------------------------------- #


class EnergyModel(nn.Module, ABC):
    """Common interface for all learned energy heads."""

    @abstractmethod
    def forward(self, latent: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim), (B, solution_dim) -> (B,) scalar energy, lower is better."""


class MLPEnergy(EnergyModel):
    def __init__(
        self, latent_dim: int, solution_dim: int, hidden_dims: tuple[int, ...] = (256, 256)
    ) -> None:
        super().__init__()
        dims = [latent_dim + solution_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU()]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, candidate], dim=-1)
        return self.net(x).squeeze(-1)


class TransformerEnergy(EnergyModel):
    def __init__(
        self,
        latent_dim: int,
        solution_dim: int,
        n_tokens: int = 16,
        token_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.tokenize = nn.Linear(latent_dim + solution_dim, n_tokens * token_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(token_dim, 1)

    def forward(self, latent: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        x = torch.cat([latent, candidate], dim=-1)
        tokens = self.tokenize(x).view(batch, self.n_tokens, self.token_dim) + self.pos_embed
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.readout(pooled).squeeze(-1)


class DeepSetsEnergy(EnergyModel):
    """Permutation-invariant energy over `n_elements` chunks of the candidate
    (e.g. per-cell / per-variable slices), following Zaheer et al.'s Deep Sets:
    phi is applied per-element, summed, then rho maps the pooled representation
    to a scalar. Useful when constraint structure is largely symmetric across
    elements; not appropriate when element order/identity itself matters.
    """

    def __init__(
        self,
        latent_dim: int,
        solution_dim: int,
        n_elements: int,
        element_hidden: int = 64,
        rho_hidden: int = 128,
    ) -> None:
        super().__init__()
        if solution_dim % n_elements != 0:
            raise ValueError(
                f"solution_dim ({solution_dim}) must be divisible by n_elements ({n_elements})"
            )
        self.n_elements = n_elements
        self.element_dim = solution_dim // n_elements
        self.phi = nn.Sequential(
            nn.Linear(latent_dim + self.element_dim, element_hidden),
            nn.GELU(),
            nn.Linear(element_hidden, element_hidden),
        )
        self.rho = nn.Sequential(
            nn.Linear(element_hidden, rho_hidden),
            nn.GELU(),
            nn.Linear(rho_hidden, 1),
        )

    def forward(self, latent: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        batch = candidate.shape[0]
        elements = candidate.view(batch, self.n_elements, self.element_dim)
        latent_broadcast = latent.unsqueeze(1).expand(-1, self.n_elements, -1)
        phi_in = torch.cat([latent_broadcast, elements], dim=-1)
        phi_out = self.phi(phi_in)
        pooled = phi_out.sum(dim=1)
        return self.rho(pooled).squeeze(-1)


_ENERGY_MODELS: dict[str, type[EnergyModel]] = {
    "mlp": MLPEnergy,
    "transformer": TransformerEnergy,
    "deep_sets": DeepSetsEnergy,
}


def build_energy_model(kind: str, latent_dim: int, solution_dim: int, **kwargs) -> EnergyModel:
    if kind not in _ENERGY_MODELS:
        raise KeyError(f"Unknown energy model kind '{kind}'. Available: {sorted(_ENERGY_MODELS)}")
    return _ENERGY_MODELS[kind](latent_dim=latent_dim, solution_dim=solution_dim, **kwargs)


# --------------------------------------------------------------------------- #
# Differentiable constraint-violation penalties (one per domain)
# --------------------------------------------------------------------------- #


class ConstraintEnergy(nn.Module, ABC):
    """Hand-specified, differentiable constraint penalty for a domain.

    Operates on a *batch* of raw problem tensors (as produced by
    `Domain.encode_problem`, stacked) and a batch of continuous-relaxation
    candidate tensors (as produced by `Domain.encode_solution` / latent
    optimization) and returns a non-negative scalar per batch element: 0 means
    "no soft violation detected", larger is worse.
    """

    @abstractmethod
    def forward(self, problem: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor: ...


class SudokuConstraintEnergy(ConstraintEnergy):
    """Row / column / box duplicate-value penalties + given-clue consistency."""

    def __init__(self) -> None:
        super().__init__()
        boxes = []
        for br in range(3):
            for bc in range(3):
                idx = [
                    (r * 9 + c)
                    for r in range(br * 3, br * 3 + 3)
                    for c in range(bc * 3, bc * 3 + 3)
                ]
                boxes.append(idx)
        self.register_buffer("box_index", torch.tensor(boxes, dtype=torch.long))  # (9, 9)

    def forward(self, problem: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        batch = candidate.shape[0]
        probs = torch.softmax(candidate.view(batch, 81, 9), dim=-1)
        grid = probs.view(batch, 9, 9, 9)  # (B, row, col, digit)

        row_sums = grid.sum(dim=2)  # (B, row, digit) -- want == 1 for each digit
        col_sums = grid.sum(dim=1)  # (B, col, digit)
        box_probs = probs[:, self.box_index, :]  # (B, 9 boxes, 9 cells, 9 digits)
        box_sums = box_probs.sum(dim=2)  # (B, box, digit)

        dup_penalty = (
            ((row_sums - 1.0) ** 2).sum(dim=(1, 2))
            + ((col_sums - 1.0) ** 2).sum(dim=(1, 2))
            + ((box_sums - 1.0) ** 2).sum(dim=(1, 2))
        )

        prob_grid = problem.view(batch, 81, 10)
        given_mask = 1.0 - prob_grid[..., 0]  # (B, 81), 1 where a clue is given
        given_digit = prob_grid[..., 1:10]  # (B, 81, 9) one-hot of the given digit
        log_probs = torch.log(probs.clamp(min=1e-8))
        given_penalty = -(given_digit * log_probs).sum(dim=-1) * given_mask
        given_penalty = given_penalty.sum(dim=-1)

        return dup_penalty + given_penalty


class SATConstraintEnergy(ConstraintEnergy):
    """Soft clause-satisfaction margin penalty."""

    def __init__(self, n_vars: int, max_clauses: int, margin: float = 0.3) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.max_clauses = max_clauses
        self.margin = margin

    def forward(self, problem: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        batch = candidate.shape[0]
        clause_mat = problem.view(batch, self.max_clauses, self.n_vars)
        x = torch.tanh(candidate)  # (B, n_vars) in (-1, 1)
        literal_scores = clause_mat * x.unsqueeze(1)  # (B, clauses, vars), 0 where literal absent
        mask = clause_mat != 0
        neg_fill = torch.finfo(literal_scores.dtype).min / 4
        masked_scores = literal_scores.masked_fill(~mask, neg_fill)
        clause_score, _ = masked_scores.max(dim=-1)  # (B, clauses)
        has_literals = mask.any(dim=-1).float()
        violation = torch.relu(self.margin - clause_score) * has_literals
        return violation.sum(dim=-1)


class GraphColoringConstraintEnergy(ConstraintEnergy):
    """Soft same-color-across-edge penalty."""

    def __init__(self, n_nodes: int, k_colors: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.k_colors = k_colors

    def forward(self, problem: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        batch = candidate.shape[0]
        adjacency = problem.view(batch, self.n_nodes, self.n_nodes)
        probs = torch.softmax(candidate.view(batch, self.n_nodes, self.k_colors), dim=-1)
        same_color = torch.einsum("bik,bjk->bij", probs, probs)  # (B, N, N)
        triu = torch.triu(
            torch.ones(self.n_nodes, self.n_nodes, device=candidate.device), diagonal=1
        )
        penalty = (adjacency * same_color * triu).sum(dim=(1, 2))
        return penalty


class MazeConstraintEnergy(ConstraintEnergy):
    """Wall-avoidance + start/goal-inclusion + a soft local-connectivity proxy.

    Exact path connectivity is not a smooth function of the candidate mask, so
    this penalty is a differentiable *proxy* only -- `Domain.verify` (used by
    the hard self-correction loop in `inference/solve.py`) is the source of
    truth for whether a decoded path is actually connected.
    """

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.h = height
        self.w = width

    def forward(self, problem: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        batch = candidate.shape[0]
        channels = problem.view(batch, 3, self.h, self.w)
        walls, start_ch, goal_ch = channels[:, 0], channels[:, 1], channels[:, 2]
        m = torch.sigmoid(candidate.view(batch, self.h, self.w))

        wall_penalty = (m * walls).sum(dim=(1, 2))

        start_prob = (m * start_ch).sum(dim=(1, 2))
        goal_prob = (m * goal_ch).sum(dim=(1, 2))
        endpoint_penalty = (1.0 - start_prob) ** 2 + (1.0 - goal_prob) ** 2

        padded = torch.nn.functional.pad(m.unsqueeze(1), (1, 1, 1, 1))
        neighbor_sum = (
            padded[:, :, :-2, 1:-1]
            + padded[:, :, 2:, 1:-1]
            + padded[:, :, 1:-1, :-2]
            + padded[:, :, 1:-1, 2:]
        ).squeeze(1)
        isolation_penalty = (m * torch.exp(-2.0 * neighbor_sum)).sum(dim=(1, 2))

        return wall_penalty + endpoint_penalty + 0.5 * isolation_penalty


_CONSTRAINT_ENERGIES: dict[str, type[ConstraintEnergy]] = {
    "sudoku": SudokuConstraintEnergy,
    "sat": SATConstraintEnergy,
    "graph_coloring": GraphColoringConstraintEnergy,
    "maze": MazeConstraintEnergy,
}


def build_constraint_energy(domain_name: str, domain) -> ConstraintEnergy:
    """Construct the constraint energy matching a `Domain` instance's parameters."""
    if domain_name == "sudoku":
        return SudokuConstraintEnergy()
    if domain_name == "sat":
        return SATConstraintEnergy(n_vars=domain.n_vars, max_clauses=domain.max_clauses)
    if domain_name == "graph_coloring":
        return GraphColoringConstraintEnergy(n_nodes=domain.n_nodes, k_colors=domain.k_colors)
    if domain_name == "maze":
        return MazeConstraintEnergy(height=domain.h, width=domain.w)
    raise KeyError(
        f"No constraint energy registered for domain '{domain_name}'. Available: {sorted(_CONSTRAINT_ENERGIES)}"
    )


class CompositeEnergy(nn.Module):
    """Weighted combination of a learned `EnergyModel` and a `ConstraintEnergy`.

    E_total = w_learned * E_learned(latent, candidate) + w_constraint * E_constraint(problem, candidate)

    Either term can be dropped by passing `None` and its weight is then ignored.
    """

    def __init__(
        self,
        learned: EnergyModel | None = None,
        constraint: ConstraintEnergy | None = None,
        w_learned: float = 1.0,
        w_constraint: float = 1.0,
    ) -> None:
        super().__init__()
        if learned is None and constraint is None:
            raise ValueError("CompositeEnergy needs at least one of `learned` or `constraint`")
        self.learned = learned
        self.constraint = constraint
        self.w_learned = w_learned
        self.w_constraint = w_constraint

    def forward(
        self, problem: torch.Tensor, latent: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        total = torch.zeros(candidate.shape[0], device=candidate.device, dtype=candidate.dtype)
        if self.learned is not None:
            total = total + self.w_learned * self.learned(latent, candidate)
        if self.constraint is not None:
            total = total + self.w_constraint * self.constraint(problem, candidate)
        return total
