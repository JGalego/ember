"""Encoders: map a flattened problem tensor into a latent vector.

Every encoder shares the same interface -- `forward(problem) -> latent`, with
`problem` of shape (B, problem_dim) and `latent` of shape (B, latent_dim) --
so any encoder can be swapped for any other through Hydra config alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Encoder(nn.Module, ABC):
    """Common interface for all problem encoders."""

    latent_dim: int

    @abstractmethod
    def forward(self, problem: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch of flattened problems (B, problem_dim) into (B, latent_dim)."""


class MLPEncoder(Encoder):
    """Plain feed-forward encoder."""

    def __init__(
        self, problem_dim: int, latent_dim: int, hidden_dims: tuple[int, ...] = (256, 256)
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        dims = [problem_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU()]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, problem: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(problem)


class TransformerEncoder(Encoder):
    """Tokenizes the flattened problem into `n_tokens` pseudo-tokens via a learned
    linear projection, runs a standard Transformer encoder stack over them, and
    mean-pools the result into a latent vector.
    """

    def __init__(
        self,
        problem_dim: int,
        latent_dim: int,
        n_tokens: int = 16,
        token_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.tokenize = nn.Linear(problem_dim, n_tokens * token_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(token_dim, latent_dim)

    def forward(self, problem: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        batch = problem.shape[0]
        tokens = self.tokenize(problem).view(batch, self.n_tokens, self.token_dim)
        tokens = tokens + self.pos_embed
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.readout(pooled)


class GNNEncoder(Encoder):
    """Lightweight message-passing encoder (no torch_geometric dependency).

    The flattened problem is projected into `n_nodes` pseudo-node features.
    If an explicit adjacency matrix is available for the domain (e.g. graph
    coloring), pass it via `forward(..., adjacency=...)`; otherwise message
    passing defaults to a fully-connected graph over the pseudo-nodes. See
    PyTorch Geometric (REFERENCES.md) for a production-grade GNN library this
    could be swapped for.
    """

    def __init__(
        self,
        problem_dim: int,
        latent_dim: int,
        n_nodes: int = 16,
        node_dim: int = 32,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.node_dim = node_dim
        self.input_proj = nn.Linear(problem_dim, n_nodes * node_dim)
        self.message_layers = nn.ModuleList(
            [nn.Linear(node_dim * 2, node_dim) for _ in range(n_layers)]
        )
        self.update_layers = nn.ModuleList(
            [nn.Linear(node_dim * 2, node_dim) for _ in range(n_layers)]
        )
        self.readout = nn.Linear(node_dim, latent_dim)

    def forward(self, problem: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        batch = problem.shape[0]
        h = self.input_proj(problem).view(batch, self.n_nodes, self.node_dim)

        if adjacency is None:
            adj = torch.ones(self.n_nodes, self.n_nodes, device=problem.device) - torch.eye(
                self.n_nodes, device=problem.device
            )
        else:
            adj = adjacency.to(h.dtype)

        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers, strict=True):
            if adj.dim() == 2:
                deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
                agg = torch.einsum("ij,bjd->bid", adj, h) / deg
            else:
                deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
                agg = torch.einsum("bij,bjd->bid", adj, h) / deg
            msg = torch.relu(msg_layer(torch.cat([h, agg], dim=-1)))
            h = torch.relu(upd_layer(torch.cat([h, msg], dim=-1)))

        pooled = h.mean(dim=1)
        return self.readout(pooled)


_ENCODERS: dict[str, type[Encoder]] = {
    "mlp": MLPEncoder,
    "transformer": TransformerEncoder,
    "gnn": GNNEncoder,
}


def build_encoder(kind: str, problem_dim: int, latent_dim: int, **kwargs) -> Encoder:
    if kind not in _ENCODERS:
        raise KeyError(f"Unknown encoder kind '{kind}'. Available: {sorted(_ENCODERS)}")
    return _ENCODERS[kind](problem_dim=problem_dim, latent_dim=latent_dim, **kwargs)
