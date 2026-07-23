"""Decoders: refine an optimized candidate (in continuous solution space) back
into solution space, ready for a domain's hard `decode_solution`/`verify`.

Design note (see docs/architecture.md for the full rationale): unlike
autoregressive seq2seq decoders, latent optimization here (see
`models/optimizer.py`) is performed *directly on the candidate solution's
continuous relaxation* -- e.g. per-cell logits for Sudoku, a tanh-space
assignment for SAT -- rather than on some separate low-dimensional bottleneck.
The problem `Encoder` instead provides a conditioning context vector and a
learned initialization for that optimization. The `Decoder` then acts as a
learned post-processing / denoising step: it takes the optimized candidate
(optionally together with the encoder's context latent) and outputs a
refined candidate of the same shape, which the domain's `decode_solution`
turns into a concrete, checkable structure (grid, assignment, coloring,
path).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Decoder(nn.Module, ABC):
    """Common interface for all decoders."""

    @abstractmethod
    def forward(self, candidate: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        """(B, solution_dim), optional (B, latent_dim) context -> refined (B, solution_dim)."""


class MLPDecoder(Decoder):
    """Residual feed-forward decoder: output = candidate + MLP([candidate, latent])."""

    def __init__(
        self, solution_dim: int, latent_dim: int = 0, hidden_dims: tuple[int, ...] = (256, 256)
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        dims = [solution_dim + latent_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU()]
        layers.append(nn.Linear(dims[-1], solution_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, candidate: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        x = candidate if latent is None else torch.cat([candidate, latent], dim=-1)
        return candidate + self.net(x)


class TransformerDecoder(Decoder):
    """Tokenizes [candidate, latent] the same way `TransformerEncoder` tokenizes
    a problem, runs a Transformer encoder stack over the tokens, and projects
    back to `solution_dim` as a residual correction.
    """

    def __init__(
        self,
        solution_dim: int,
        latent_dim: int = 0,
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
        self.tokenize = nn.Linear(solution_dim + latent_dim, n_tokens * token_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = nn.Linear(n_tokens * token_dim, solution_dim)

    def forward(self, candidate: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        batch = candidate.shape[0]
        x = candidate if latent is None else torch.cat([candidate, latent], dim=-1)
        tokens = self.tokenize(x).view(batch, self.n_tokens, self.token_dim) + self.pos_embed
        encoded = self.transformer(tokens).reshape(batch, -1)
        return candidate + self.readout(encoded)


_DECODERS: dict[str, type[Decoder]] = {
    "mlp": MLPDecoder,
    "transformer": TransformerDecoder,
}


def build_decoder(kind: str, solution_dim: int, latent_dim: int = 0, **kwargs) -> Decoder:
    if kind not in _DECODERS:
        raise KeyError(f"Unknown decoder kind '{kind}'. Available: {sorted(_DECODERS)}")
    return _DECODERS[kind](solution_dim=solution_dim, latent_dim=latent_dim, **kwargs)
