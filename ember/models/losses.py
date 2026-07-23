"""Contrastive losses for training an energy function from (problem, positive
solution, negative/perturbed solution) triples. Every loss has the signature
`loss(e_pos, e_neg, **kwargs) -> scalar Tensor`, where `e_pos` is (B,) and
`e_neg` is (B,) or (B, K) for K negatives per positive, so they can be swapped
through Hydra config alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _broadcast_negatives(e_neg: torch.Tensor) -> torch.Tensor:
    return e_neg.unsqueeze(1) if e_neg.dim() == 1 else e_neg


def contrastive_loss(e_pos: torch.Tensor, e_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Square-square energy-based contrastive loss (LeCun et al., *A Tutorial
    on Energy-Based Learning*): pulls the positive energy toward 0 and pushes
    the negative energy above `margin`.
    """
    e_neg = _broadcast_negatives(e_neg)
    return (e_pos.pow(2) + torch.relu(margin - e_neg).pow(2).mean(dim=1)).mean()


def margin_ranking_loss(
    e_pos: torch.Tensor, e_neg: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Pairwise ranking loss requiring E(pos) + margin <= E(neg), computed with
    `torch.nn.functional.margin_ranking_loss`. Assumes one negative per positive.
    """
    e_neg = e_neg if e_neg.dim() == 1 else e_neg.mean(dim=1)
    target = torch.ones_like(e_pos)
    return F.margin_ranking_loss(e_neg, e_pos, target, margin=margin)


def info_nce_loss(
    e_pos: torch.Tensor, e_neg: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """InfoNCE / CPC-style loss: treats `-energy` as a similarity logit and
    contrasts the positive against the K negatives in each row of `e_neg`.
    """
    e_neg = _broadcast_negatives(e_neg)
    logits = torch.cat([-e_pos.unsqueeze(1), -e_neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def hinge_loss(e_pos: torch.Tensor, e_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Structured hinge loss supporting multiple negatives per positive:
    penalizes any negative whose energy gap to the positive is smaller than
    `margin`, i.e. E(neg) - E(pos) < margin.
    """
    e_neg = _broadcast_negatives(e_neg)
    gap = e_neg - e_pos.unsqueeze(1)
    return torch.relu(margin - gap).mean()


_LOSSES = {
    "contrastive": contrastive_loss,
    "margin_ranking": margin_ranking_loss,
    "info_nce": info_nce_loss,
    "hinge": hinge_loss,
}


def build_loss(kind: str):
    if kind not in _LOSSES:
        raise KeyError(f"Unknown loss kind '{kind}'. Available: {sorted(_LOSSES)}")
    return _LOSSES[kind]
