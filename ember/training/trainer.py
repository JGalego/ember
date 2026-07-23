"""Lightning module wiring encoder + energy (learned + constraint) + contrastive
loss into a single trainable unit, plus a plain-PyTorch fallback loop for
environments without Lightning installed.
"""

from __future__ import annotations

from typing import Any

import torch

from ember.datasets.domain import Domain
from ember.models import (
    CompositeEnergy,
    build_constraint_energy,
    build_encoder,
    build_energy_model,
    build_loss,
)

try:
    import lightning.pytorch as pl

    _HAS_LIGHTNING = True
except ImportError:  # pragma: no cover - exercised only when lightning isn't installed
    pl = None
    _HAS_LIGHTNING = False


def _build_stack(
    domain: Domain,
    encoder_cfg: dict[str, Any],
    energy_cfg: dict[str, Any],
    use_constraint_energy: bool,
    w_learned: float,
    w_constraint: float,
) -> tuple[torch.nn.Module, CompositeEnergy]:
    encoder_cfg = dict(encoder_cfg)
    latent_dim = encoder_cfg.pop("latent_dim", 64)
    encoder_kind = encoder_cfg.pop("kind", "mlp")
    encoder = build_encoder(
        encoder_kind, problem_dim=domain.problem_dim, latent_dim=latent_dim, **encoder_cfg
    )

    energy_cfg = dict(energy_cfg)
    energy_kind = energy_cfg.pop("kind", "mlp")
    learned = build_energy_model(
        energy_kind, latent_dim=latent_dim, solution_dim=domain.solution_dim, **energy_cfg
    )
    constraint = build_constraint_energy(domain.name, domain) if use_constraint_energy else None
    composite = CompositeEnergy(
        learned=learned, constraint=constraint, w_learned=w_learned, w_constraint=w_constraint
    )
    return encoder, composite


class _EBMModuleBase:
    """Shared training-step logic used by both the Lightning module and the
    plain-PyTorch fallback trainer below.
    """

    encoder: torch.nn.Module
    energy: CompositeEnergy
    domain: Domain
    loss_fn: Any
    negatives_per_positive: int
    perturb_noise: float
    loss_kwargs: dict[str, Any]

    def compute_loss(
        self, problem: torch.Tensor, solution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = solution.shape[0]
        negatives = torch.stack(
            [
                self.domain.perturb(solution[i], noise=self.perturb_noise)
                for i in range(batch)
                for _ in range(self.negatives_per_positive)
            ]
        ).view(batch, self.negatives_per_positive, -1)

        latent = self.encoder(problem)
        e_pos = self.energy(problem, latent, solution)
        e_negs = torch.stack(
            [
                self.energy(problem, latent, negatives[:, k])
                for k in range(self.negatives_per_positive)
            ],
            dim=1,
        )
        loss = self.loss_fn(e_pos, e_negs, **self.loss_kwargs)
        return loss, e_pos, e_negs


if _HAS_LIGHTNING:

    class EBMLightningModule(pl.LightningModule, _EBMModuleBase):
        """Trains an encoder + composite energy with a configurable contrastive loss."""

        def __init__(
            self,
            domain: Domain,
            encoder_cfg: dict[str, Any],
            energy_cfg: dict[str, Any],
            loss_kind: str = "contrastive",
            loss_kwargs: dict[str, Any] | None = None,
            use_constraint_energy: bool = True,
            w_learned: float = 1.0,
            w_constraint: float = 1.0,
            negatives_per_positive: int = 1,
            perturb_noise: float = 0.5,
            lr: float = 1e-3,
        ) -> None:
            super().__init__()
            self.save_hyperparameters(ignore=["domain"])
            self.domain = domain
            self.encoder, self.energy = _build_stack(
                domain, encoder_cfg, energy_cfg, use_constraint_energy, w_learned, w_constraint
            )
            self.loss_fn = build_loss(loss_kind)
            self.loss_kwargs = loss_kwargs or {}
            self.negatives_per_positive = negatives_per_positive
            self.perturb_noise = perturb_noise

        def training_step(self, batch, batch_idx):
            problem, solution = batch
            loss, e_pos, e_neg = self.compute_loss(problem, solution)
            self.log("train_loss", loss, prog_bar=True, batch_size=problem.shape[0])
            self.log("train_e_pos", e_pos.mean(), batch_size=problem.shape[0])
            self.log("train_e_neg", e_neg.mean(), batch_size=problem.shape[0])
            return loss

        def validation_step(self, batch, batch_idx):
            problem, solution = batch
            loss, e_pos, e_neg = self.compute_loss(problem, solution)
            self.log("val_loss", loss, prog_bar=True, batch_size=problem.shape[0])
            self.log(
                "val_energy_gap", (e_neg.mean(dim=1) - e_pos).mean(), batch_size=problem.shape[0]
            )
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

else:  # pragma: no cover
    EBMLightningModule = None


class PlainEBMTrainer(_EBMModuleBase):
    """Minimal PyTorch training loop, used when `lightning` isn't installed or
    for quick scripted experiments outside the Hydra/Lightning entrypoint.
    """

    def __init__(
        self,
        domain: Domain,
        encoder_cfg: dict[str, Any],
        energy_cfg: dict[str, Any],
        loss_kind: str = "contrastive",
        loss_kwargs: dict[str, Any] | None = None,
        use_constraint_energy: bool = True,
        w_learned: float = 1.0,
        w_constraint: float = 1.0,
        negatives_per_positive: int = 1,
        perturb_noise: float = 0.5,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.domain = domain
        self.device = torch.device(device)
        self.encoder, self.energy = _build_stack(
            domain, encoder_cfg, energy_cfg, use_constraint_energy, w_learned, w_constraint
        )
        self.encoder.to(self.device)
        self.energy.to(self.device)
        self.loss_fn = build_loss(loss_kind)
        self.loss_kwargs = loss_kwargs or {}
        self.negatives_per_positive = negatives_per_positive
        self.perturb_noise = perturb_noise
        params = list(self.encoder.parameters()) + list(self.energy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def fit(self, dataloader, epochs: int = 1) -> list[float]:
        history = []
        self.encoder.train()
        self.energy.train()
        for _ in range(epochs):
            for problem, solution in dataloader:
                problem, solution = problem.to(self.device), solution.to(self.device)
                self.optimizer.zero_grad()
                loss, _, _ = self.compute_loss(problem, solution)
                loss.backward()
                self.optimizer.step()
                history.append(loss.item())
        return history
