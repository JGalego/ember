"""Hydra entrypoint: generate datasets, build the EBM stack, train with Lightning.

Examples:
    python scripts/train.py
    python scripts/train.py dataset=sat model=transformer loss=info_nce
    python scripts/train.py dataset=sudoku training.max_epochs=20 training.accelerator=gpu
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ember.datasets import get_domain
from ember.training import EBMLightningModule, make_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    domain_params = OmegaConf.to_container(cfg.dataset.get("params", {}), resolve=True)
    domain = get_domain(cfg.dataset.name, **domain_params)

    train_ds = make_dataset(domain, cfg.dataset.n_train, cfg.dataset.seed_train)
    val_ds = make_dataset(domain, cfg.dataset.n_val, cfg.dataset.seed_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    module = EBMLightningModule(
        domain=domain,
        encoder_cfg=OmegaConf.to_container(cfg.model.encoder, resolve=True),
        energy_cfg=OmegaConf.to_container(cfg.model.energy, resolve=True),
        loss_kind=cfg.loss.kind,
        loss_kwargs=OmegaConf.to_container(cfg.loss.get("kwargs", {}), resolve=True),
        use_constraint_energy=cfg.use_constraint_energy,
        w_learned=cfg.w_learned,
        w_constraint=cfg.w_constraint,
        negatives_per_positive=cfg.negatives_per_positive,
        perturb_noise=cfg.perturb_noise,
        lr=cfg.lr,
    )

    if cfg.training.compile:
        module.encoder = torch.compile(module.encoder)
        module.energy = torch.compile(module.energy)

    run_name = f"{cfg.dataset.name}-{cfg.model.encoder.kind}"
    run_ckpt_dir = Path(cfg.checkpoint_dir) / run_name

    resume_path = None
    resume_cfg = cfg.training.get("resume_from")
    if resume_cfg:
        if resume_cfg == "auto":
            candidate = run_ckpt_dir / "last.ckpt"
            if candidate.exists():
                resume_path = str(candidate)
                print(f"training.resume_from=auto: resuming from {resume_path}")
            else:
                print(
                    f"training.resume_from=auto but no checkpoint found at {candidate}; starting fresh"
                )
        else:
            resume_path = resume_cfg
            print(f"Resuming from {resume_path}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_ckpt_dir),
        filename="{epoch:03d}-{val_loss:.4f}",
        every_n_epochs=cfg.training.checkpoint_every_n_epochs,
        save_top_k=cfg.training.save_top_k,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    logger = CSVLogger(save_dir=cfg.checkpoint_dir, name=run_name)
    if cfg.training.use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(project=cfg.training.wandb_project, name=run_name)
        except ImportError:
            print(
                "wandb not installed; falling back to CSVLogger. Install the `wandb` extra to enable it."
            )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        default_root_dir=cfg.checkpoint_dir,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path)

    ckpt_path = f"{cfg.checkpoint_dir}/{run_name}-final.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
