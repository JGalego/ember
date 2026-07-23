"""Builds an untrained "demo" encoder/energy/decoder stack for a domain: a
small MLP encoder/energy/decoder from a fixed seed, with the hand-specified
constraint energy weighted heavily relative to the (untrained) learned energy
head. This is what `ember.api.main` and `scripts/run_benchmarks.py` use
when no trained checkpoint is supplied -- useful for smoke-testing the solver
pipeline immediately, not a substitute for actually training a model.
"""

from __future__ import annotations

import torch

from ember.datasets.domain import Domain
from ember.models.decoder import Decoder, build_decoder
from ember.models.encoder import Encoder, build_encoder
from ember.models.energy_model import (
    CompositeEnergy,
    build_constraint_energy,
    build_energy_model,
)


def build_demo_bundle(
    domain: Domain,
    latent_dim: int = 32,
    w_learned: float = 0.1,
    w_constraint: float = 5.0,
    seed: int = 0,
) -> tuple[Encoder, CompositeEnergy, Decoder]:
    torch.manual_seed(seed)
    encoder = build_encoder("mlp", problem_dim=domain.problem_dim, latent_dim=latent_dim)
    learned = build_energy_model("mlp", latent_dim=latent_dim, solution_dim=domain.solution_dim)
    constraint = build_constraint_energy(domain.name, domain)
    energy = CompositeEnergy(
        learned=learned, constraint=constraint, w_learned=w_learned, w_constraint=w_constraint
    )
    decoder = build_decoder("mlp", solution_dim=domain.solution_dim, latent_dim=latent_dim)
    encoder.eval()
    energy.eval()
    decoder.eval()
    return encoder, energy, decoder
