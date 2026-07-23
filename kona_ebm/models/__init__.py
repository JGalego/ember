"""Model building blocks: encoders, energy heads, decoders, latent optimizer, losses.

Every component in this package is swappable via Hydra config (see
configs/model/*.yaml) -- no architecture is hardcoded into the training or
inference pipelines.
"""

from kona_ebm.models.decoder import Decoder, MLPDecoder, TransformerDecoder, build_decoder
from kona_ebm.models.demo_bundle import build_demo_bundle
from kona_ebm.models.encoder import (
    Encoder,
    GNNEncoder,
    MLPEncoder,
    TransformerEncoder,
    build_encoder,
)
from kona_ebm.models.energy_model import (
    CompositeEnergy,
    ConstraintEnergy,
    DeepSetsEnergy,
    EnergyModel,
    GraphColoringConstraintEnergy,
    MazeConstraintEnergy,
    MLPEnergy,
    SATConstraintEnergy,
    SudokuConstraintEnergy,
    TransformerEnergy,
    build_constraint_energy,
    build_energy_model,
)
from kona_ebm.models.losses import (
    build_loss,
    contrastive_loss,
    hinge_loss,
    info_nce_loss,
    margin_ranking_loss,
)
from kona_ebm.models.optimizer import LatentOptimizer, OptimizationResult, multi_start_optimize

__all__ = [
    "Encoder",
    "MLPEncoder",
    "TransformerEncoder",
    "GNNEncoder",
    "build_encoder",
    "EnergyModel",
    "MLPEnergy",
    "TransformerEnergy",
    "DeepSetsEnergy",
    "ConstraintEnergy",
    "SudokuConstraintEnergy",
    "SATConstraintEnergy",
    "GraphColoringConstraintEnergy",
    "MazeConstraintEnergy",
    "CompositeEnergy",
    "build_energy_model",
    "build_constraint_energy",
    "Decoder",
    "MLPDecoder",
    "TransformerDecoder",
    "build_decoder",
    "build_demo_bundle",
    "LatentOptimizer",
    "OptimizationResult",
    "multi_start_optimize",
    "contrastive_loss",
    "margin_ranking_loss",
    "info_nce_loss",
    "hinge_loss",
    "build_loss",
]
