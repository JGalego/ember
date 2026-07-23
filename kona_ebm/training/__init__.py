"""Training loop: dataset wrapping, Lightning module, and shared metrics."""

from kona_ebm.training.dataset import CSPDataset, make_dataset
from kona_ebm.training.metrics import SolveRecord, aggregate
from kona_ebm.training.trainer import EBMLightningModule, PlainEBMTrainer

__all__ = [
    "CSPDataset",
    "make_dataset",
    "SolveRecord",
    "aggregate",
    "EBMLightningModule",
    "PlainEBMTrainer",
]
