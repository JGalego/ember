"""Training loop: dataset wrapping, Lightning module, and shared metrics."""

from ember.training.dataset import CSPDataset, make_dataset
from ember.training.metrics import SolveRecord, aggregate
from ember.training.trainer import EBMLightningModule, PlainEBMTrainer

__all__ = [
    "CSPDataset",
    "make_dataset",
    "SolveRecord",
    "aggregate",
    "EBMLightningModule",
    "PlainEBMTrainer",
]
