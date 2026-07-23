"""Constraint-satisfaction problem domains: dataset generators + tensor encodings.

Importing this package registers every built-in domain (sudoku, sat,
graph_coloring, maze) with the `ember.datasets.domain` registry, so they
become available via `get_domain(name)`.
"""

from ember.datasets.domain import (
    Domain,
    ProblemInstance,
    available_domains,
    get_domain,
    register_domain,
)
from ember.datasets.graph_coloring import GraphColoringDomain
from ember.datasets.maze import MazeDomain
from ember.datasets.sat import SATDomain
from ember.datasets.sudoku import SudokuDomain

__all__ = [
    "Domain",
    "ProblemInstance",
    "register_domain",
    "get_domain",
    "available_domains",
    "SudokuDomain",
    "SATDomain",
    "GraphColoringDomain",
    "MazeDomain",
]
