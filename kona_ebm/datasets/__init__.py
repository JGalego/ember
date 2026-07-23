"""Constraint-satisfaction problem domains: dataset generators + tensor encodings.

Importing this package registers every built-in domain (sudoku, sat,
graph_coloring, maze) with the `kona_ebm.datasets.domain` registry, so they
become available via `get_domain(name)`.
"""

from kona_ebm.datasets.domain import (
    Domain,
    ProblemInstance,
    available_domains,
    get_domain,
    register_domain,
)
from kona_ebm.datasets.graph_coloring import GraphColoringDomain
from kona_ebm.datasets.maze import MazeDomain
from kona_ebm.datasets.sat import SATDomain
from kona_ebm.datasets.sudoku import SudokuDomain

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
