"""Inference: solve(), the reusable latent-optimization helpers, and plotting."""

from ember.inference.optimize import make_energy_fn, optimize_candidate, optimize_multi_start
from ember.inference.solve import SolveResult, solve
from ember.inference.visualize import (
    plot_convergence_curves,
    plot_energy_curve,
    plot_latent_trajectory,
    save_optimization_movie,
)

__all__ = [
    "make_energy_fn",
    "optimize_candidate",
    "optimize_multi_start",
    "solve",
    "SolveResult",
    "plot_energy_curve",
    "plot_convergence_curves",
    "plot_latent_trajectory",
    "save_optimization_movie",
]
