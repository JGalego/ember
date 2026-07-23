"""Plotting helpers for Step 12: energy vs. iteration, latent trajectories,
convergence-curve comparisons, and optimization "movies".

Uses matplotlib's non-interactive Agg backend throughout, so these work in
headless environments (CI, containers) without a display.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def plot_energy_curve(
    energy_history: list[float], path: str, title: str = "Energy vs. iteration"
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energy_history)
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence_curves(
    histories: dict[str, list[float]], path: str, title: str = "Convergence comparison"
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, hist in histories.items():
        ax.plot(hist, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pca_2d(trajectory_stack: torch.Tensor) -> torch.Tensor:
    centered = trajectory_stack - trajectory_stack.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    return centered @ v[:, :2]


def plot_latent_trajectory(
    trajectory: list[torch.Tensor], path: str, title: str = "Candidate trajectory (PCA)"
) -> None:
    """PCA-projects a sequence of (flattened) candidate/latent tensors to 2D
    and plots the optimization path through that projection.
    """
    if len(trajectory) < 2:
        raise ValueError("Need at least 2 trajectory points to plot")
    stack = torch.stack([t.reshape(-1) for t in trajectory])
    proj = _pca_2d(stack).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(proj[:, 0], proj[:, 1], "-o", markersize=3, alpha=0.7)
    ax.scatter([proj[0, 0]], [proj[0, 1]], c="green", label="start", zorder=5)
    ax.scatter([proj[-1, 0]], [proj[-1, 1]], c="red", label="end", zorder=5)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_optimization_movie(trajectory: list[torch.Tensor], path: str, fps: int = 5) -> None:
    """Generic diagnostic animation: reshapes each trajectory step's candidate
    into a near-square 2D grid and renders it as a heatmap frame per
    iteration. This is a domain-agnostic diagnostic (it does not draw Sudoku
    digits, graph edges, etc.) -- see `benchmarks/` for domain-specific plots.
    """
    if len(trajectory) < 2:
        raise ValueError("Need at least 2 trajectory points to animate")

    flat = [t.reshape(-1) for t in trajectory]
    n = flat[0].numel()
    side = int(n**0.5)
    if side * side != n:
        side += 1
        pad = side * side - n
        flat = [torch.cat([f, torch.full((pad,), float("nan"))]) for f in flat]
    frames = [f.reshape(side, side).numpy() for f in flat]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap="viridis", animated=True)
    title = ax.set_title("iteration 0")
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        title.set_text(f"iteration {i}")
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
