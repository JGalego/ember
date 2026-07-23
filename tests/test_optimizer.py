import pytest
import torch

from kona_ebm.models.optimizer import LatentOptimizer, multi_start_optimize


def quadratic_energy(z: torch.Tensor) -> torch.Tensor:
    """A simple convex bowl with a unique minimum at z=0; any correct gradient
    descent should drive this to (near) zero energy.
    """
    return (z**2).sum(dim=-1)


@pytest.mark.parametrize("method", ["sgd", "adam", "lbfgs"])
def test_optimizer_minimizes_convex_energy(method):
    torch.manual_seed(0)
    init = torch.randn(4, 10) * 3
    lr = 0.05 if method != "lbfgs" else 0.5
    optimizer = LatentOptimizer(method=method, lr=lr, max_iters=200, tol=1e-8, patience=20)
    result = optimizer.optimize(quadratic_energy, init)
    assert result.energy_history[-1] < result.energy_history[0]
    assert result.energy_history[-1] < 1.0
    assert result.n_iters > 0


def test_optimizer_records_trajectory_when_requested():
    torch.manual_seed(0)
    init = torch.randn(2, 5)
    optimizer = LatentOptimizer(
        method="adam", lr=0.1, max_iters=10, patience=100, record_trajectory=True
    )
    result = optimizer.optimize(quadratic_energy, init)
    assert len(result.trajectory) == result.n_iters
    assert result.trajectory[0].shape == init.shape


def test_optimizer_rejects_unknown_method():
    with pytest.raises(ValueError):
        LatentOptimizer(method="not_a_real_optimizer")


def test_multi_start_optimize_picks_lowest_energy_per_batch_element():
    torch.manual_seed(0)

    def offset_energy(z: torch.Tensor) -> torch.Tensor:
        # minimum at z = 5 for every element -- an optimizer starting far
        # from it should end up worse than one starting close to it.
        return ((z - 5.0) ** 2).sum(dim=-1)

    close_init = torch.full((3, 4), 5.1)
    far_init = torch.full((3, 4), -50.0)
    optimizer = LatentOptimizer(method="adam", lr=0.05, max_iters=5, patience=100)

    best, results = multi_start_optimize(offset_energy, [far_init, close_init], optimizer=optimizer)
    final_energy = offset_energy(best)
    # the close-init trajectory should have been selected for every batch element
    assert torch.allclose(best, results[1].candidate)
    assert (final_energy < offset_energy(results[0].candidate)).all()
