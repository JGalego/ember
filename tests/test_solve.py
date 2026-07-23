import torch

from ember.datasets import get_domain
from ember.inference import solve
from ember.inference.optimize import optimize_candidate
from ember.models import build_demo_bundle


def test_solve_pipeline_returns_well_formed_result():
    domain = get_domain("graph_coloring", n_nodes=8, k_colors=3, edge_prob=0.3)
    inst = domain.generate(1, seed=0)[0]
    encoder, energy, decoder = build_demo_bundle(domain, seed=1)

    result = solve(
        domain,
        inst.problem,
        encoder,
        energy,
        decoder=decoder,
        method="adam",
        lr=0.2,
        max_iters=100,
        n_starts=3,
        max_restarts=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert isinstance(result.violations, int)
    assert result.n_iters > 0
    assert result.n_restarts >= 1
    assert result.runtime_s >= 0
    assert len(result.solution) == domain.n_nodes


def test_solve_self_correction_reduces_violations_via_restarts():
    domain = get_domain("graph_coloring", n_nodes=10, k_colors=3, edge_prob=0.4)
    inst = domain.generate(1, seed=2)[0]
    encoder, energy, decoder = build_demo_bundle(domain, seed=2, w_learned=0.0, w_constraint=1.0)

    single_restart = solve(
        domain,
        inst.problem,
        encoder,
        energy,
        decoder=decoder,
        max_iters=20,
        n_starts=1,
        max_restarts=1,
        generator=torch.Generator().manual_seed(0),
    )
    multi_restart = solve(
        domain,
        inst.problem,
        encoder,
        energy,
        decoder=decoder,
        max_iters=20,
        n_starts=1,
        max_restarts=8,
        energy_threshold=-1.0,
        generator=torch.Generator().manual_seed(0),
    )
    # more restarts should never do worse than a single short attempt
    assert multi_restart.violations <= single_restart.violations


def test_optimize_candidate_reduces_energy():
    domain = get_domain("sat", n_vars=15, max_clauses=50)
    inst = domain.generate(1, seed=3)[0]
    encoder, energy, _ = build_demo_bundle(domain, seed=3)

    problem_t = domain.encode_problem(inst.problem).unsqueeze(0)
    with torch.no_grad():
        latent = encoder(problem_t)
    init = domain.random_solution(
        inst.problem, generator=torch.Generator().manual_seed(0)
    ).unsqueeze(0)

    result = optimize_candidate(
        energy, problem_t, latent, init, method="adam", lr=0.2, max_iters=100
    )
    assert result.energy_history[-1] <= result.energy_history[0]
