import pytest
import torch

from kona_ebm.datasets import available_domains, get_domain
from kona_ebm.models.energy_model import (
    CompositeEnergy,
    DeepSetsEnergy,
    build_constraint_energy,
    build_energy_model,
)

LATENT_DIM = 16
SOLUTION_DIM = 40
BATCH = 4


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_learned_energy_output_shape(kind):
    energy = build_energy_model(kind, latent_dim=LATENT_DIM, solution_dim=SOLUTION_DIM)
    latent = torch.randn(BATCH, LATENT_DIM)
    candidate = torch.randn(BATCH, SOLUTION_DIM)
    e = energy(latent, candidate)
    assert e.shape == (BATCH,)
    assert torch.isfinite(e).all()


def test_deep_sets_energy_requires_divisible_solution_dim():
    with pytest.raises(ValueError):
        DeepSetsEnergy(latent_dim=LATENT_DIM, solution_dim=SOLUTION_DIM, n_elements=7)


def test_deep_sets_energy_is_permutation_invariant_over_elements():
    n_elements = 8
    energy = DeepSetsEnergy(latent_dim=LATENT_DIM, solution_dim=SOLUTION_DIM, n_elements=n_elements)
    energy.eval()
    latent = torch.randn(1, LATENT_DIM)
    candidate = torch.randn(1, SOLUTION_DIM)
    element_dim = SOLUTION_DIM // n_elements
    perm = torch.randperm(n_elements)
    permuted = candidate.view(1, n_elements, element_dim)[:, perm].reshape(1, SOLUTION_DIM)
    with torch.no_grad():
        e1 = energy(latent, candidate)
        e2 = energy(latent, permuted)
    assert torch.allclose(e1, e2, atol=1e-5)


@pytest.mark.parametrize("domain_name", available_domains())
def test_constraint_energy_penalizes_perturbation_more_than_ground_truth(domain_name):
    domain = get_domain(domain_name)
    inst = domain.generate(3, seed=5)
    constraint = build_constraint_energy(domain_name, domain)

    problems = torch.stack([domain.encode_problem(i.problem) for i in inst])
    solutions = torch.stack([domain.encode_solution(i.solution) for i in inst])
    generator = torch.Generator().manual_seed(0)
    negatives = torch.stack([domain.perturb(s, noise=2.0, generator=generator) for s in solutions])

    e_pos = constraint(problems, solutions)
    e_neg = constraint(problems, negatives)
    assert torch.isfinite(e_pos).all() and torch.isfinite(e_neg).all()
    assert (e_pos >= 0).all()
    # a heavily-perturbed candidate should look worse, on average, than ground truth
    assert e_neg.mean().item() >= e_pos.mean().item()


def test_composite_energy_combines_learned_and_constraint_with_weights():
    domain = get_domain("graph_coloring")
    learned = build_energy_model("mlp", latent_dim=LATENT_DIM, solution_dim=domain.solution_dim)
    constraint = build_constraint_energy("graph_coloring", domain)
    composite = CompositeEnergy(
        learned=learned, constraint=constraint, w_learned=0.0, w_constraint=1.0
    )

    inst = domain.generate(1, seed=1)[0]
    problem = domain.encode_problem(inst.problem).unsqueeze(0)
    solution = domain.encode_solution(inst.solution).unsqueeze(0)
    latent = torch.randn(1, LATENT_DIM)

    with torch.no_grad():
        e_composite = composite(problem, latent, solution)
        e_constraint_only = constraint(problem, solution)
    assert torch.allclose(e_composite, e_constraint_only)


def test_composite_energy_requires_at_least_one_term():
    with pytest.raises(ValueError):
        CompositeEnergy(learned=None, constraint=None)
