import pytest
import torch

from kona_ebm.datasets import available_domains, get_domain


@pytest.mark.parametrize("domain_name", available_domains())
def test_generate_and_ground_truth_is_valid(domain_name):
    domain = get_domain(domain_name)
    instances = domain.generate(5, seed=42)
    assert len(instances) == 5
    for inst in instances:
        valid, violations = domain.verify(inst.problem, inst.solution)
        assert valid, f"{domain_name} ground-truth solution has {violations} violations"


@pytest.mark.parametrize("domain_name", available_domains())
def test_encode_shapes(domain_name):
    domain = get_domain(domain_name)
    inst = domain.generate(1, seed=1)[0]
    problem_t = domain.encode_problem(inst.problem)
    solution_t = domain.encode_solution(inst.solution)
    assert problem_t.shape == (domain.problem_dim,)
    assert solution_t.shape == (domain.solution_dim,)
    assert torch.isfinite(problem_t).all()
    assert torch.isfinite(solution_t).all()


@pytest.mark.parametrize("domain_name", available_domains())
def test_decode_round_trip_recovers_ground_truth(domain_name):
    domain = get_domain(domain_name)
    inst = domain.generate(1, seed=2)[0]
    solution_t = domain.encode_solution(inst.solution)
    decoded = domain.decode_solution(inst.problem, solution_t)
    valid, violations = domain.verify(inst.problem, decoded)
    assert (
        valid
    ), f"{domain_name} encode->decode round trip broke ground truth: {violations} violations"


@pytest.mark.parametrize("domain_name", available_domains())
def test_random_solution_shape(domain_name):
    domain = get_domain(domain_name)
    inst = domain.generate(1, seed=3)[0]
    generator = torch.Generator().manual_seed(0)
    rnd = domain.random_solution(inst.problem, generator=generator)
    assert rnd.shape == (domain.solution_dim,)


@pytest.mark.parametrize("domain_name", available_domains())
def test_perturb_changes_the_tensor(domain_name):
    domain = get_domain(domain_name)
    inst = domain.generate(1, seed=4)[0]
    solution_t = domain.encode_solution(inst.solution)
    generator = torch.Generator().manual_seed(0)
    perturbed = domain.perturb(solution_t, generator=generator)
    assert perturbed.shape == solution_t.shape
    assert not torch.equal(perturbed, solution_t)


@pytest.mark.parametrize("domain_name", available_domains())
def test_generation_is_deterministic_given_seed(domain_name):
    domain = get_domain(domain_name)
    a = domain.generate(3, seed=99)
    b = domain.generate(3, seed=99)
    for inst_a, inst_b in zip(a, b, strict=True):
        assert inst_a.problem == inst_b.problem
        assert inst_a.solution == inst_b.solution
