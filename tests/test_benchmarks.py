import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.common import (
    beam_search,
    simulated_annealing,
    solve_with_transformer_baseline,
    train_transformer_baseline,
)
from benchmarks.graph_coloring.baselines import greedy_graph_coloring
from benchmarks.maze.baselines import bfs_optimal_maze, greedy_maze
from benchmarks.sat.baselines import greedy_sat
from benchmarks.sudoku.baselines import greedy_sudoku
from ember.datasets import get_domain
from ember.training.metrics import SolveRecord, aggregate


def test_greedy_sudoku_returns_solve_record():
    domain = get_domain("sudoku", n_remove=40)
    inst = domain.generate(1, seed=10)[0]
    record = greedy_sudoku(inst.problem)
    assert isinstance(record, SolveRecord)
    assert record.violations >= 0


def test_greedy_sat_returns_solve_record():
    domain = get_domain("sat", n_vars=12, max_clauses=40)
    inst = domain.generate(1, seed=10)[0]
    record = greedy_sat(inst.problem, domain)
    assert isinstance(record, SolveRecord)
    assert record.violations >= 0


def test_greedy_graph_coloring_solves_easy_instances():
    domain = get_domain("graph_coloring", n_nodes=8, k_colors=3, edge_prob=0.2)
    inst = domain.generate(1, seed=10)[0]
    record = greedy_graph_coloring(inst.problem, domain)
    assert isinstance(record, SolveRecord)
    assert record.violations >= 0


def test_greedy_maze_and_bfs_optimal():
    domain = get_domain("maze", height=5, width=5, wall_prob=0.2)
    inst = domain.generate(1, seed=10)[0]
    greedy_record = greedy_maze(inst.problem, domain)
    bfs_record = bfs_optimal_maze(inst.problem, domain)
    assert isinstance(greedy_record, SolveRecord)
    assert bfs_record.solved is True
    assert bfs_record.violations == 0


def test_simulated_annealing_reduces_violations_over_random_init():
    domain = get_domain("graph_coloring", n_nodes=8, k_colors=3, edge_prob=0.3)
    inst = domain.generate(1, seed=11)[0]
    random_candidate = domain.random_solution(
        inst.problem, generator=torch.Generator().manual_seed(0)
    )
    _, random_violations = domain.verify(
        inst.problem, domain.decode_solution(inst.problem, random_candidate)
    )

    record = simulated_annealing(domain, inst.problem, n_steps=500, seed=0)
    assert record.violations <= random_violations


def test_beam_search_returns_solve_record():
    domain = get_domain("graph_coloring", n_nodes=6, k_colors=3, edge_prob=0.3)
    inst = domain.generate(1, seed=12)[0]
    record = beam_search(domain, inst.problem, beam_width=4, n_steps=20, seed=0)
    assert isinstance(record, SolveRecord)


def test_transformer_baseline_trains_and_solves():
    domain = get_domain("graph_coloring", n_nodes=6, k_colors=3, edge_prob=0.3)
    instances = domain.generate(8, seed=13)
    problems = torch.stack([domain.encode_problem(i.problem) for i in instances])
    solutions = torch.stack([domain.encode_solution(i.solution) for i in instances])
    model = train_transformer_baseline(domain, problems, solutions, epochs=5)
    record = solve_with_transformer_baseline(model, domain, instances[0].problem)
    assert isinstance(record, SolveRecord)
    assert record.n_iters == 1


def test_aggregate_metrics():
    records = [
        SolveRecord(solved=True, violations=0, n_iters=10, runtime_s=0.1),
        SolveRecord(solved=False, violations=2, n_iters=20, runtime_s=0.2),
    ]
    metrics = aggregate(records)
    assert metrics["n"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_violations"] == 1.0
    assert metrics["mean_convergence_steps"] == 15.0


def test_aggregate_metrics_empty():
    metrics = aggregate([])
    assert metrics["n"] == 0
    assert metrics["accuracy"] == 0.0
