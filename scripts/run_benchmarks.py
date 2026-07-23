"""Compare the EBM solver against greedy / simulated annealing / beam search /
transformer-baseline on a domain (Step 13).

Example:
    python scripts/run_benchmarks.py --domain sudoku --n_test 20
    python scripts/run_benchmarks.py --domain sudoku --n_test 20 --checkpoint checkpoints/sudoku-mlp-final.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # so `benchmarks` (not pip-installed) is importable

from benchmarks.common import (
    beam_search,
    simulated_annealing,
    solve_with_transformer_baseline,
    summarize,
    train_transformer_baseline,
)
from benchmarks.graph_coloring.baselines import greedy_graph_coloring
from benchmarks.maze.baselines import bfs_optimal_maze, greedy_maze
from benchmarks.sat.baselines import greedy_sat
from benchmarks.sudoku.baselines import greedy_sudoku
from kona_ebm.datasets import get_domain
from kona_ebm.inference import solve
from kona_ebm.models import build_demo_bundle
from kona_ebm.training import EBMLightningModule
from kona_ebm.training.metrics import SolveRecord

GREEDY_BASELINES = {
    "sudoku": lambda domain, problem: greedy_sudoku(problem),
    "sat": lambda domain, problem: greedy_sat(problem, domain),
    "graph_coloring": lambda domain, problem: greedy_graph_coloring(problem, domain),
    "maze": lambda domain, problem: greedy_maze(problem, domain),
}

EXTRA_BASELINES = {
    "maze": {"bfs_optimal": lambda domain, problem: bfs_optimal_maze(problem, domain)},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--n_test", type=int, default=20)
    parser.add_argument(
        "--n_train_baseline",
        type=int,
        default=64,
        help="instances used to fit the transformer baseline",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="EBM Lightning checkpoint; omit to use untrained demo weights",
    )
    parser.add_argument("--sa_steps", type=int, default=1000)
    parser.add_argument("--beam_steps", type=int, default=100)
    parser.add_argument("--ebm_max_iters", type=int, default=200)
    parser.add_argument("--ebm_n_starts", type=int, default=4)
    args = parser.parse_args()

    domain = get_domain(args.domain)
    test_instances = domain.generate(args.n_test, args.seed)

    if args.checkpoint:
        module = EBMLightningModule.load_from_checkpoint(
            args.checkpoint, domain=domain, map_location="cpu"
        )
        encoder, energy = module.encoder, module.energy
        decoder = None
    else:
        encoder, energy, decoder = build_demo_bundle(domain)
        print("No --checkpoint given: EBM solver is using untrained demo weights (see README).")

    import torch

    train_instances = domain.generate(args.n_train_baseline, args.seed + 1)
    train_problems = torch.stack([domain.encode_problem(i.problem) for i in train_instances])
    train_solutions = torch.stack([domain.encode_solution(i.solution) for i in train_instances])
    tf_baseline = train_transformer_baseline(domain, train_problems, train_solutions, epochs=100)

    records: dict[str, list[SolveRecord]] = {
        "ebm": [],
        "greedy": [],
        "simulated_annealing": [],
        "beam_search": [],
        "transformer_baseline": [],
    }
    for name in EXTRA_BASELINES.get(args.domain, {}):
        records[name] = []

    for i, inst in enumerate(test_instances):
        result = solve(
            domain,
            inst.problem,
            encoder,
            energy,
            decoder=decoder,
            max_iters=args.ebm_max_iters,
            n_starts=args.ebm_n_starts,
        )
        records["ebm"].append(
            SolveRecord(
                solved=result.valid,
                violations=result.violations,
                n_iters=result.n_iters,
                runtime_s=result.runtime_s,
            )
        )
        records["greedy"].append(GREEDY_BASELINES[args.domain](domain, inst.problem))
        records["simulated_annealing"].append(
            simulated_annealing(domain, inst.problem, n_steps=args.sa_steps, seed=args.seed + i)
        )
        records["beam_search"].append(
            beam_search(domain, inst.problem, n_steps=args.beam_steps, seed=args.seed + i)
        )
        records["transformer_baseline"].append(
            solve_with_transformer_baseline(tf_baseline, domain, inst.problem)
        )
        for name, fn in EXTRA_BASELINES.get(args.domain, {}).items():
            records[name].append(fn(domain, inst.problem))

    summary = summarize(records)
    header = (
        f"{'solver':<22}{'accuracy':>10}{'mean_viol':>12}{'mean_iters':>12}{'mean_runtime_s':>16}"
    )
    print(f"\n=== {args.domain} (n={args.n_test}) ===")
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(
            f"{name:<22}{m['accuracy']:>10.2%}{m['mean_violations']:>12.2f}"
            f"{m['mean_convergence_steps']:>12.1f}{m['mean_runtime_s']:>16.4f}"
        )


if __name__ == "__main__":
    main()
