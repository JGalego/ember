"""Evaluate a trained checkpoint's solve-time accuracy, violations, convergence
steps, and runtime on fresh test instances (uses the Step 9 solver + Step 13
metrics; see scripts/run_benchmarks.py for baseline comparisons).

Example:
    python scripts/evaluate.py --domain sudoku --checkpoint checkpoints/sudoku-mlp-final.ckpt --n 50
"""

from __future__ import annotations

import argparse

from ember.datasets import get_domain
from ember.inference import solve
from ember.training import EBMLightningModule
from ember.training.metrics import SolveRecord, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--method", default="adam", choices=["sgd", "adam", "lbfgs"])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--n_starts", type=int, default=4)
    parser.add_argument("--max_restarts", type=int, default=3)
    args = parser.parse_args()

    domain = get_domain(args.domain)
    module = EBMLightningModule.load_from_checkpoint(
        args.checkpoint, domain=domain, map_location="cpu"
    )
    module.eval()

    instances = domain.generate(args.n, args.seed)
    records = []
    for inst in instances:
        result = solve(
            domain,
            inst.problem,
            module.encoder,
            module.energy,
            decoder=None,
            method=args.method,
            lr=args.lr,
            max_iters=args.max_iters,
            n_starts=args.n_starts,
            max_restarts=args.max_restarts,
        )
        records.append(
            SolveRecord(
                solved=result.valid,
                violations=result.violations,
                n_iters=result.n_iters,
                runtime_s=result.runtime_s,
            )
        )

    metrics = aggregate(records)
    print(f"Domain: {args.domain}  Checkpoint: {args.checkpoint}  N: {metrics['n']}")
    for key, value in metrics.items():
        if key != "n":
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
