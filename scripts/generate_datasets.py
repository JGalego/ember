"""Generate a (problem, solution) dataset for a domain and save it as JSONL.

Example:
    python scripts/generate_datasets.py --domain sudoku --n 1000 --seed 1 --out data/sudoku_train.jsonl

Note on scale: the spec calls for datasets of "millions" of puzzles. Generation
here is CPU-bound pure Python (fast for SAT/graph coloring/maze, slower for
Sudoku's backtracking solver) and embarrassingly parallel across instances --
for million-scale generation, shard by seed range across worker processes
rather than raising --n on a single run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.datasets import get_domain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, help="sudoku | sat | graph_coloring | maze")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    domain = get_domain(args.domain)
    instances = domain.generate(args.n, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for inst in instances:
            f.write(
                json.dumps({"problem": inst.problem, "solution": inst.solution, "meta": inst.meta})
                + "\n"
            )

    print(f"Wrote {len(instances)} {args.domain} instances to {out_path}")


if __name__ == "__main__":
    main()
