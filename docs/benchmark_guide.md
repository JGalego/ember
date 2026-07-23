# Benchmark guide

`scripts/run_benchmarks.py` compares the EBM solver against four baselines
on a domain, reporting accuracy, mean violations, mean convergence
steps/iterations, and mean runtime (Step 13).

```bash
python scripts/run_benchmarks.py --domain sudoku --n_test 20
python scripts/run_benchmarks.py --domain sudoku --n_test 20 --checkpoint checkpoints/sudoku-mlp-final.ckpt
```

Without `--checkpoint`, the EBM solver uses the same untrained "demo
weights" as the API (see `models/demo_bundle.py`) -- useful for a quick
sanity comparison, but a trained checkpoint is the fair comparison for
actually judging the learned energy head.

## Baselines

| Baseline | Where | Description |
|---|---|---|
| `greedy` | `benchmarks/<domain>/baselines.py` | Domain-specific, single-pass, no backtracking: row-major smallest-valid-digit for Sudoku, per-variable majority-vote for SAT, Welsh-Powell-style degree order for graph coloring, myopic Manhattan-distance walker for mazes. Weak by design -- a real point of comparison should out-perform it easily. |
| `simulated_annealing` | `benchmarks/common.py` | Generic: works through `Domain.random_solution` / `perturb` / `decode_solution` / `verify`, annealing on violation count. |
| `beam_search` | `benchmarks/common.py` | Generic: maintains `beam_width` candidates, expands each with `branching` perturbations per step, keeps the best `beam_width` by violation count. |
| `transformer_baseline` | `benchmarks/common.py` | A small Transformer trained with plain supervised regression (`problem -> solution` in one forward pass, MSE loss against ground truth) -- the "traditional transformer, one-shot, no iterative optimization" point of comparison the whole EBM approach is contrasted against. |
| `bfs_optimal` (maze only) | `benchmarks/maze/baselines.py` | The exact BFS shortest path -- the same algorithm used to generate ground truth. Included as an honest upper bound: the EBM solver isn't expected to beat an exact polynomial-time algorithm on a problem that already has one; the point of the maze domain is to demonstrate the energy/optimization machinery on a different constraint structure (connectivity, not just per-element consistency), not to out-solve BFS.

## Metrics

Defined in `ember/training/metrics.py`'s `SolveRecord` / `aggregate`:

- **accuracy**: fraction of instances where `Domain.verify` returned `True`.
- **mean_violations**: average constraint-violation count (0 for a fully
  valid solution).
- **mean_convergence_steps**: average iterations/steps the solver actually
  ran (multi-start sums across all starts for the EBM solver; a single pass
  = 1 for the transformer baseline).
- **mean_runtime_s**: wall-clock seconds per instance.

## Adding a new baseline

Any callable `(domain, problem_raw) -> SolveRecord` can be dropped into
`GREEDY_BASELINES` (or a new dict) in `scripts/run_benchmarks.py`. If it's
domain-specific, put the implementation in `benchmarks/<domain>/baselines.py`
next to the existing greedy solver; if it's domain-generic (works through
`Domain`'s interface alone, like simulated annealing and beam search),
add it to `benchmarks/common.py` instead.

## A note on scale

The default benchmark run sizes (`--n_test 20`, a handful of SA/beam
steps) are chosen to finish in seconds for CI and quick iteration. For a
statistically meaningful comparison, increase `--n_test`, `--sa_steps`, and
`--beam_steps`, and train the EBM solver (and the transformer baseline,
via `--n_train_baseline`) properly rather than relying on the untrained
demo weights.
