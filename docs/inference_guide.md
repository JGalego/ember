# Inference guide

## The solver

```python
from kona_ebm.datasets import get_domain
from kona_ebm.models import build_demo_bundle  # or load a trained checkpoint
from kona_ebm.inference import solve

domain = get_domain("sudoku", n_remove=45)
instance = domain.generate(1, seed=0)[0]

encoder, energy, decoder = build_demo_bundle(domain)  # untrained demo weights
result = solve(domain, instance.problem, encoder, energy, decoder=decoder)

print(result.valid, result.violations, result.final_energy, result.n_iters)
print(result.solution)  # a 9x9 grid of digits
```

`solve()` (`kona_ebm/inference/solve.py`) runs the full Step 9 pipeline:

1. **Encode**: `latent = encoder(problem)`.
2. **Optimize** (Step 7 + Step 10): `n_starts` random initializations, each
   run through `LatentOptimizer` (SGD/Adam/LBFGS) until convergence or
   `max_iters`; the lowest-energy candidate across starts is kept.
3. **Decode** (Step 8): the optimized candidate goes through the `Decoder`'s
   residual refinement.
4. **Verify** (Step 9): `Domain.decode_solution` (hard argmax/threshold) +
   `Domain.verify` (exact constraint check).
5. **Self-correct** (Step 11): if the decoded candidate isn't valid and its
   energy is still above `energy_threshold`, the whole thing restarts from
   fresh random initializations, up to `max_restarts` times, keeping
   whichever restart had the fewest violations.

## Choosing an optimizer

| `method` | Notes |
|---|---|
| `sgd` | Simple, needs more iterations and a hand-tuned `lr`. |
| `adam` | Good default; adaptive per-coordinate step size handles the very different scales of Sudoku's one-hot logits vs. a maze's sigmoid mask reasonably well. |
| `lbfgs` | Fewer, more expensive iterations (line search); can converge faster on smooth energies but is more sensitive to a non-smooth constraint energy (e.g. the maze's proxy). |

## Multi-start and self-correction in practice

Both matter more than they might look: an untrained "demo" energy (just the
hand-specified constraint penalty, see `models/demo_bundle.py`) already
solves most small Sudoku/graph-coloring/maze instances correctly purely
because gradient descent on a well-formed penalty function is, in effect, a
real (if slow, if locally-stuck-prone) constraint solver. Multi-start
mitigates getting stuck in a bad local minimum; self-correction mitigates
the (rarer, once multi-start is in play) case where every start still
converges to something invalid.

## Lower-level building blocks

`kona_ebm/inference/optimize.py` exposes `optimize_candidate` and
`optimize_multi_start` if you want the optimization step without the full
solve/decode/verify/restart pipeline (e.g. for visualization or debugging):

```python
from kona_ebm.inference.optimize import optimize_candidate

problem_t = domain.encode_problem(instance.problem).unsqueeze(0)
latent = encoder(problem_t)
init = domain.random_solution(instance.problem).unsqueeze(0)

result = optimize_candidate(energy, problem_t, latent, init, method="adam", lr=0.2, max_iters=200, record_trajectory=True)
```

## Visualizing an optimization run

```python
from kona_ebm.inference import plot_energy_curve, plot_latent_trajectory, save_optimization_movie

plot_energy_curve(result.energy_history, "energy.png")
plot_latent_trajectory(result.trajectory, "trajectory.png")       # PCA-projected to 2D
save_optimization_movie(result.trajectory[::4], "movie.gif")       # generic heatmap animation
```

`save_optimization_movie` is domain-agnostic (it reshapes the flattened
candidate into a near-square grid and animates it) -- it's a diagnostic,
not a semantic rendering of e.g. Sudoku digits or graph edges.

## Serving over HTTP

See the API section of the main `README.md` and `kona_ebm/api/main.py`'s
docstring: `POST /solve`, `POST /energy`, `POST /optimize`, all accepting a
`domain` name plus domain-specific raw JSON (a 9x9 grid for Sudoku, a
clause list for SAT, an adjacency matrix for graph coloring, a
`{walls, start, goal}` dict for mazes).
