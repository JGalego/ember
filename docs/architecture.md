# Architecture

## Overview

```
                    ┌─────────────┐
   problem  ─────►  │   Encoder   │ ─────► latent (context vector)
                    └─────────────┘              │
                                                  │ conditions
                                                  ▼
   init candidate ─────────────────────►  ┌──────────────┐
   (random, in continuous                 │ CompositeEnergy│ ◄── problem
    solution space)                       │ = w1*Learned   │
        │                                 │ + w2*Constraint│
        │      gradient descent           └──────────────┘
        │      candidate -= lr * dE/dcandidate     │
        └──────────────◄────────────────────────────┘
                        │  (repeat until converged / multi-start / self-correct)
                        ▼
                 optimized candidate
                        │
                        ▼
                 ┌─────────────┐
                 │   Decoder   │  (residual refinement)
                 └─────────────┘
                        │
                        ▼
             Domain.decode_solution()  (hard argmax / threshold)
                        │
                        ▼
                Domain.verify()  -->  (valid?, violations)
```

## A design decision worth calling out explicitly

The design brief's Step 7 pseudocode reads:

```
latent = encoder(problem)
repeat:
    energy = E(problem, latent)
    latent = latent - alpha * grad(E)
decoder converts latent to solution
```

Read literally, this treats "latent" as both the encoder's output *and* the
thing iteratively optimized, then decoded. But Step 4's energy signature is
`E(problem, candidate_solution)`, and Step 5's constraint penalties (row/
column/box duplicates for Sudoku, clause satisfaction for SAT, same-color
edges for graph coloring, wall/connectivity for mazes) need to operate on a
*structured* representation of the candidate -- e.g. Sudoku's per-cell
one-hot digit logits, reshaped to (81, 9) -- not an arbitrary bottleneck
vector the encoder happens to produce.

This repo resolves that by choosing, deliberately:

- **The object being optimized (the "latent" of Step 7) is the candidate
  solution's own continuous relaxation** -- exactly the space
  `Domain.encode_solution` / `Domain.random_solution` / `Domain.perturb`
  operate in (per-cell logits, a tanh-space assignment, per-node color
  logits, a path-membership mask). This is what `models/optimizer.py`'s
  `LatentOptimizer` and `inference/solve.py` actually optimize.
- **The `Encoder` (Step 3) produces a separate conditioning context
  vector**, `latent_dim`-sized, used two ways: (a) it's concatenated into
  the learned `EnergyModel`'s input so the energy is problem-aware, and
  (b) nothing structurally prevents using it to seed a smarter
  initialization, though the current solver uses random multi-start inits
  for the (documented, textbook) reason that random restarts are what let
  `multi_start_optimize` escape different local minima.
- **The `Decoder` (Step 8) is a residual post-processing network**:
  `output = candidate + f(candidate, latent_context)`. It refines the
  optimized candidate before the domain's hard, non-differentiable
  `decode_solution` (argmax / threshold) turns it into a checkable
  structure.

This is the same shape of design used throughout the EBM-for-structured-
prediction literature (LeCun et al.'s tutorial, section on energy-based
inference: `argmin_y E(x, y)` where `y` is the output variable itself,
decoded afterward if it's a continuous relaxation of something discrete)
-- not a deviation invented to dodge the ambiguity, but the standard
reading once "candidate_solution" (Step 4) and "latent" (Step 7) are forced
to be the same object.

## Components

| Module | Responsibility |
|---|---|
| `datasets/domain.py` | `Domain` ABC: `generate`, `encode_problem`, `encode_solution`, `decode_solution`, `verify`, `random_solution`, `perturb`. Every CSP domain is a plugin implementing this interface. |
| `datasets/{sudoku,sat,graph_coloring,maze}.py` | Concrete domains, each with a planted-solution or backtracking generator. |
| `models/encoder.py` | `Encoder` ABC + `MLPEncoder` / `TransformerEncoder` / `GNNEncoder`, all `problem -> latent`. |
| `models/energy_model.py` | `EnergyModel` ABC (`MLPEnergy` / `TransformerEnergy` / `DeepSetsEnergy`, learned), `ConstraintEnergy` ABC (one differentiable penalty per domain), and `CompositeEnergy` combining both with configurable weights. |
| `models/decoder.py` | `Decoder` ABC + `MLPDecoder` / `TransformerDecoder`, residual refinement of the optimized candidate. |
| `models/optimizer.py` | `LatentOptimizer` (SGD / Adam / LBFGS inner loop) + `multi_start_optimize`. |
| `models/losses.py` | `contrastive_loss`, `margin_ranking_loss`, `info_nce_loss`, `hinge_loss` -- all `(e_pos, e_neg) -> scalar`. |
| `training/trainer.py` | `EBMLightningModule` (contrastive training) + `PlainEBMTrainer` (plain-PyTorch fallback). |
| `inference/solve.py` | The full pipeline: encode -> multi-start optimize -> decode -> verify -> restart-if-invalid. |
| `inference/visualize.py` | Energy curves, PCA'd candidate trajectories, optimization "movies". |
| `benchmarks/` | Greedy / simulated annealing / beam search / transformer baselines, one comparison harness. |
| `ember/api/main.py` | FastAPI `/solve`, `/energy`, `/optimize`. |

## Why a custom GNN instead of PyTorch Geometric

`GNNEncoder` is a small hand-rolled message-passing layer (mean-aggregate
over an adjacency matrix, or a fully-connected pseudo-graph if none is
given) rather than a PyTorch Geometric model. This keeps the dependency
footprint light for a research scaffold; PyG (cited in `REFERENCES.md`) is
the natural upgrade path if you need production-grade GNN architectures
(GAT, GraphSAGE, etc.).

## Why the maze's constraint energy is a proxy, not exact

Sudoku/SAT/graph-coloring constraints (duplicate values, unsatisfied
clauses, same-color edges) have natural smooth relaxations. Path
*connectivity* does not -- "is this set of cells a connected path from A to
B" is a discrete graph property with no obvious gradient. `MazeConstraintEnergy`
uses a differentiable proxy (wall avoidance + endpoint inclusion + a local
neighbor-count term that discourages isolated selected cells) to *guide*
optimization, and leans on `Domain.verify`'s exact BFS-based connectivity
check plus `inference/solve.py`'s self-correction (restart-if-invalid) to
catch what the proxy misses.
