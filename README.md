# Ember

A research implementation of an **energy-based reasoning framework** for
constraint-satisfaction problems (Sudoku, SAT, graph coloring, mazes),
built around iterative energy minimization instead of autoregressive
decoding.

> **What this is, and isn't.** Ember is inspired by publicly discussed
> energy-based modeling (EBM) concepts -- principally Yann LeCun's EBM
> tutorial and "A Path Towards Autonomous Machine Intelligence" -- and, in
> part, by public descriptions of Logical Intelligence's Kona-1. It's an
> original implementation of ideas from the public EBM literature, not a
> reproduction of, or claim of equivalence to, Kona-1 or any other
> proprietary system. See `REFERENCES.md` for sources.

## The idea

Instead of generating a solution token-by-token, this framework:

1. Learns a scalar **energy function** `E(problem, candidate_solution)`
   (lower = better) over *complete* candidate solutions, combining a
   learned neural energy head with hand-specified, differentiable
   constraint-violation penalties.
2. At inference time, **encodes** the problem, then **optimizes a candidate
   solution directly by gradient descent on the energy** (SGD / Adam /
   LBFGS), from multiple random restarts, until it converges or a
   validity threshold is met -- self-correcting by restarting if the
   decoded candidate still violates constraints.
3. **Decodes** the optimized candidate back into a concrete structure
   (grid, assignment, coloring, path) and verifies it against the
   problem's actual constraints.

```
problem --> [Encoder] --> latent context
                             |
                             v
random init --> [ latent optimization: candidate -= lr * dE/dcandidate ] --> optimized candidate
                             |
                             v
                        [Decoder] --> [Domain.decode_solution] --> [Domain.verify] --> solution
```

See `docs/architecture.md` for the full design, including how this
reconciles the spec's "optimize the latent" framing with the constraint
energy's need for a structured (per-cell / per-variable) candidate
representation.

## What's demonstrated here vs. what needs real compute

Everything in this repo runs end-to-end and is genuinely tested (see
`tests/`, 86 passing tests, ~87% coverage), but it's scaled for a laptop/CI
box, not a training cluster:

- Dataset generators can produce arbitrarily many instances, but the repo's
  own scripts/configs default to thousands, not the "millions" the original
  design brief calls for -- see `scripts/generate_datasets.py` for notes on
  scaling via sharded generation.
- Default model configs are small (hidden dims in the hundreds, a couple of
  Transformer layers) so that `scripts/train.py` finishes in seconds on CPU.
  Scale them up (`model.encoder.hidden_dims=[...]`, more layers/heads) for a
  real training run.
- FSDP / DDP / mixed precision / `torch.compile` are wired through
  `configs/training/default.yaml` and `scripts/train.py`, but this repo's
  own CI only exercises the single-device CPU path -- multi-GPU scaling
  hasn't been benchmarked here.
- The FastAPI service (`ember/api/main.py`) serves **untrained demo
  weights** by default (a fixed-seed MLP stack with the constraint energy
  weighted heavily), which is enough to solve small real instances out of
  the box -- point it at a real checkpoint via
  `EMBER_CHECKPOINT_<DOMAIN>` for trained weights.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# generate a small dataset
python scripts/generate_datasets.py --domain sudoku --n 1000 --seed 1 --out data/sudoku_train.jsonl

# train (Hydra config, override anything on the command line)
python scripts/train.py dataset=sudoku model=mlp loss=contrastive training.max_epochs=5

# evaluate a trained checkpoint's solve-time accuracy
python scripts/evaluate.py --domain sudoku --checkpoint checkpoints/sudoku-mlp-final.ckpt --n 50

# compare the EBM solver against greedy / simulated annealing / beam search / a transformer baseline
python scripts/run_benchmarks.py --domain sudoku --n_test 20

# serve the API (untrained demo weights unless EMBER_CHECKPOINT_SUDOKU etc. is set)
python scripts/serve_api.py --port 8000
# or: uvicorn ember.api.main:app --reload

# run the test suite
pytest tests/ -q
```

Docker:

```bash
docker compose up api      # serves the FastAPI app on :8000
docker compose run train   # runs scripts/train.py dataset=sudoku inside the container
```

## Project layout

```
ember/
├── datasets/       # Domain interface + sudoku/sat/graph_coloring/maze generators
├── models/         # encoder.py, energy_model.py, decoder.py, optimizer.py, losses.py
├── training/       # dataset.py, trainer.py (Lightning module), metrics.py
├── inference/       # solve.py, optimize.py, visualize.py
└── api/            # FastAPI app: /solve, /energy, /optimize

configs/            # Hydra configs: dataset/, model/, optimizer/, loss/, training/
benchmarks/         # baselines (greedy, SA, beam search, transformer) per domain + shared harness
scripts/            # generate_datasets.py, train.py, evaluate.py, run_benchmarks.py, serve_api.py
tests/              # pytest suite covering datasets/encoder/decoder/energy/optimizer/losses/solve/benchmarks/api
docs/               # architecture.md, training_guide.md, inference_guide.md, benchmark_guide.md
```

Every model component (encoder kind, energy kind, decoder kind, loss,
latent-optimization method) is swappable through Hydra config alone --
nothing is hardcoded. See `docs/training_guide.md` for the full list of
overrides.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) -- design rationale, how the
  pieces fit together, and the "latent = candidate solution" design decision
- [`docs/training_guide.md`](docs/training_guide.md) -- Hydra configs, losses,
  contrastive training details
- [`docs/inference_guide.md`](docs/inference_guide.md) -- latent optimization,
  multi-start, self-correction, the solver API
- [`docs/benchmark_guide.md`](docs/benchmark_guide.md) -- baselines, metrics,
  how to run comparisons
- [`CONTRIBUTING.md`](CONTRIBUTING.md) -- adding a new CSP domain, code style,
  running tests
- [`REFERENCES.md`](REFERENCES.md) -- full citations

## License

MIT. See `LICENSE`.
