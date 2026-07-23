# Contributing

This is a research scaffold: the goal is to make experimentation easy, so
contributions that add a new domain, encoder/energy/decoder architecture,
loss, or baseline are especially welcome.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
ruff check .
black --check .
```

## Adding a new CSP domain

Implement the `Domain` ABC (`ember/datasets/domain.py`) in a new module
under `ember/datasets/`:

```python
from ember.datasets.domain import Domain, ProblemInstance, register_domain

@register_domain
class MyDomain(Domain):
    name = "my_domain"

    def __init__(self, ...):
        self.problem_dim = ...   # flattened problem tensor size
        self.solution_dim = ...  # flattened continuous-relaxation candidate size

    def generate(self, n, seed): ...          # -> list[ProblemInstance], deterministic given seed
    def encode_problem(self, problem): ...    # raw -> Tensor(problem_dim,)
    def encode_solution(self, solution): ...  # raw -> Tensor(solution_dim,), continuous relaxation
    def decode_solution(self, problem, tensor): ...  # Tensor(solution_dim,) -> raw structure
    def verify(self, problem, solution): ...  # -> (is_valid: bool, num_violations: int)
    def random_solution(self, problem, generator=None): ...  # -> Tensor(solution_dim,)
    # `perturb` has a sensible Gaussian-noise default; override for structured negatives
```

Then, to make it a full citizen of the repo:

1. Import it in `ember/datasets/__init__.py` (so `@register_domain` runs
   and `get_domain("my_domain")` works).
2. Add `ConstraintEnergy` support in `ember/models/energy_model.py`:
   a differentiable penalty class, plus a branch in `build_constraint_energy`.
3. Add a Hydra dataset config: `configs/dataset/my_domain.yaml`.
4. (Optional but recommended) add a greedy baseline under
   `benchmarks/my_domain/baselines.py`, following the pattern in
   `benchmarks/sudoku/baselines.py` etc.
5. Add it to the `test_datasets.py`/`test_energy_model.py` parametrizations
   (they already iterate `available_domains()`, so a registered domain gets
   picked up automatically -- just double check any domain-specific
   assumptions in new tests you add).

Everything downstream (encoders, energy models, decoders, the optimizer,
the solver, the API, the benchmark harness) works with any `Domain` that
implements the interface -- that's the point of the abstraction.

## Adding a new encoder / energy / decoder / loss

Each of these has a small registry + `build_*` factory
(`build_encoder`, `build_energy_model`, `build_decoder`, `build_loss`).
Add your class, register it in the module's `_XXX` dict, and it's
immediately usable from Hydra configs (`model.encoder.kind=my_new_encoder`).

## Code style

- Type hints on public functions/methods.
- Docstrings only where the *why* isn't obvious from the code (see the
  repo's own modules for the target density -- most functions have none,
  classes and modules get a short one).
- `black` (line length 100) and `ruff` (see `pyproject.toml` for the
  enabled rule set) must pass clean.
- Deterministic by default: dataset generators take an explicit `seed`;
  tests that need randomness use an explicit `torch.Generator`, never the
  global RNG, so they don't flake based on test execution order.

## Tests

`tests/` mirrors the module list from the design brief: datasets, encoder,
decoder, energy model, optimizer, losses, plus end-to-end solve and API
tests. New functionality should come with tests in the matching file (or a
new one, for a genuinely new module).
