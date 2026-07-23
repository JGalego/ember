# Training guide

Training is a Hydra + Lightning entrypoint: `scripts/train.py`. Every piece
is a config group under `configs/`, composed via Hydra's `defaults` list and
overridable on the command line.

## Basic usage

```bash
python scripts/train.py
python scripts/train.py dataset=sat model=transformer loss=info_nce
python scripts/train.py dataset=sudoku training.max_epochs=20 training.batch_size=64
```

## Config groups

| Group | Options | What it controls |
|---|---|---|
| `dataset` | `sudoku`, `sat`, `graph_coloring`, `maze` | Which `Domain`, how many train/val instances, domain constructor params (`n_remove`, `n_vars`, `n_nodes`, `height`/`width`, ...). |
| `model` | `mlp`, `transformer`, `gnn`, `deep_sets` | Encoder/energy/decoder kinds and their hyperparameters. |
| `loss` | `contrastive`, `margin_ranking`, `info_nce`, `hinge` | Which contrastive loss trains the energy (see `models/losses.py`). |
| `optimizer` | `sgd`, `adam`, `lbfgs` | **Not** the training optimizer (always Adam, set via top-level `lr`) -- this configures the *inference-time* latent optimizer used by `scripts/evaluate.py` / `scripts/run_benchmarks.py`. |
| `training` | `default` | Lightning `Trainer` passthrough: epochs, batch size, precision, accelerator/devices/strategy, gradient clipping, `torch.compile`, W&B toggle. |

Root-level knobs in `configs/config.yaml`: `seed`, `use_constraint_energy`,
`w_learned` / `w_constraint` (how the composite energy weighs the learned
head vs. the hand-specified constraint penalty), `negatives_per_positive`,
`perturb_noise`, `lr`.

## Model/dataset compatibility

Most combinations just work. Two exceptions:

- **`model=deep_sets`**: `DeepSetsEnergy` requires `solution_dim %
  n_elements == 0`, and `n_elements` is domain-specific. Override it:
  ```bash
  python scripts/train.py dataset=sudoku model=deep_sets model.energy.n_elements=81
  python scripts/train.py dataset=graph_coloring model=deep_sets model.energy.n_elements=12
  python scripts/train.py dataset=sat model=deep_sets model.energy.n_elements=20
  python scripts/train.py dataset=maze model=deep_sets model.energy.n_elements=49
  ```
- **`model=gnn`**: the encoder's message passing defaults to a
  fully-connected pseudo-graph unless an explicit adjacency matrix is
  passed in (only wired up for `graph_coloring` in this repo's training
  loop -- other domains still work, just without a meaningful adjacency
  structure).

## What actually gets trained

`EBMLightningModule` (`kona_ebm/training/trainer.py`) wraps an `Encoder` and
a `CompositeEnergy` (learned + optional constraint energy). Each training
step:

1. Encodes the problem batch: `latent = encoder(problem)`.
2. Computes `e_pos = energy(problem, latent, ground_truth_solution)`.
3. Perturbs the ground truth (`Domain.perturb`, `negatives_per_positive`
   times) to build negatives, and computes `e_neg = energy(problem, latent,
   negative)` for each.
4. Applies the configured loss to `(e_pos, e_neg)` and backpropagates.

Validation mirrors this and additionally logs the mean energy gap
`E(neg) - E(pos)` -- watch this to sanity-check that the model is actually
learning to separate valid from invalid solutions, independent of the loss
value's absolute scale.

## Scaling (Step 14)

`configs/training/default.yaml` wires through Lightning's `accelerator`,
`devices`, `strategy` (`ddp`/`fsdp`), `precision` (`16-mixed`/`bf16-mixed`),
`gradient_clip_val`, `accumulate_grad_batches`, and a `compile: true` flag
that wraps the encoder/energy submodules with `torch.compile`. These are
wired through and functional on a single device; this repo's own CI only
exercises the single-CPU path, so treat multi-GPU/FSDP as untested-at-scale
plumbing, not a benchmarked result.

Weights & Biases logging is opt-in (`training.use_wandb=true`,
`training.wandb_project=...`); it falls back to a `CSVLogger` if `wandb`
isn't installed or you haven't run `wandb login`.

## Checkpoints

`trainer.save_checkpoint(...)` writes a standard Lightning checkpoint to
`{checkpoint_dir}/{dataset}-{encoder_kind}-final.ckpt`. Load it with
`EBMLightningModule.load_from_checkpoint(path, domain=domain)` (the
`domain` argument is required since it's excluded from the saved
hyperparameters -- it's a Python object, not something you'd want
serialized into a YAML hparams file).
