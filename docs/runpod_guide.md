# Training on RunPod (single GPU)

Ember's default configs are sized to train in seconds on CPU (see
`docs/training_guide.md`). This guide covers actually training at a
meaningful scale on a rented GPU. A single RTX 4090 (~$0.35-0.50/hr
Community Cloud) or A40/A10 (~$0.50-0.80/hr, more VRAM headroom) is plenty
for these model sizes -- there's no need for multi-GPU here.

Two ways to get the code onto the pod:

- **Option 2 (recommended): pull the prebuilt image from GHCR.** Cleaner --
  the exact environment is pinned, no `pip install` on the pod, and restarts
  are instant.
- **Option 1: clone the repo directly onto a stock PyTorch pod.** Simpler
  for a single throwaway run, no registry/image step at all.

## Option 2: prebuilt image from GHCR

`.github/workflows/docker-publish.yml` builds `Dockerfile` and pushes it to
GitHub Container Registry on every push to `main` (and on `v*` tags, or
manually via "Run workflow" in the Actions tab).

1. **Trigger a build.** Push to `main`, or run the workflow manually from
   the GitHub Actions tab. It publishes `ghcr.io/<owner>/<repo>:latest`
   (lowercased) plus a `:<short-sha>` tag.
2. **Make the package pullable (first time only).** Packages published via
   `GITHUB_TOKEN` default to *private* even on a public repo. Either:
   - Go to `https://github.com/<owner>/<repo>/pkgs/container/<repo>` ->
     Package settings -> Danger Zone -> Change visibility -> **Public**, or
   - Keep it private and add a registry credential in RunPod instead (see
     step 3 below -- Settings -> Registry Credentials).
3. **Create a Template pointing at the image.** RunPod deploys pods *from
   Templates* -- there's no separate "custom image" toggle at deploy time.
   Console -> **Templates** -> **New Template**:
   - **Container Image**: `ghcr.io/<owner>/<repo>:latest`
   - **Container Disk**: ~20 GB is enough for the image + a modest dataset.
   - **Volume Disk** + **Volume Mount Path** (e.g. `/workspace`): attach a
     volume if you want checkpoints/datasets to survive the pod stopping --
     the container disk itself is ephemeral.
   - **Container Start Command** (may be under an "Advanced" toggle): the
     image's default `CMD` serves the FastAPI app; for training, override it
     with something like (**one line, no backslashes** -- see the note below,
     and see "Troubleshooting: pod keeps re-running the job" for why this
     ends in `; sleep infinity`):
     ```
     python scripts/train.py dataset=sudoku training.max_epochs=50 training.accelerator=gpu training.devices=1 training.precision=16 checkpoint_dir=/workspace/checkpoints ; sleep infinity
     ```
   - If the image is private: Console -> **Settings -> Registry
     Credentials** -> add one (name + your GitHub username + a PAT scoped
     `read:packages`), then select that credential in the template.
   - Save the template.
4. **Deploy from that template.** Pods -> Deploy -> pick a GPU (RTX 4090 /
   A40 / A10) -> in the template picker, search for the template you just
   created (it shows up under "My Templates", not the official RunPod list).
5. **Watch it run.** Tail logs from the RunPod console. When training
   finishes, pull the checkpoint off (RunPod's file browser, `scp`, or push
   it to S3/GCS from inside the pod) before you terminate -- GPU time bills
   by the hour, so stop the pod promptly once you have what you need.

## Option 1: clone the repo on a stock PyTorch template

No registry/image step -- good for a one-off run.

1. Deploy a pod from RunPod's "RunPod PyTorch 2.x" template, same GPU choice
   as above.
2. Open a terminal (web terminal or SSH).
3. ```bash
   git clone https://github.com/<owner>/<repo> && cd <repo>
   pip install -e .
   ```
4. ```bash
   python scripts/train.py dataset=sudoku training.max_epochs=50 training.accelerator=gpu training.devices=1 training.precision=16
   ```
   Running this from an interactive terminal doesn't hit the restart-loop
   issue below (that's specific to a pod's *Container Start Command*), so no
   `sleep infinity` needed here.
5. Same persistence caveat as above: attach/use a RunPod Volume for
   checkpoints, or copy them off before terminating the pod.

## Scaling up for a real run

The CPU-sized defaults in `configs/` are for quick local iteration, not
this run. Now that GPU time is available, bump:

- `dataset.n_train=...` / `dataset.n_val=...` -- generate a real-sized
  dataset (see `scripts/generate_datasets.py` for sharding notes if you want
  it larger still).
- `model.encoder.hidden_dims=[...]` / Transformer layer counts -- see
  `docs/training_guide.md` for the model config groups.
- `training.max_epochs=...` -- more epochs now that each one is cheap.

Two Hydra overrides actually engage the GPU + mixed precision:
`training.accelerator=gpu training.devices=1` and `training.precision=16`.
The FSDP/DDP knobs in `configs/training/default.yaml` are for multi-GPU and
aren't needed for a single-pod run.

A longer run is also more likely to get interrupted (pod pre-emption, OOM,
a network blip) partway through, so `scripts/train.py` checkpoints
periodically rather than only at the very end: `training.checkpoint_every_n_epochs`
(default 1) and `training.save_top_k` (default 1, by `val_loss`) control
that, and it always keeps a `last.ckpt` alongside the best ones, under
`{checkpoint_dir}/{dataset}-{encoder_kind}/`. To pick a run back up, add
`training.resume_from=auto` (uses `last.ckpt` if one exists, otherwise
starts fresh with a printed note) or point it at an exact checkpoint path.

## Troubleshooting: `LexerNoViableAltException`

If Hydra fails to start with a stack trace ending in
`LexerNoViableAltException`, the cause is almost always an **unquoted
override value that starts with digits and ends in a word**, e.g.
`training.precision=16-mixed` or `training.precision=bf16-mixed`. Hydra's
command-line grammar tries to lex the leading digits as a number and then
can't parse the trailing `-mixed`. Fixes, in order of preference:

- Use the bare-integer shorthand instead, if there is one --
  `training.precision=16` behaves identically to `"16-mixed"` for
  Lightning's `Trainer`.
- Otherwise, quote the value so Hydra treats it as a literal string, e.g.
  `training.precision=\'bf16-mixed\'` (the escaped single quotes need to
  survive whatever is invoking the command -- a real shell will pass them
  through; a UI text field that doesn't go through a shell may not).
- Also make sure the command is a **single line with no trailing
  backslashes** -- backslash line-continuation is bash syntax and can leave
  a stray `\` character in an argument if pasted into a field that doesn't
  run the command through `bash -c`.

## Troubleshooting: `RuntimeError: The NVIDIA driver on your system is too old`

This means the container's PyTorch build was compiled against a CUDA
runtime newer than the pod's NVIDIA driver supports -- a plain
`pip install torch` pulls whatever CUDA version PyPI bundles by default,
which drifts forward over time and can outrun an older (or Community Cloud)
host's driver. `Dockerfile` now pins torch to an explicit, older CUDA build
(`--index-url https://download.pytorch.org/whl/cu128`) before installing
the rest of `requirements.txt`, specifically to avoid this. If you still
hit it (e.g. a pod with an even older driver than cu128 needs, or the
`cu128` wheel channel has since been retired for the current torch
release):

- Check the pod's driver capability against what your image's torch build
  needs, or just try a different pod / GPU type -- driver versions vary
  across RunPod's Community Cloud fleet more than on Secure Cloud.
- If `cu128` wheels are no longer published for the current torch release
  by the time you read this, drop to an older tag (`cu126`, `cu124`, ...)
  in the `Dockerfile`'s `pip install torch --index-url ...` line -- check
  https://pytorch.org/get-started/locally/ for what's currently available,
  rebuild, and push (the GHCR workflow picks it up automatically on the
  next push to `main`).

## Troubleshooting: pod keeps re-running the training job from scratch

If your logs show `Seed set to 42` and a full, error-free training run
(ending in `Saved checkpoint to ...`) repeating every minute or so, nothing
is actually broken -- this is RunPod restarting the pod's container every
time its main process exits, success or failure alike, because a
Container Start Command is treated as a persistent process. Since that
command *is* the one-shot `scripts/train.py` invocation, finishing
training just gets it relaunched from the top, regenerating the same
(seeded) dataset and reinitializing the same model every time -- purely
wasted GPU-hours, not a data or training bug. It'll happen fast if you're
still on the small default dataset size; it's the same issue on a real,
longer run too, just on a slower cycle.

Fix: make the container idle instead of exit once training finishes, by
appending `; sleep infinity` to the Container Start Command (already
included in the Option 2 example above). Watch the logs for the final
`Saved checkpoint` line, then stop the pod yourself -- it will no longer
restart on its own.

If a run *did* get interrupted before you added `sleep infinity` (or for
any other reason -- OOM, pre-emption), you don't need to start over: add
`training.resume_from=auto` to the command to pick up from the last
periodic checkpoint instead of epoch 0 (see "Scaling up for a real run"
above).
