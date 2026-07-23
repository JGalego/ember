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
     with something like:
     ```
     python scripts/train.py dataset=sudoku training.max_epochs=50 \
       training.accelerator=gpu training.devices=1 training.precision=16-mixed \
       checkpoint_dir=/workspace/checkpoints
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
   python scripts/train.py dataset=sudoku training.max_epochs=50 \
     training.accelerator=gpu training.devices=1 training.precision=16-mixed
   ```
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
`training.accelerator=gpu training.devices=1` and
`training.precision=16-mixed`. The FSDP/DDP knobs in
`configs/training/default.yaml` are for multi-GPU and aren't needed for a
single-pod run.
