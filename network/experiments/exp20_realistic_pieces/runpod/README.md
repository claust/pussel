# RunPod Training Setup

This directory contains everything needed to train on RunPod GPUs.

> **⚠️ Cost note:** RunPod is a **third-party cloud service that rents GPUs by
> the hour — you pay real money for it.** Nothing here runs automatically or is
> wired into the app or CI; it only incurs charges if you manually create a pod
> on runpod.io and run training on it. Remember to **stop/terminate the pod**
> when you're done, or it keeps billing.
>
> This path is **optional** and exists purely for speed (an RTX 4090 is ~10x
> faster than an M4 Mac). For the free default, **train locally** — e.g. on the
> Mac Mini — with `cd network && uv run python train.py`. Local training is
> slower per epoch but costs nothing.

## Quick Start

1. **Generate the package locally:**
   ```bash
   cd network/experiments/exp20_realistic_pieces
   ./runpod/prepare_package.sh
   ```

2. **Upload to RunPod:**
   ```bash
   # Get connection details from RunPod dashboard
   scp -P <PORT> -i ~/.ssh/runpod_key runpod_training.tar.gz root@<IP>:/workspace/
   ```

3. **On RunPod (via web terminal or SSH):**
   ```bash
   cd /workspace
   tar -xzf runpod_training.tar.gz
   ./setup_and_train.sh
   ```

## Files

- `prepare_package.sh` - Creates the training package locally
- `setup_and_train.sh` - Setup script that runs on RunPod

The package ships the experiment code with relative imports rewritten
for flat execution: `train.py` (unified entry point, AMP enabled
automatically on CUDA), `harness.py` (val-based checkpoint selection,
eval-mode metrics), `splits.py` + `splits/realistic_4x4_v1.json` (frozen
train/val/test split), `dataset.py`, `model.py`, `visualize.py`.

Training selects the best checkpoint on the validation split and
evaluates the test split exactly once at the end (`--eval-test`).

## SSH Key Setup

If SSH fails with "Permission denied":

1. Generate a key without passphrase:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/runpod_key -N ""
   ```

2. Add the public key to RunPod Settings > SSH Public Keys:
   ```bash
   cat ~/.ssh/runpod_key.pub
   ```

3. Restart the pod after adding the key

4. If still failing, add manually via web terminal:
   ```bash
   echo '<your-public-key>' >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

## Expected Training Time

- RTX 4090: ~3 hours for 50 epochs with 12k puzzles
- RTX 3090: ~4-5 hours
- A100: ~2 hours
