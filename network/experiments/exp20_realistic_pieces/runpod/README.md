# RunPod Training Setup

This directory contains everything needed to train on RunPod GPUs.

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
- `train_cuda.py` - CUDA-optimized training script with AMP
- `dataset.py` - Dataset loading (standalone, no relative imports)
- `model.py` - Model definition (standalone)
- `visualize.py` - Visualization utilities (standalone)

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
