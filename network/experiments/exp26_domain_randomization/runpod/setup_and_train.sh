#!/bin/bash
# exp26 RunPod setup + training.
# Run after extracting runpod_training.tar.gz into /workspace.
#
# Steps: install deps -> extract source puzzles -> generate the RGBA piece
# dataset (parallel, resumable) -> train with domain randomization,
# selecting the checkpoint on val and touching the synthetic test set once.

set -e

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GEN_WORKERS="${GEN_WORKERS:-$(nproc)}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
AUG_PRESET="${AUG_PRESET:-full}"
# Where the generated RGBA pieces live. IMPORTANT: keep this on the
# container disk (/root), NOT on /workspace — RunPod network volumes
# (MooseFS) charge ~4x quota for the ~192k small PNGs (9 GB vs the true
# 2.2 GB) and blow a 10 GB volume. The container disk is ephemeral, but
# generation is resumable (--skip-existing) and takes ~15 min; the
# checkpoints/results below still go to the persistent /workspace.
RGBA_DIR="${RGBA_DIR:-/root/realistic_4x4_rgba}"

echo "========================================"
echo "exp26 RunPod Setup (preset=$AUG_PRESET, epochs=$EPOCHS, batch=$BATCH_SIZE)"
echo "========================================"

cd /workspace

echo "Installing dependencies..."
# torch/torchvision come from the RunPod image (e.g. torch 2.8.0+cu128 on
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 — cu128 is required for
# Blackwell cards like the RTX PRO 6000). Do NOT reinstall them here: a bare
# 'pip install torchvision' could pull a wheel built against a different
# torch/CUDA and break the GPU build.
python - <<'EOF'
import torch, torchvision  # noqa: F401
print(f"Using image torch {torch.__version__}, torchvision {torchvision.__version__}")
EOF
# --break-system-packages: Ubuntu 24.04 images mark the system Python as
# externally managed (PEP 668); the RunPod image's own torch lives there,
# so installing alongside it is exactly what we want.
pip install --quiet --break-system-packages pillow numpy tqdm timm matplotlib

echo "Checking CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if [ -f puzzles.tar.gz ] && [ ! -d /datasets/puzzles ]; then
    echo "Extracting source puzzles..."
    mkdir -p /datasets
    ( cd /datasets && tar -xzf /workspace/puzzles.tar.gz )
fi

SOURCE_COUNT=$(ls /datasets/puzzles 2>/dev/null | wc -l)
echo "Source puzzles: $SOURCE_COUNT"
if [ "$SOURCE_COUNT" -eq 0 ]; then
    echo "ERROR: no source puzzles in /datasets/puzzles"
    exit 1
fi

# Generate RGBA pieces for ALL source puzzles (the frozen split's
# train/val/test IDs span the whole set). Resumable via --skip-existing.
echo ""
echo "Generating RGBA pieces (workers=$GEN_WORKERS) -> $RGBA_DIR ..."
python generate_dataset.py \
    --source-dir /datasets/puzzles \
    --output-dir "$RGBA_DIR" \
    --n-puzzles 100000 \
    --workers "$GEN_WORKERS" \
    --skip-existing

GEN_COUNT=$(ls "$RGBA_DIR" 2>/dev/null | wc -l)
echo "Generated puzzle dirs: $GEN_COUNT"

echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
python train.py \
    --dataset-root "$RGBA_DIR" \
    --puzzle-root /datasets/puzzles \
    --output-dir /workspace/outputs \
    --aug-preset "$AUG_PRESET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$TRAIN_WORKERS" \
    --eval-test

echo ""
echo "========================================"
echo "Training Complete! Results in /workspace/outputs/"
echo "========================================"
ls -la /workspace/outputs/
echo ""
echo "Download (both the harness checkpoint and the raw north-star one):"
echo "  scp -P <PORT> -i ~/.ssh/runpod_key 'root@<IP>:/workspace/outputs/*' ./outputs/"
