#!/bin/bash
# exp27 RunPod setup + training.
# Run after extracting runpod_training.tar.gz into /workspace.
#
# Steps: install deps -> point HF at the bundled offline weights -> extract
# source puzzles -> generate the RGBA piece dataset (parallel, resumable) ->
# train the frozen-feature model, selecting the checkpoint on val and
# touching the synthetic test set once.

set -e

EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GEN_WORKERS="${GEN_WORKERS:-$(nproc)}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
AUG_PRESET="${AUG_PRESET:-full}"
POSITION_HEAD="${POSITION_HEAD:-dense}"
# Keep generated pieces on the container disk, NOT /workspace — RunPod
# network volumes (MooseFS) charge ~4x quota for ~192k small PNGs (exp26
# lesson). Generation is resumable (--skip-existing) and takes ~15 min.
RGBA_DIR="${RGBA_DIR:-/root/realistic_4x4_rgba}"

echo "========================================"
echo "exp27 RunPod Setup (head=$POSITION_HEAD, epochs=$EPOCHS, batch=$BATCH_SIZE)"
echo "========================================"

cd /workspace

echo "Installing dependencies..."
# torch/torchvision come from the RunPod image; do NOT reinstall them (a bare
# pip install could pull a wheel against a different CUDA and break the GPU).
python - <<'EOF'
import torch, torchvision  # noqa: F401
print(f"Using image torch {torch.__version__}, torchvision {torchvision.__version__}")
EOF
# --break-system-packages: Ubuntu 24.04 marks system Python externally managed.
pip install --quiet --break-system-packages pillow numpy tqdm timm matplotlib

# Bundled DINOv2 weights: use the shipped HF cache and stay offline so a
# flaky/blocked hub can never stall or fail the run.
export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1
python - <<'EOF'
import timm
m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
print(f"Encoder loaded offline OK ({sum(p.numel() for p in m.parameters()):,} params)")
EOF

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
    --position-head "$POSITION_HEAD" \
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
