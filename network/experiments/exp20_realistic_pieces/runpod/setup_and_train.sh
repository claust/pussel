#!/bin/bash
# RunPod setup and training script
# Run this after extracting runpod_training.tar.gz to /workspace

set -e

echo "========================================"
echo "RunPod Training Setup"
echo "========================================"

cd /workspace

# Install dependencies
echo "Installing Python dependencies..."
pip install --quiet torch torchvision pillow tqdm timm matplotlib

# Verify CUDA
echo ""
echo "Checking CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Extract datasets if needed
if [ -f dataset.tar.gz ] && [ ! -d realistic_4x4_20k ]; then
    echo ""
    echo "Extracting dataset..."
    tar -xzf dataset.tar.gz 2>/dev/null || tar -xf dataset.tar.gz
fi

if [ -f puzzles.tar.gz ] && [ ! -d /datasets/puzzles ]; then
    echo ""
    echo "Extracting source puzzles..."
    mkdir -p /datasets
    cd /datasets
    tar -xzf /workspace/puzzles.tar.gz 2>/dev/null || tar -xf /workspace/puzzles.tar.gz
    cd /workspace
fi

# Verify datasets
echo ""
echo "Checking datasets..."
PUZZLE_COUNT=$(ls realistic_4x4_20k 2>/dev/null | wc -l)
SOURCE_COUNT=$(ls /datasets/puzzles 2>/dev/null | wc -l)
echo "  Generated puzzles: $PUZZLE_COUNT folders"
echo "  Source puzzles: $SOURCE_COUNT images"

if [ "$PUZZLE_COUNT" -eq 0 ]; then
    echo "ERROR: No generated puzzles found in realistic_4x4_20k/"
    exit 1
fi

if [ "$SOURCE_COUNT" -eq 0 ]; then
    echo "ERROR: No source puzzles found in /datasets/puzzles/"
    exit 1
fi

# Start training
echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
echo "Epochs: 50"
echo "Training puzzles: 10800"
echo "Test puzzles: 1200"
echo "Batch size: 128"
echo ""

python train_cuda.py \
    --dataset-root /workspace/realistic_4x4_20k \
    --epochs 50 \
    --batch-size 128 \
    --n-train 10800 \
    --n-test 1200 \
    --num-workers 8

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Results saved to /workspace/outputs/"
ls -la /workspace/outputs/
