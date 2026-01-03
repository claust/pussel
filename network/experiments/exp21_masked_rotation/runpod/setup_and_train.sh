#!/bin/bash
# RunPod setup and training script for exp21
# Generates dataset from source puzzles, then trains

set -e

echo "========================================"
echo "Exp21: Masked Rotation Correlation"
echo "========================================"
echo "Hypothesis: Masking out black background in"
echo "rotation correlation will improve test accuracy"
echo ""

cd /workspace

# Install dependencies
echo "Installing Python dependencies..."
pip install --quiet torch torchvision pillow tqdm timm matplotlib

# Verify CUDA
echo ""
echo "Checking CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Extract source puzzles
if [ -f puzzles.tar.gz ] && [ ! -d /workspace/puzzles ]; then
    echo ""
    echo "Extracting source puzzles..."
    tar -xzf puzzles.tar.gz
fi

# Verify source puzzles
SOURCE_COUNT=$(ls /workspace/puzzles 2>/dev/null | grep -c "puzzle_" || echo "0")
echo "Source puzzle images: $SOURCE_COUNT"

if [ "$SOURCE_COUNT" -eq 0 ]; then
    echo "ERROR: No source puzzles found!"
    exit 1
fi

# Generate realistic pieces dataset (if not already done)
DATASET_ROOT="/workspace/realistic_4x4"
N_PUZZLES=12000

if [ ! -d "$DATASET_ROOT" ] || [ $(ls "$DATASET_ROOT" 2>/dev/null | grep -c "puzzle_" || echo "0") -lt 1000 ]; then
    echo ""
    echo "========================================"
    echo "Generating Realistic Pieces Dataset"
    echo "========================================"
    echo "This will create 4x4 puzzle pieces with Bezier edges"
    echo "Puzzles: $N_PUZZLES"
    echo ""

    # Add puzzle_shapes to Python path
    export PYTHONPATH="/workspace:$PYTHONPATH"

    python generate_dataset.py \
        --source-dir /workspace/puzzles \
        --output-dir "$DATASET_ROOT" \
        --n-puzzles $N_PUZZLES \
        --seed 42

    echo ""
    echo "Dataset generation complete!"
fi

# Verify generated dataset
PUZZLE_COUNT=$(ls "$DATASET_ROOT" 2>/dev/null | grep -c "puzzle_" || echo "0")
echo ""
echo "Dataset verification:"
echo "  Dataset root: $DATASET_ROOT"
echo "  Generated puzzle folders: $PUZZLE_COUNT"

if [ "$PUZZLE_COUNT" -eq 0 ]; then
    echo "ERROR: No generated puzzles found!"
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
echo "Mask threshold: 0.02"
echo ""

python train_cuda.py \
    --dataset-root "$DATASET_ROOT" \
    --puzzle-root /workspace/puzzles \
    --epochs 50 \
    --batch-size 128 \
    --n-train 10800 \
    --n-test 1200 \
    --num-workers 8 \
    --mask-threshold 0.02

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Results saved to /workspace/outputs/"
ls -la /workspace/outputs/
echo ""
echo "To download results:"
echo "  scp -P 30611 -i ~/.ssh/id_ed25519 'root@38.80.152.77:/workspace/outputs/*' ./outputs/"
