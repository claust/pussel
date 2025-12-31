#!/bin/bash
# RunPod setup script for exp20 training
# Run this after SSH-ing into the pod

set -e

echo "=========================================="
echo "Setting up exp20 training environment"
echo "=========================================="

# Install system dependencies
apt-get update && apt-get install -y python3-pip unzip

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install pillow matplotlib numpy

# Extract dataset (assumes dataset.tar.gz is uploaded)
if [ -f "dataset.tar.gz" ]; then
    echo "Extracting dataset..."
    mkdir -p datasets
    tar -xzf dataset.tar.gz -C datasets/
    echo "Dataset extracted to datasets/"
fi

# Create output directory
mkdir -p outputs

echo ""
echo "=========================================="
echo "Setup complete! Run training with:"
echo "=========================================="
echo ""
echo "python train_cuda.py --epochs 50 --n-train 10000 --n-test 1000 --batch-size 128"
echo ""
