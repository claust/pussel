#!/bin/bash
# Prepare package for RunPod upload
# Run this after dataset generation is complete

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETWORK_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PACKAGE_DIR="$NETWORK_DIR/runpod_package"

echo "Creating RunPod package..."
mkdir -p "$PACKAGE_DIR"

# Copy training code
echo "Copying training code..."
cp "$SCRIPT_DIR/train_cuda.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/dataset.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/model.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/visualize.py" "$PACKAGE_DIR/"
cp "$SCRIPT_DIR/runpod_setup.sh" "$PACKAGE_DIR/"

# Copy shared puzzle_shapes library
echo "Copying puzzle_shapes library..."
cp -r "$NETWORK_DIR/../shared/puzzle_shapes" "$PACKAGE_DIR/"

# Create dataset tarball (this is the big one)
DATASET_DIR="$NETWORK_DIR/datasets/realistic_4x4_20k"
if [ -d "$DATASET_DIR" ]; then
    echo "Creating dataset tarball (this may take a while)..."
    cd "$NETWORK_DIR/datasets"
    tar -czf "$PACKAGE_DIR/dataset.tar.gz" realistic_4x4_20k/
    echo "Dataset tarball created: $(du -h "$PACKAGE_DIR/dataset.tar.gz")"
else
    echo "Warning: Dataset directory not found at $DATASET_DIR"
    echo "Run dataset generation first!"
fi

# Also copy source puzzles for reference
echo "Creating source puzzles tarball..."
cd "$NETWORK_DIR/datasets"
tar -czf "$PACKAGE_DIR/puzzles.tar.gz" puzzles/
echo "Source puzzles tarball: $(du -h "$PACKAGE_DIR/puzzles.tar.gz")"

echo ""
echo "=========================================="
echo "Package created at: $PACKAGE_DIR"
echo "=========================================="
ls -lh "$PACKAGE_DIR/"
echo ""
echo "Upload to RunPod with:"
echo "  scp -r $PACKAGE_DIR/* root@<pod-ip>:/workspace/"
