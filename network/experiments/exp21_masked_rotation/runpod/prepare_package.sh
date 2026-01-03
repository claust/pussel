#!/bin/bash
# Prepare RunPod training package for exp21
# Includes source puzzles and generation script - dataset created on RunPod

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
NETWORK_DIR="$(dirname "$(dirname "$EXP_DIR")")"
OUTPUT_DIR="$NETWORK_DIR/runpod_package"
# Main worktree has the source puzzles
MAIN_WORKTREE="/Users/claus/Repos/pussel"

echo "========================================"
echo "Preparing Exp21 RunPod Training Package"
echo "========================================"
echo "Key change: Masked rotation correlation"
echo "Strategy: Generate dataset on RunPod"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Use gtar on macOS if available
if command -v gtar &> /dev/null; then
    TAR_CMD="gtar --no-mac-metadata"
else
    TAR_CMD="tar"
fi

# Copy exp21 Python files
echo "Copying exp21 Python files..."
for file in dataset.py model.py visualize.py train_cuda.py; do
    cp "$EXP_DIR/$file" "$OUTPUT_DIR/"
done

# Copy exp20's generate_dataset.py (needed to create pieces)
echo "Copying generate_dataset.py from exp20..."
EXP20_DIR="$NETWORK_DIR/experiments/exp20_realistic_pieces"
cp "$EXP20_DIR/generate_dataset.py" "$OUTPUT_DIR/"

# Copy puzzle_shapes library
echo "Copying puzzle_shapes library..."
PUZZLE_SHAPES_DIR="$NETWORK_DIR/../shared/puzzle_shapes/puzzle_shapes"
if [ -d "$PUZZLE_SHAPES_DIR" ]; then
    cp -r "$PUZZLE_SHAPES_DIR" "$OUTPUT_DIR/"
else
    echo "ERROR: puzzle_shapes not found at $PUZZLE_SHAPES_DIR"
    exit 1
fi

# Copy setup script
cp "$SCRIPT_DIR/setup_and_train.sh" "$OUTPUT_DIR/"
chmod +x "$OUTPUT_DIR/setup_and_train.sh"

# Find and package source puzzles
PUZZLE_DIRS=(
    "$MAIN_WORKTREE/network/datasets/puzzles"
    "$NETWORK_DIR/datasets/puzzles"
)

PUZZLE_DIR=""
for dir in "${PUZZLE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        PUZZLE_DIR="$dir"
        break
    fi
done

if [ -n "$PUZZLE_DIR" ]; then
    SOURCE_COUNT=$(ls "$PUZZLE_DIR" 2>/dev/null | grep -c "puzzle_" || echo "0")
    echo "Found source puzzles at: $PUZZLE_DIR ($SOURCE_COUNT images)"

    echo "Creating source puzzles archive (this may take a moment)..."
    cd "$(dirname "$PUZZLE_DIR")"
    $TAR_CMD -czf "$OUTPUT_DIR/puzzles.tar.gz" puzzles
    echo "  Created puzzles.tar.gz ($(du -h "$OUTPUT_DIR/puzzles.tar.gz" | cut -f1))"
else
    echo "ERROR: No source puzzles found!"
    exit 1
fi

# Create final package
echo ""
echo "Creating final package..."
cd "$OUTPUT_DIR"

tar -czf runpod_training.tar.gz \
    train_cuda.py \
    dataset.py \
    model.py \
    visualize.py \
    generate_dataset.py \
    puzzle_shapes \
    setup_and_train.sh \
    puzzles.tar.gz

echo ""
echo "========================================"
echo "Package Ready!"
echo "========================================"
echo "Location: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR/runpod_training.tar.gz"
echo ""
echo "To upload to RunPod:"
echo "  scp -P 30611 -i ~/.ssh/id_ed25519 $OUTPUT_DIR/runpod_training.tar.gz root@38.80.152.77:/workspace/"
echo ""
echo "Then on RunPod:"
echo "  cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh"
