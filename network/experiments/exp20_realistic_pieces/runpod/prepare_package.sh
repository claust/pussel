#!/bin/bash
# Prepare RunPod training package
# Creates a single tar.gz that can be uploaded to RunPod

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
NETWORK_DIR="$(dirname "$EXP_DIR")"
OUTPUT_DIR="$NETWORK_DIR/runpod_package"
DATASETS_DIR="${DATASETS_DIR:-/datasets}"

echo "========================================"
echo "Preparing RunPod Training Package"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy Python files (fixing relative imports)
echo "Copying Python files..."
for file in dataset.py model.py visualize.py; do
    cp "$EXP_DIR/$file" "$OUTPUT_DIR/"
    # Convert relative imports to absolute
    sed -i '' 's/from \.dataset/from dataset/g' "$OUTPUT_DIR/$file" 2>/dev/null || \
    sed -i 's/from \.dataset/from dataset/g' "$OUTPUT_DIR/$file"
    sed -i '' 's/from \.model/from model/g' "$OUTPUT_DIR/$file" 2>/dev/null || \
    sed -i 's/from \.model/from model/g' "$OUTPUT_DIR/$file"
done

# Copy training script
cp "$EXP_DIR/train_cuda.py" "$OUTPUT_DIR/"

# Copy setup script
cp "$SCRIPT_DIR/setup_and_train.sh" "$OUTPUT_DIR/"
chmod +x "$OUTPUT_DIR/setup_and_train.sh"

# Copy puzzle shapes library
echo "Copying puzzle shapes library..."
cp -r "$EXP_DIR/puzzle_shapes" "$OUTPUT_DIR/"

# Check for generated dataset
DATASET_DIR="$EXP_DIR/datasets/realistic_4x4_20k"
if [ ! -d "$DATASET_DIR" ]; then
    DATASET_DIR="$EXP_DIR/realistic_4x4_20k"
fi

if [ -d "$DATASET_DIR" ]; then
    PUZZLE_COUNT=$(ls "$DATASET_DIR" | wc -l | tr -d ' ')
    echo "Creating dataset archive ($PUZZLE_COUNT puzzles)..."

    # Use gtar on macOS if available (handles extended attributes better)
    if command -v gtar &> /dev/null; then
        TAR_CMD="gtar --no-mac-metadata"
    else
        TAR_CMD="tar"
    fi

    cd "$(dirname "$DATASET_DIR")"
    $TAR_CMD -czf "$OUTPUT_DIR/dataset.tar.gz" "$(basename "$DATASET_DIR")"
    echo "  Created dataset.tar.gz ($(du -h "$OUTPUT_DIR/dataset.tar.gz" | cut -f1))"
else
    echo "WARNING: No dataset found. Run generate_puzzles.py first."
fi

# Check for source puzzles
if [ -d "$DATASETS_DIR/puzzles" ]; then
    SOURCE_COUNT=$(ls "$DATASETS_DIR/puzzles" | wc -l | tr -d ' ')
    echo "Creating source puzzles archive ($SOURCE_COUNT images)..."

    cd "$DATASETS_DIR"
    $TAR_CMD -czf "$OUTPUT_DIR/puzzles.tar.gz" puzzles
    echo "  Created puzzles.tar.gz ($(du -h "$OUTPUT_DIR/puzzles.tar.gz" | cut -f1))"
else
    echo "WARNING: No source puzzles found at $DATASETS_DIR/puzzles"
fi

# Create single upload package
echo ""
echo "Creating final package..."
cd "$OUTPUT_DIR"
tar -czf runpod_training.tar.gz \
    train_cuda.py \
    dataset.py \
    model.py \
    visualize.py \
    setup_and_train.sh \
    puzzle_shapes \
    dataset.tar.gz \
    puzzles.tar.gz 2>/dev/null || \
tar -czf runpod_training.tar.gz \
    train_cuda.py \
    dataset.py \
    model.py \
    visualize.py \
    setup_and_train.sh \
    puzzle_shapes

echo ""
echo "========================================"
echo "Package Ready!"
echo "========================================"
echo "Location: $OUTPUT_DIR"
echo ""
ls -lh "$OUTPUT_DIR"
echo ""
echo "To upload to RunPod:"
echo "  scp -P <PORT> -i ~/.ssh/runpod_key $OUTPUT_DIR/runpod_training.tar.gz root@<IP>:/workspace/"
echo ""
echo "Then on RunPod:"
echo "  cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh"
