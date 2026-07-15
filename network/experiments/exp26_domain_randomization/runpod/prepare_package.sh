#!/bin/bash
# Prepare the exp26 (domain randomization) RunPod training package.
#
# Unlike exp20, exp26 does NOT ship a pre-generated piece dataset: the RGBA
# pieces are generated ON the pod from the source puzzles (see
# setup_and_train.sh) because they are large and the generation is cheap
# and parallel. This package therefore contains only the code, the frozen
# split, puzzle_shapes, and the source puzzles.
#
# exp26 reuses several exp20 modules (dataset.py, model.py, harness.py,
# splits.py, visualize.py); they are copied in and their package-relative
# imports are flattened for RunPod's flat execution layout.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP26_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$(dirname "$EXP26_DIR")"
NETWORK_DIR="$(dirname "$EXPERIMENTS_DIR")"
EXP20_DIR="$EXPERIMENTS_DIR/exp20_realistic_pieces"
OUTPUT_DIR="$NETWORK_DIR/runpod_package_exp26"
DATASETS_DIR="${DATASETS_DIR:-$NETWORK_DIR/datasets}"

echo "========================================"
echo "Preparing exp26 RunPod Package"
echo "========================================"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/splits"

flatten_imports() {
    # from ..exp20_realistic_pieces.foo import  ->  from foo import
    # from .foo import                          ->  from foo import
    local f="$1"
    sed -i '' -E 's/from \.\.exp20_realistic_pieces\.([a-z_]+) import/from \1 import/g' "$f" 2>/dev/null || \
        sed -i -E 's/from \.\.exp20_realistic_pieces\.([a-z_]+) import/from \1 import/g' "$f"
    sed -i '' -E 's/from \.([a-z_]+) import/from \1 import/g' "$f" 2>/dev/null || \
        sed -i -E 's/from \.([a-z_]+) import/from \1 import/g' "$f"
}

echo "Copying exp20 harness modules..."
for file in dataset.py model.py visualize.py splits.py harness.py; do
    cp "$EXP20_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done

echo "Copying exp26 modules..."
for file in augment.py aug_dataset.py generate_dataset.py train.py; do
    cp "$EXP26_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done

echo "Copying frozen split..."
cp "$EXP20_DIR/splits/"*.json "$OUTPUT_DIR/splits/"

echo "Copying setup script..."
cp "$SCRIPT_DIR/setup_and_train.sh" "$OUTPUT_DIR/"
chmod +x "$OUTPUT_DIR/setup_and_train.sh"

echo "Copying puzzle_shapes library..."
PUZZLE_SHAPES_SRC="$NETWORK_DIR/../shared/puzzle_shapes/puzzle_shapes"
if [ ! -d "$PUZZLE_SHAPES_SRC" ]; then
    echo "ERROR: puzzle_shapes not found at $PUZZLE_SHAPES_SRC"
    exit 1
fi
cp -r "$PUZZLE_SHAPES_SRC" "$OUTPUT_DIR/"

if command -v gtar &> /dev/null; then
    TAR_CMD="gtar --no-mac-metadata"
else
    TAR_CMD="tar"
fi

echo "Archiving source puzzles..."
if [ -d "$DATASETS_DIR/puzzles" ]; then
    SOURCE_COUNT=$(ls "$DATASETS_DIR/puzzles" | wc -l | tr -d ' ')
    echo "  $SOURCE_COUNT source puzzles"
    ( cd "$DATASETS_DIR" && $TAR_CMD -czf "$OUTPUT_DIR/puzzles.tar.gz" puzzles )
    echo "  Created puzzles.tar.gz ($(du -h "$OUTPUT_DIR/puzzles.tar.gz" | cut -f1))"
else
    echo "ERROR: No source puzzles at $DATASETS_DIR/puzzles (set DATASETS_DIR)"
    exit 1
fi

echo "Creating final package..."
( cd "$OUTPUT_DIR" && tar -czf runpod_training.tar.gz \
    dataset.py model.py visualize.py splits.py harness.py \
    augment.py aug_dataset.py generate_dataset.py train.py \
    splits setup_and_train.sh puzzle_shapes puzzles.tar.gz )

echo ""
echo "========================================"
echo "Package Ready: $OUTPUT_DIR"
echo "========================================"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Upload:"
echo "  scp -P <PORT> -i ~/.ssh/runpod_key $OUTPUT_DIR/runpod_training.tar.gz root@<IP>:/workspace/"
echo "On RunPod:"
echo "  cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh"
