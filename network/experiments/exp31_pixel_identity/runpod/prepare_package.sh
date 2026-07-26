#!/bin/bash
# Prepare the exp31 (pixel identity) RunPod training package.
#
# Like exp26/exp30, exp31 does NOT ship a pre-generated piece dataset: the
# RGBA pieces are generated ON the pod from the source puzzles (see
# setup_and_train.sh) because they are large and the generation is cheap and
# parallel. This package therefore contains only the code, the frozen split,
# puzzle_shapes, and the source puzzles.
#
# exp31 reuses modules from four earlier experiments — exp20 (dataset, model,
# harness, splits, visualize), exp24 (data_prep: the real pad_to_square
# framing geometry), exp26 (augment, aug_dataset) and exp30 (framing,
# framed_dataset, generate_dataset) — so all of them are copied in and their
# package-relative imports are flattened for RunPod's flat execution layout.
#
# exp31 generates into exp30's dataset root (realistic_4x4_rgba_v2): the
# stored pieces are identical, since everything exp31 adds happens at load
# time.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP31_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$(dirname "$EXP31_DIR")"
NETWORK_DIR="$(dirname "$EXPERIMENTS_DIR")"
EXP20_DIR="$EXPERIMENTS_DIR/exp20_realistic_pieces"
EXP24_DIR="$EXPERIMENTS_DIR/exp24_piece_classifier"
EXP26_DIR="$EXPERIMENTS_DIR/exp26_domain_randomization"
EXP30_DIR="$EXPERIMENTS_DIR/exp30_generator_fixes"
OUTPUT_DIR="$NETWORK_DIR/runpod_package_exp31"
DATASETS_DIR="${DATASETS_DIR:-$NETWORK_DIR/datasets}"

echo "========================================"
echo "Preparing exp31 RunPod Package"
echo "========================================"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/splits"

flatten_imports() {
    # from ..expNN_whatever.foo import  ->  from foo import
    # from .foo import                  ->  from foo import
    local f="$1"
    local expr='s/from \.\.exp[0-9]+_[a-z_]+\.([a-z_]+) import/from \1 import/g; s/from \.([a-z_]+) import/from \1 import/g'
    sed -i '' -E "$expr" "$f" 2>/dev/null || sed -i -E "$expr" "$f"
}

echo "Copying exp20 harness modules..."
for file in dataset.py model.py visualize.py splits.py harness.py; do
    cp "$EXP20_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done

echo "Copying exp24 real-path framing module..."
cp "$EXP24_DIR/data_prep.py" "$OUTPUT_DIR/"
flatten_imports "$OUTPUT_DIR/data_prep.py"

echo "Copying exp26 augmentation modules..."
for file in augment.py aug_dataset.py; do
    cp "$EXP26_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done

echo "Copying exp30 framing + generator modules..."
for file in framing.py framed_dataset.py generate_dataset.py probes.py; do
    cp "$EXP30_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done

echo "Copying exp31 modules..."
for file in capture.py capture_dataset.py augmentation.py ncc_probe.py train.py segmentation_validation.py; do
    cp "$EXP31_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done
# ncc_probe.py (the section 8 acceptance gate) is copied in the loop above,
# not optionally: setup_and_train.sh refuses to train without it.

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
    dataset.py model.py visualize.py splits.py harness.py data_prep.py \
    augment.py aug_dataset.py framing.py framed_dataset.py \
    generate_dataset.py probes.py capture.py capture_dataset.py augmentation.py ncc_probe.py train.py \
    segmentation_validation.py \
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
