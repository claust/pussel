#!/bin/bash
# Prepare the exp27 (frozen pretrained features) RunPod training package.
#
# Like exp26, the RGBA pieces are generated ON the pod (container disk, see
# setup_and_train.sh). The package contains the code, the frozen split,
# puzzle_shapes, the source puzzles — and the pre-downloaded DINOv2 weights
# (HF hub access from pods has been flaky; the encoder must load offline).
#
# exp27 reuses exp20 modules (dataset.py, model.py as exp20_model.py,
# harness.py, splits.py, visualize.py) and exp26 modules (augment.py,
# aug_dataset.py, generate_dataset.py); package-relative imports are
# flattened for RunPod's flat execution layout.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP27_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$(dirname "$EXP27_DIR")"
NETWORK_DIR="$(dirname "$EXPERIMENTS_DIR")"
EXP20_DIR="$EXPERIMENTS_DIR/exp20_realistic_pieces"
EXP26_DIR="$EXPERIMENTS_DIR/exp26_domain_randomization"
OUTPUT_DIR="$NETWORK_DIR/runpod_package_exp27"
DATASETS_DIR="${DATASETS_DIR:-$NETWORK_DIR/datasets}"
HF_MODEL_DIR="${HF_MODEL_DIR:-$HOME/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m}"

echo "========================================"
echo "Preparing exp27 RunPod Package"
echo "========================================"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/splits"

flatten_imports() {
    # from ..exp20_realistic_pieces.model import X -> from exp20_model import X
    # from ..exp20_realistic_pieces.foo import     -> from foo import
    # from ..exp26_domain_randomization.foo import -> from foo import
    # from .foo import                             -> from foo import
    local f="$1"
    if sed --version >/dev/null 2>&1; then SED=(sed -i -E); else SED=(sed -i '' -E); fi
    "${SED[@]}" 's/from \.\.exp20_realistic_pieces\.model import/from exp20_model import/g' "$f"
    "${SED[@]}" 's/from \.\.exp20_realistic_pieces\.([a-z_]+) import/from \1 import/g' "$f"
    "${SED[@]}" 's/from \.\.exp26_domain_randomization\.([a-z_]+) import/from \1 import/g' "$f"
    "${SED[@]}" 's/from \.model import/from exp27_model import/g' "$f"
    "${SED[@]}" 's/from \.([a-z_]+) import/from \1 import/g' "$f"
}

echo "Copying exp20 harness modules..."
cp "$EXP20_DIR/model.py" "$OUTPUT_DIR/exp20_model.py"
flatten_imports "$OUTPUT_DIR/exp20_model.py"
for file in dataset.py visualize.py splits.py harness.py; do
    cp "$EXP20_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done
# flatten_imports rewrote harness.py's "from .model import" to exp27_model
# (that rule targets exp27's own files); harness means the exp20 model.
if sed --version >/dev/null 2>&1; then
    sed -i -E 's/^from exp27_model import/from exp20_model import/' "$OUTPUT_DIR/harness.py"
else
    sed -i '' -E 's/^from exp27_model import/from exp20_model import/' "$OUTPUT_DIR/harness.py"
fi

echo "Copying exp26 data modules..."
for file in augment.py aug_dataset.py generate_dataset.py; do
    cp "$EXP26_DIR/$file" "$OUTPUT_DIR/"
    flatten_imports "$OUTPUT_DIR/$file"
done
# aug_dataset.py imports "from dataset import ..." (exp20) — already flat.

echo "Copying exp27 modules..."
cp "$EXP27_DIR/model.py" "$OUTPUT_DIR/exp27_model.py"
cp "$EXP27_DIR/train.py" "$OUTPUT_DIR/train.py"
flatten_imports "$OUTPUT_DIR/exp27_model.py"
flatten_imports "$OUTPUT_DIR/train.py"

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

echo "Bundling pretrained DINOv2 weights (offline HF cache)..."
if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "ERROR: cached weights not found at $HF_MODEL_DIR"
    echo "Run the encoder once locally first (python -m experiments.exp27_pretrained_features.model)"
    exit 1
fi
mkdir -p "$OUTPUT_DIR/hf_cache/hub"
cp -RL "$HF_MODEL_DIR" "$OUTPUT_DIR/hf_cache/hub/"

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
( cd "$OUTPUT_DIR" && $TAR_CMD -czf runpod_training.tar.gz \
    dataset.py exp20_model.py exp27_model.py visualize.py splits.py harness.py \
    augment.py aug_dataset.py generate_dataset.py train.py \
    splits setup_and_train.sh puzzle_shapes hf_cache puzzles.tar.gz )

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
