# Puzzle Piece Shape Generation

This folder contains tools for generating realistic puzzle piece shapes using cubic Bezier curves. The goal is to create parameterized puzzle piece outlines that can be used for synthetic data generation or visualization.

## Overview

Puzzle pieces have a characteristic shape with four edges, where each edge can be:
- **Tab** (knob/out): A protruding connector
- **Blank** (socket/in): An indentation that receives a tab
- **Flat**: A straight edge (typically on border pieces)

This project uses **4 cubic Bezier curves per edge** to create smooth, realistic-looking tabs and blanks with the classic "mushroom" shape.

## Directory Structure

```
puzzle_shape_generation/
├── README.md                    # This file
├── PUZZLE_PIECE_SHAPE_GENERATION.md  # Detailed technical notes
├── reference_pieces.json        # JSON config defining 6 sample pieces
├── scripts/
│   ├── bezier_piece_generator.py   # Main generator (creates PNG pieces)
│   ├── compare_with_references.py  # Visual comparison tool
│   ├── shape_comparator.py         # Quantitative shape comparison (IoU, Hausdorff)
│   ├── standardize_references.py   # Preprocessing for reference images
│   └── optimize_parameters.py      # Parameter optimizer using shape metrics
├── reference_images/
│   ├── pieces_1.png             # Reference: grid of puzzle pieces
│   ├── pieces_2.webp            # Reference: individual pieces
│   ├── individual/              # Cropped individual pieces from references
│   └── standardized/            # Preprocessed pieces (transparent bg, solid fill)
└── outputs/                     # Generated comparison images
```

## Scripts

### 1. bezier_piece_generator.py

The main tool for generating puzzle piece shapes. Creates PNG images with transparent backgrounds.

**Basic Usage:**
```bash
cd scripts

# Generate a random puzzle piece
python bezier_piece_generator.py

# Specify output path and size
python bezier_piece_generator.py -o my_piece.png --size 256

# Use a preset configuration (ref1-ref6)
python bezier_piece_generator.py --preset ref1

# Generate from a JSON configuration file
python bezier_piece_generator.py --json ../reference_pieces.json --output-dir ../outputs/from_json

# Export preset configurations to JSON for editing
python bezier_piece_generator.py --export-json my_presets.json

# Generate comparison visualization
python bezier_piece_generator.py --compare
```

**Key Features:**
- Generates complete 4-sided puzzle pieces
- Configurable edge types (tab/blank/flat) per edge
- 6 tunable parameters per tab/blank (see Parameterization below)
- JSON import/export for batch generation
- Transparent PNG output

### 2. compare_with_references.py

Creates side-by-side visual comparisons between reference images and generated pieces.

```bash
cd scripts
python compare_with_references.py
```

**Outputs:**
- `outputs/comparison_with_references.png` - Reference vs generated comparison
- `outputs/parameter_exploration.png` - Grid showing parameter variations

### 3. shape_comparator.py

Quantitatively compares generated pieces against reference images using contour-based metrics.

```bash
cd scripts

# Compare a single piece (1-6)
python shape_comparator.py 1

# Compare all pieces
python shape_comparator.py --all

# Verbose output with details
python shape_comparator.py 1 --verbose
```

**Metrics:**
- **IoU (Intersection over Union)**: Shape overlap (1.0 = perfect)
- **Mean Contour Distance**: Average distance between contour points (0 = perfect)
- **Hausdorff Distance**: Maximum distance between contours (0 = perfect)

### 4. standardize_references.py

Preprocesses reference images by removing backgrounds and applying solid fills. This creates consistent images for shape comparison.

```bash
cd scripts
python standardize_references.py
```

**Converts:**
- `reference_images/individual/` (colored pieces on white background)
- to `reference_images/standardized/` (solid red fill on transparent background)

### 5. optimize_parameters.py

Automatically optimizes the parameters in `reference_pieces.json` to minimize the difference between generated pieces and reference images.

```bash
cd scripts

# Optimize a single piece (1-6)
python optimize_parameters.py 1

# Optimize all pieces
python optimize_parameters.py --all

# Dry run (preview without saving)
python optimize_parameters.py --all --dry-run

# Use differential evolution (recommended for better results)
python optimize_parameters.py --all --method differential_evolution

# Higher precision (more iterations)
python optimize_parameters.py --all --method differential_evolution --max-iter 100

# Save to a different file
python optimize_parameters.py --all -o optimized_pieces.json

# Verbose output
python optimize_parameters.py 1 -v
```

**Optimization methods:**
- `L-BFGS-B` (default): Fast local optimization, good for fine-tuning
- `differential_evolution`: Global optimization, better at finding optimal solutions but slower
- `Powell`, `Nelder-Mead`: Alternative local methods

**Optimized parameters:** neck_width, bulb_width, height, neck_ratio, curvature, asymmetry (position is fixed by default for stability)

## Parameterization

Each tab/blank is controlled by 7 semantic parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `neck_width` | Width of the narrow waist | 0.06 - 0.12 |
| `bulb_width` | Width of the bulb head | 0.20 - 0.32 |
| `height` | How far the tab protrudes | 0.12 - 0.30 |
| `position` | Where along edge the tab is centered | 0.35 - 0.65 |
| `neck_ratio` | Height of waist as proportion of total | 0.15 - 0.55 |
| `curvature` | How rounded the bulb is | 0.30 - 1.0 |
| `asymmetry` | Tilt direction (-1=left, 0=center, +1=right) | -0.15 - 0.15 |

All values are relative to the edge length (normalized to 1.0).

**Key insight:** The `neck_width` should be significantly smaller than `bulb_width` (ratio ~0.3-0.4) to create the characteristic "mushroom" shape.

**Best parameters for classic puzzle pieces:**
```python
TabParameters(
    neck_width=0.09,   # Narrow waist
    bulb_width=0.26,   # Wide bulb head
    height=0.20,       # Moderate protrusion
    neck_ratio=0.35,   # Waist at 35% height
    curvature=0.88,    # Very rounded
)
```

## JSON Configuration Format

Pieces can be defined in JSON for batch generation:

```json
{
  "pieces": [
    {
      "edge_types": ["tab", "blank", "blank", "tab"],
      "edge_params": [
        {
          "position": 0.5,
          "neck_width": 0.1,
          "bulb_width": 0.24,
          "height": 0.22,
          "neck_ratio": 0.35,
          "curvature": 0.85,
          "asymmetry": 0.0
        },
        ...
      ],
      "size": 1.0
    }
  ]
}
```

Edge order: bottom, right, top, left (counter-clockwise from bottom-left corner).

## Dependencies

```bash
pip install numpy matplotlib pillow opencv-python scipy
```

## Technical Details

The shape generation uses **4 cubic Bezier curves** per tab/blank:
1. **Curve 1**: Edge to neck base (smooth entry)
2. **Curve 2**: Neck through waist to bulb side (S-curve creating waist)
3. **Curve 3**: Bulb (semicircular arc)
4. **Curve 4**: Bulb side through waist back to edge (mirror)

This approach provides:
- Smooth, C1-continuous curves
- Natural-looking "mushroom" shapes
- Fine control over proportions
- 16 control points per edge (64 per piece)
- Effectively controlled by 6 intuitive parameters

See `PUZZLE_PIECE_SHAPE_GENERATION.md` for detailed technical notes on the Bezier curve mathematics and design decisions.
