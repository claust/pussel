# Puzzle Piece Shape Generation

This folder contains tools for generating realistic puzzle piece shapes using cubic Bezier curves. The goal is to create parameterized puzzle piece outlines that can be used for synthetic data generation or visualization.

## Overview

Puzzle pieces have a characteristic shape with four edges, where each edge can be:
- **Tab** (knob/out): A protruding connector
- **Blank** (socket/in): An indentation that receives a tab
- **Flat**: A straight edge (typically on border pieces)

This project uses **5 cubic Bezier curves per edge** to create smooth, realistic-looking tabs and blanks with the classic "mushroom" shape.

## Directory Structure

```
puzzle_shape_generation/
├── README.md                    # This file
├── PUZZLE_PIECE_SHAPE_GENERATION.md  # Detailed technical notes
├── reference_pieces.json        # JSON config defining 6 sample pieces
├── scripts/
│   ├── piece_generator.py          # Main CLI wrapper (creates PNG pieces)
│   ├── puzzle_cutter.py            # Cut images into interlocking puzzle pieces
│   ├── edge_grid.py                # Edge grid generation for puzzle cutting
│   ├── image_masking.py            # Image cutting with polygon masks
│   ├── models.py                   # Data classes (PieceConfig, TabParameters)
│   ├── geometry.py                 # Core math for Bezier shape generation
│   ├── rendering.py                # PNG rendering and visualization
│   ├── io_utils.py                 # JSON loading and saving
│   ├── comparison.py               # Reference comparison logic
│   ├── shape_comparator.py         # Quantitative shape comparison (IoU, Hausdorff)
│   ├── optimize_parameters.py      # Parameter optimizer using shape metrics
│   ├── visualize_comparison.py     # Overlay visualization for debugging
│   └── diagnose_curves.py          # Tool to visualize the curve structure
├── reference_images/
│   ├── pieces_1.png             # Reference: grid of puzzle pieces
│   ├── pieces_2.webp            # Reference: individual pieces
│   ├── individual/              # Cropped individual pieces from references
│   └── standardized/            # Preprocessed pieces (transparent bg, solid fill)
└── outputs/                     # Generated comparison images
```

## Scripts

### 1. piece_generator.py

The main CLI wrapper for generating puzzle piece shapes. It uses the modular libraries (`geometry`, `rendering`, etc.) to create PNG images with transparent backgrounds.

**Basic Usage:**
```bash
cd scripts

# Generate a random puzzle piece
python piece_generator.py

# Specify output path and size
python piece_generator.py -o my_piece.png --size 256

# Use a preset configuration (ref1-ref6)
python piece_generator.py --preset ref1

# Generate from a JSON configuration file
python piece_generator.py --json ../reference_pieces.json --output-dir ../outputs/from_json

# Export preset configurations to JSON for editing
python piece_generator.py --export-json my_presets.json

# Generate comparison visualization
python piece_generator.py --compare
```

**Key Features:**
- Generates complete 4-sided puzzle pieces
- Configurable edge types (tab/blank/flat) per edge
- 8 tunable parameters per tab/blank (see Parameterization below)
- JSON import/export for batch generation
- Transparent PNG output

### 2. puzzle_cutter.py

Cuts any image into interlocking jigsaw puzzle pieces with realistic Bezier curve edges. Unlike `piece_generator.py` which creates individual piece shapes, this script generates a complete grid of interlocking pieces from a source image.

**Basic Usage:**
```bash
cd scripts

# Cut into approximately 25 pieces (grid auto-calculated)
python puzzle_cutter.py photo.jpg --pieces 25 --output-dir output/

# Specify exact grid dimensions
python puzzle_cutter.py photo.jpg --rows 4 --cols 6 --output-dir output/

# With reproducible random seed
python puzzle_cutter.py photo.jpg --pieces 100 --seed 42

# Custom padding for larger tabs
python puzzle_cutter.py photo.jpg --pieces 50 --padding 30
```

**All Options:**
```
positional arguments:
  input_image           Path to the input image file

options:
  --pieces N            Target number of pieces (default: 100)
  --rows R              Explicit number of rows (use with --cols)
  --cols C              Explicit number of columns (use with --rows)
  --output-dir DIR      Output directory for pieces (default: output/)
  --padding N           Padding around each piece in pixels (default: 20)
  --seed N              Random seed for reproducible results
  --points-per-curve N  Points per Bezier curve (default: 20)
  --prefix STR          Filename prefix (default: piece)
  --quiet               Suppress progress output
```

**Output Format:**
- Individual PNG files: `{prefix}_r{row:02d}_c{col:02d}.png`
- RGBA format with transparent backgrounds
- Each piece includes padding to accommodate tab protrusions

**Key Features:**
- **Auto grid calculation**: Specify target piece count, algorithm finds optimal rows × cols for roughly square pieces based on image aspect ratio
- **Shared edge generation**: Interior edges are generated once and shared between adjacent pieces, ensuring perfect interlocking (tab on one piece matches blank on neighbor)
- **Anti-aliased cutting**: 4× supersampling for smooth piece edges
- **Border handling**: Border pieces have flat edges, interior pieces have curved tabs/blanks
- **Reproducible**: Use `--seed` for consistent edge generation across runs

**How It Works:**

1. **Grid Dimension Calculation**: Given a target piece count and image dimensions, calculates optimal rows × cols:
   ```
   cols = sqrt(target_pieces × aspect_ratio)
   rows = sqrt(target_pieces / aspect_ratio)
   ```

2. **Edge Grid Generation**: Creates two arrays of shared edges:
   - Horizontal edges: `(rows+1) × cols` — boundaries between vertically adjacent pieces
   - Vertical edges: `rows × (cols+1)` — boundaries between horizontally adjacent pieces
   - Border edges are flat (straight lines)
   - Interior edges get random tab/blank shapes using the same Bezier curve system as `piece_generator.py`

3. **Edge Interlocking**: Each interior edge is generated once with a random direction (tab or blank). Adjacent pieces share the same edge curves, but traversed in opposite directions—so if piece A has a tab on its right edge, piece B (to the right) automatically has a matching blank on its left edge.

4. **Image Cutting**: For each piece position:
   - Assembles the 4 boundary edges (top, right, bottom, left)
   - Samples points along the Bezier curves to create a polygon
   - Creates an anti-aliased mask using 4× supersampling
   - Extracts the piece from the source image with transparent background

**Example Output:**
```
$ python puzzle_cutter.py sunset.jpg --pieces 25 --seed 42
Input image: sunset.jpg (1920x1080)
Grid: 4 rows x 7 cols = 28 pieces
Piece size: ~274x270 pixels
Random seed: 42
Generating edge grid...
Cutting 28 pieces...
Saved 28 pieces to output/
```

**Supporting Modules:**
- **edge_grid.py**: `EdgeGrid` class and grid generation functions
- **image_masking.py**: `CoordinateMapper` class and image cutting utilities

### 3. Modular Libraries

The core logic is split into several modules:
- **models.py**: Defines `BezierCurve`, `TabParameters`, and `PieceConfig` dataclasses.
- **geometry.py**: Core math for generating tab edges using the 5-curve "mushroom" model.
- **rendering.py**: Matplotlib-based rendering to PNG and curve visualization.
- **io_utils.py**: JSON serialization utilities.
- **comparison.py**: Logic for comparing generated pieces against reference images.
- **edge_grid.py**: Grid generation for puzzle cutting (used by `puzzle_cutter.py`).
- **image_masking.py**: Image cutting utilities with polygon masks (used by `puzzle_cutter.py`).

### 4. shape_comparator.py

Quantitatively compares generated pieces against reference images using contour-based metrics.

```bash
cd scripts

# Compare a single piece (1-6)
python shape_comparator.py 1

# Compare all pieces
python shape_comparator.py --all
```

**Metrics:**
- **IoU (Intersection over Union)**: Shape overlap (1.0 = perfect)
- **Mean Contour Distance**: Average distance between contour points (0 = perfect)
- **Hausdorff Distance**: Maximum distance between contours (0 = perfect)

### 5. visualize_comparison.py

Visual debugging tool that overlays generated contours on top of reference images.

```bash
cd scripts
python visualize_comparison.py 1
```

### 6. optimize_parameters.py

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

Each tab/blank is controlled by 8 semantic parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `neck_width` | Width of the narrow waist | 0.06 - 0.12 |
| `bulb_width` | Width of the bulb head | 0.20 - 0.32 |
| `height` | How far the tab protrudes | 0.12 - 0.30 |
| `position` | Where along edge the tab is centered | 0.35 - 0.65 |
| `neck_ratio` | Height of waist as proportion of total | 0.15 - 0.55 |
| `curvature` | How rounded the bulb is | 0.30 - 1.0 |
| `asymmetry` | Tilt direction (-1=left, 0=center, +1=right) | -0.15 - 0.15 |
| `corner_slope`| Tangent angle at corners | 0.00 - 0.20 |

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
    corner_slope=0.10, # Realistic corner transitions
)
```

## JSON Configuration Format
...
## Dependencies
...
## Technical Details

The shape generation uses **5 cubic Bezier curves** per tab/blank to create the "mushroom" shape:
1. **Curve 1**: Edge start to neck base (flat entry with adjustable corner slope)
2. **Curve 2**: Neck base through waist to bulb side (S-curve creating the locking mechanism)
3. **Curve 3**: Bulb top (semicircular arc)
4. **Curve 4**: Bulb side through waist back to neck base (mirror of Curve 2)
5. **Curve 5**: Neck base to edge end (mirror of Curve 1)

This approach provides:
- Smooth, C1-continuous curves
- Natural-looking, symmetric or asymmetric "mushroom" shapes
- Robust locking mechanism appearance
- 25 control points per edge (100 per piece)
- Effectively controlled by 8 intuitive parameters

See `PUZZLE_PIECE_SHAPE_GENERATION.md` for detailed technical notes on the Bezier curve mathematics and design decisions.
