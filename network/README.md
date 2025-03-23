# Puzzle Generator Utilities

This collection of utilities helps generate and process puzzle images for training machine learning models.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Utilities

### 1. Puzzle Piece Generator

Generates puzzle pieces from complete puzzle images.

#### Usage

```bash
python puzzle_generator.py path/to/puzzle_image.jpg
```

### 2. Puzzle Image Resizer

Resizes all puzzle images in a dataset to 512x512 pixels.

#### Usage

```bash
python resize_puzzles.py path/to/dataset
```

To specify a different output directory:

```bash
python resize_puzzles.py path/to/dataset --output-dir path/to/output
```

To specify custom dimensions:

```bash
python resize_puzzles.py path/to/dataset --width 640 --height 640
```

### 3. Puzzle Piece Visualizer

Visualizes a puzzle piece's placement on the original puzzle with a red outline. This utility:
- Shows the piece in its original position with a red bounding box
- Makes the piece semi-transparent (50% opacity) for better visualization
- Automatically rotates the piece back to its original orientation
- Handles different piece formats and rotations

#### Usage

```bash
python visualize_piece.py path/to/puzzle.jpg path/to/piece.png
```

To specify a custom output path:

```bash
python visualize_piece.py path/to/puzzle.jpg path/to/piece.png --output path/to/output.png
```

By default, the visualization is saved as `temp_{piece_name}.png` in the same directory as the puzzle image.

## Output Format

The generated pieces will be saved in the following directory structure:

```
datasets/example/pieces/
└── puzzle_name/
    ├── piece_001_x100_y150_x170_y220_r0.png
    ├── piece_002_x200_y250_x270_y320_r90.png
    └── ...
```

Each piece filename encodes its position and rotation:
- `piece_001`: Piece ID
- `x100_y150_x170_y220`: Bounding box coordinates (x1, y1, x2, y2)
- `r0`: Rotation in degrees (0, 90, 180, or 270)

## Examples

```bash
# Generate ~500 pieces from example puzzle
python puzzle_generator.py datasets/example/puzzle_001.jpg

# Generate ~1000 pieces from all puzzles in example folder
python puzzle_generator.py datasets/example --pieces 1000

# Resize all puzzles in the example dataset to 512x512
python resize_puzzles.py datasets/example --output-dir datasets/example/resized

# Resize all puzzles to 800x600
python resize_puzzles.py datasets/example --output-dir datasets/example/resized --width 800 --height 600

# Visualize a puzzle piece placement
python visualize_piece.py datasets/example/resized/puzzle_002.jpg datasets/example/pieces/puzzle_002/piece_001_x73_y0_x146_y73_r180.png
```
