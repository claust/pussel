# Puzzle Piece Prediction Model

This project consists of two main components:
1. Utilities for generating and processing puzzle datasets
2. A CNN-based model for predicting puzzle piece positions and rotations

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Development

### Code Quality Tools

The project uses several tools to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting with additional plugins:
  - flake8-docstrings
  - flake8-import-order
  - flake8-bugbear
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality checks

### Running Code Quality Checks

```bash
# Format code
black .
isort .

# Run linting
flake8

# Run type checking
mypy .
```

## Dataset Preparation

### Dataset Utilities

The following utilities help prepare puzzle datasets for training:

#### 1. Puzzle Piece Generator

Generates puzzle pieces from complete puzzle images.

```bash
python puzzle_generator.py path/to/puzzle_image.jpg
```

#### 2. Puzzle Image Resizer

Resizes all puzzle images in a dataset to a standard size.

```bash
# Resize to default 512x512
python resize_puzzles.py path/to/dataset

# Specify output directory
python resize_puzzles.py path/to/dataset --output-dir path/to/output

# Specify custom dimensions
python resize_puzzles.py path/to/dataset --width 640 --height 640
```

#### 3. Puzzle Piece Visualizer

Visualizes a puzzle piece's placement on the original puzzle with a red outline.

```bash
# Basic usage
python visualize_piece.py path/to/puzzle.jpg path/to/piece.png

# Custom output path
python visualize_piece.py path/to/puzzle.jpg path/to/piece.png --output path/to/output.png
```

Features:
- Shows the piece in its original position with a red bounding box
- Makes the piece semi-transparent (50% opacity) for better visualization
- Automatically rotates the piece back to its original orientation
- Handles different piece formats and rotations

### Dataset Output Format

The generated pieces will be saved in the following directory structure:

```
datasets/example/
├── metadata.csv
├── train.txt
├── val.txt
├── puzzle_001.jpg
├── puzzle_002.jpg
├── ...
└── pieces/
    ├── puzzle_001_piece_001_x100_y150_x170_y220_r0.png
    ├── puzzle_001_piece_002_x200_y250_x270_y320_r90.png
    └── ...
```

Each piece filename encodes its position and rotation:
- `puzzle_001_piece_001`: Piece ID (piece 001 of puzzle 001)
- `x100_y150_x170_y220`: Bounding box coordinates (x1, y1, x2, y2)
- `r0`: Rotation in degrees (0, 90, 180, or 270)

### Dataset Preparation Examples

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
python visualize_piece.py datasets/example/puzzle_002.jpg datasets/example/pieces/puzzle_002_piece_001_x73_y0_x146_y73_r180.png
```

## Model Training

### Running the Training

To train the model using the default configuration:

```bash
cd network
python train.py
```

### Customizing Training Parameters

You can customize the training parameters through command-line arguments:

```bash
# Use a different backbone
python train.py --backbone efficientnet_b3 --pretrained True

# Customize loss weights
python train.py --position_weight 1.5 --rotation_weight 0.5

# Change batch size or data directory
python train.py --batch_size 64 --data_dir datasets/custom/processed

# Run for fewer epochs or change early stopping patience
python train.py --max_epochs 50 --early_stop_patience 5
```

### Viewing Training Progress with TensorBoard

To monitor training progress, use TensorBoard:

```bash
cd network
tensorboard --logdir=logs
```

Then open http://localhost:6006/ in your web browser to view training metrics, including:
- Position and rotation losses
- Rotation accuracy
- Confusion matrix

### Model Checkpoints

Model checkpoints are saved in the `network/checkpoints/` directory. The training process saves:
- The top 3 best-performing models based on validation loss
- The latest model checkpoint as `last.ckpt`

You can resume training from a checkpoint by using the Lightning CLI or implementing a custom resume function.

### Training Examples

```bash
# Train the model with default settings
cd network
python train.py

# Train with a different backbone and increased position loss weight
cd network
python train.py --backbone efficientnet_b0 --position_weight 2.0

# Train with a smaller batch size for memory-constrained systems
cd network
python train.py --batch_size 16

# View training progress
cd network
tensorboard --logdir=logs
```
