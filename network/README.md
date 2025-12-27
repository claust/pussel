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
- **isort**: Import sorting (using black-compatible profile)
- **flake8**: Linting with additional plugins:
  - flake8-docstrings
  - flake8-import-order
  - flake8-bugbear
- **pyright**: Static type checking (same as VS Code Pylance)
- **pre-commit**: Git hooks for code quality checks

### Running Code Quality Checks

```bash
# Format code (in correct order)
black .
isort . --profile black  # Use black-compatible profile

# Alternative: Run both formatters in one go
black . && isort . --profile black

# Run linting
flake8

# Run type checking
pyright .
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

#### 4. Image Downloader

Downloads puzzle images from a CSV file containing Unsplash URLs.

https://github.com/unsplash/datasets

```bash
# Download 1000 images (default)
python download_images.py

# Download a specific number of images
python download_images.py --count 500

# Specify output directory
python download_images.py --output-dir datasets/custom/raw

# Use a different CSV file
python download_images.py --csv-file my_dataset.csv
```

Features:

- Skips images that already exist in the output folder
- Shows progress bar during download
- Handles download failures gracefully
- Names images sequentially as `puzzle_001.jpg`, `puzzle_002.jpg`, etc.

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
# Download 1000 images from Unsplash dataset
python download_images.py --count 1000

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

Then open http://localhost:6006/ in your web browser to view training metrics,
including:

- Position and rotation losses
- Rotation accuracy
- Confusion matrix

### Model Checkpoints

Model checkpoints are saved in the `network/checkpoints/` directory. The
training process saves:

- The top 3 best-performing models based on validation loss
- The latest model checkpoint as `last.ckpt`

You can resume training from a checkpoint by using the Lightning CLI or
implementing a custom resume function.

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

## Performance Tuning (Apple Silicon)

### Understanding Resource Usage

On Apple Silicon Macs (M1/M2/M3/M4), the GPU uses **unified memory** shared with the CPU. There's no separate VRAM - both CPU and GPU access the same memory pool.

**CPU percentage interpretation:**
- 100% = 1 core fully utilized
- 400% = 4 cores busy
- Check your core count: `sysctl -n hw.ncpu`

**Check MPS (GPU) memory from within training:**
```python
import torch
print(f"MPS allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
print(f"MPS driver: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GB")
```

### Batch Size Guidelines

| Total RAM | Recommended Batch Size | Notes |
|-----------|------------------------|-------|
| 8 GB | 32-64 | Leave headroom for OS |
| 16 GB | 64-128 | Good balance |
| 32 GB+ | 128-256 | Can push higher |

**When increasing batch size, scale learning rate proportionally:**
```bash
# Doubling batch size (64 → 128): double the learning rates
python train.py --batch-size 128 --backbone-lr 2e-4 --head-lr 2e-3
```

### Data Loading Optimization

Use `num_workers` to parallelize data loading across CPU cores:

```bash
# Use 4 CPU cores for data loading (recommended for 10-core M4)
python train.py --num-workers 4
```

**Guidelines:**
- `num_workers=0`: Single-threaded (default, often a bottleneck)
- `num_workers=4`: Good starting point for most Macs
- `num_workers=8`: For machines with 10+ cores and fast SSD

**Signs you need more workers:**
- Low GPU utilization
- CPU usage stuck at ~100-200% (only 1-2 cores busy)
- GPU waiting for data between batches

### Memory vs Speed Trade-offs

| Setting | Memory Impact | Speed Impact |
|---------|---------------|--------------|
| ↑ Batch size | ↑ Higher | ↑ Faster (better GPU utilization) |
| ↑ num_workers | ↑ Slightly higher | ↑ Faster (parallel data loading) |
| ↑ Image size | ↑ Higher | ↓ Slower (more computation) |
| ↑ Model size | ↑ Higher | ↓ Slower (more parameters) |

### Quick Performance Check

```bash
# Monitor training process
ps aux | grep "python.*train" | grep -v grep

# Check system memory pressure
memory_pressure

# Get CPU core count
sysctl -n hw.ncpu
```
