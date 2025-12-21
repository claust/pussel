# Preprocessing Integration Plan for puzzle_generator.py

## Overview

This document outlines a plan to extend the current `puzzle_generator.py` tool
to include all necessary preprocessing steps for the deep learning model as
specified in our CNN-based approach. The goal is to create a comprehensive data
preparation pipeline that generates training-ready datasets.

## Current Functionality of puzzle_generator.py

- Generates puzzle pieces from complete puzzle images
- Creates masks to identify individual pieces
- Extracts pieces with proper alpha channel
- Applies random rotations (0°, 90°, 180°, 270°)
- Names pieces with position and rotation information
- Standardizes puzzle and piece sizes
- Normalizes pixel values
- Supports data augmentation via Albumentations

## Required Preprocessing Steps from Specification

1. ✅ Resize all puzzle images to a standard size (512×512 pixels)
2. ✅ Extract pieces according to their bounding box coordinates
3. ✅ Normalize pixel values (0-1 or standardization)
4. ✅ Create paired datasets (piece, complete puzzle, ground truth
   position/rotation)

## Integration Plan

### 1. Modify Main Function Structure ✅ (Completed)

- ✅ Add preprocessing command line arguments
- ✅ Create a unified pipeline for piece generation and preprocessing
- ✅ Remove the separate preprocessing mode in favor of an integrated workflow

### 2. Required Functions to Implement

#### A. Image Standardization ✅ (Completed)

A function to resize puzzle images to standard dimensions (e.g., 512×512
pixels). This ensures all puzzles have consistent dimensions for the model
training process, regardless of their original size.

#### B. Piece Normalization and Processing ✅ (Completed)

A comprehensive function to process individual puzzle pieces that will:

- ✅ Resize pieces to a standard size (e.g., 224×224 pixels)
- ✅ Normalize pixel values to the 0-1 range
- ✅ Handle alpha channel properly
- ✅ Extract and format metadata (position, rotation)
- ✅ Prepare the piece for model input
- ✅ Return both the processed image and its associated metadata

#### C. Direct Training Data Generation ✅ (Completed)

A streamlined function that will handle the entire pipeline:

- ✅ Take a puzzle image as input
- ✅ Generate pieces based on a mask
- ✅ Process each piece as it's generated
- ✅ Save processed pieces to the output directory
- ✅ Record metadata in a CSV file
- ✅ Split pieces into training and validation sets
- ✅ No intermediate storage of raw pieces

#### D. Data Augmentation Capabilities ✅ (Completed)

Integration with the Albumentations library for efficient and robust
augmentation:

- ✅ Brightness and contrast adjustments
- ✅ Color shifts (RGB channel adjustments)
- ✅ Affine transformations (rotation, shear, zoom)
- ✅ Random cropping and scaling
- ✅ All augmentations configurable via command-line parameters

### 3. Streamlined Dataset Structure ✅ (Completed)

```
datasets/
  ├── example/
  │   ├── raw/                # Original unprocessed data
  │   │   └── puzzles/        # Original full puzzle images
  │   └── processed/          # All processed data
  │       ├── puzzles/        # Standardized puzzle images (512×512)
  │       ├── metadata.csv    # Consolidated metadata for all pieces
  │       ├── train.txt       # List of piece IDs for training
  │       ├── val.txt         # List of piece IDs for validation
  │       └── pieces/         # All processed pieces in one directory
```

### 4. Metadata Format ✅ (Completed)

The metadata.csv file contains comprehensive information about each piece:

- ✅ Unique piece identifier
- ✅ Source puzzle identifier
- ✅ Relative path to the piece image
- ✅ Original bounding box coordinates (x1, y1, x2, y2)
- ✅ Rotation value (0, 90, 180, 270)
- ✅ Normalized coordinates (for direct model input)
- ✅ Dataset split assignment (train or validation)

## Implementation Strategy

### Phase 1: Code Refactoring ✅ (Completed)

1. ✅ Refactor existing code to improve modularity
2. ✅ Add type hints and improve documentation
3. ✅ Integrate piece generation and processing into a single workflow
4. ✅ Remove redundant storage of raw pieces

### Phase 2: Add New Functionality ✅ (Completed)

1. ✅ Implement standardization functions
2. ✅ Implement direct piece processing functions
3. ✅ Integrate Albumentations library for data augmentation
4. ✅ Create streamlined dataset organization structure

### Phase 3: User Interface ✅ (Completed)

1. ✅ Simplify command-line interface to focus on the unified pipeline
2. ✅ Add options for customizing processing parameters
3. ✅ Add progress reporting through console output

### Phase 4: Testing & Validation

1. Verify processed data format matches model requirements
2. Test with sample puzzles
3. Measure preprocessing speed and optimize if needed

## Command-Line Interface Examples

```bash
# Generate and process pieces from a puzzle
python puzzle_generator.py datasets/example/puzzle_001.jpg --output-dir datasets/example/processed

# Generate and process pieces from all puzzles in a directory
python puzzle_generator.py datasets/example/raw/puzzles --output-dir datasets/example/processed

# Generate with custom processing parameters
python puzzle_generator.py datasets/example/raw/puzzles --output-dir datasets/example/processed --piece-size 224 224 --puzzle-size 512 512 --validation-split 0.2

# Generate with augmentation
python puzzle_generator.py datasets/example/raw/puzzles --output-dir datasets/example/processed --augment --brightness-range 0.8 1.2 --rotation-range 15
```

## Additional Considerations

1. **Memory-Efficient Processing**: Process one piece at a time to avoid memory
   issues
2. **Streaming Approach**: Generate → process → store in a single pipeline
3. **Parallel Processing**: Use multiprocessing for faster preprocessing (future
   enhancement)
4. **Error Handling**: Robust error handling for malformed or missing data
5. **Logging**: Comprehensive logging for debugging and progress tracking
   (future enhancement)
6. **Configurability**: Allow configuration via file for complex preprocessing
   workflows
7. **Dependencies**: Required packages now include albumentations for data
   augmentation
