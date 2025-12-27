# Experiment 5: Cross-Puzzle Generalization

## Objective

Test whether a neural network trained on one puzzle can generalize to predict piece positions on a completely different puzzle.

**Key Question**: Can a model learn a general "matching function" that finds where a piece belongs in ANY puzzle, or does it only memorize puzzle-specific patterns?

## Background

Previous experiments showed that:
- **exp2**: A simple CNN can memorize 950 pieces from a single puzzle (MSE loss: 0.007)
- **exp4**: A larger backbone (256-dim features) achieves 99.3% accuracy on 950-class cell classification

However, perfect training accuracy doesn't prove the model learned generalizable features. It could simply be memorizing which textures map to which positions for that specific puzzle.

## Original Design Flaw (Fixed)

The original experiment had a fatal flaw: the model only received the **piece image** as input, making cross-puzzle generalization impossible by design.

### The Fix: Dual-Input Architecture

The corrected experiment uses a **DualInputCellClassifier** that receives BOTH:
1. **Piece image** (64x64): The puzzle piece to locate
2. **Puzzle image** (256x256): The complete puzzle as context

This enables learning a **matching function**: "Given this piece and this puzzle, where does the piece fit?"

## Architecture

```
DualInputCellClassifier (~1.6M parameters)
├── Shared CNN Backbone (389K params)
│   ├── Conv2D(3→32, stride=2) + BN + ReLU   → 32x32
│   ├── Conv2D(32→64, stride=2) + BN + ReLU  → 16x16
│   ├── Conv2D(64→128, stride=2) + BN + ReLU → 8x8
│   └── Conv2D(128→256, stride=2) + BN + ReLU → 4x4
├── Spatial Correlation Module
│   ├── Projects piece/puzzle features to 64-dim
│   ├── Computes correlation map: where does piece match?
│   └── Processes correlation into position features
├── Cross-Attention Fusion
│   ├── Layer normalization
│   ├── Attention-weighted puzzle features
│   └── Fusion network (512-dim output)
└── Classification Head
    └── Linear(320→512→950) for cell prediction
```

## Experimental Design

1. **Training (puzzle_001)**:
   - Input: (piece_from_001, puzzle_001_image)
   - Output: cell position in puzzle_001 (950 classes)
   - Model learns to match piece textures to puzzle regions

2. **Testing (puzzle_002)**:
   - Input: (piece_from_002, puzzle_002_image)
   - Output: cell position in puzzle_002 (950 classes)
   - Tests if matching function generalizes

## Results

| Metric | Training (puzzle_001) | Test (puzzle_002) |
|--------|----------------------|-------------------|
| Top-1 Accuracy | 11.47% | 0.11% |
| Top-5 Accuracy | 36.32% | 0.53% |
| Loss | 5.84 | 38.75 |

- **Random baseline**: 0.105% (1/950)
- **Generalization gap**: 11.37%
- **Training time**: 552.6 seconds (200 epochs)
- **Peak training accuracy**: ~45% (epoch 190)

## Conclusion

**The dual-input architecture does NOT enable cross-puzzle generalization.**

Despite providing the puzzle image as context, the model:
1. **Can use puzzle context**: Training accuracy reaches ~45%, showing the model uses both inputs
2. **Does NOT learn generalizable matching**: Test accuracy (0.11%) equals random chance
3. **Overfits to training puzzle**: The model memorizes puzzle_001-specific patterns

## Analysis

### Why Doesn't It Generalize?

Several factors likely contribute to the failure:

1. **Resolution mismatch**:
   - Pieces are 64x64 pixels, but individual cells in the 256x256 puzzle are only ~7x10 pixels
   - The puzzle image lacks sufficient detail to discriminate piece locations

2. **Texture-based shortcuts**:
   - Instead of learning edge/shape matching, the model may learn texture statistics
   - Puzzle_001's texture patterns don't transfer to puzzle_002

3. **Single-puzzle training**:
   - Training on one puzzle provides no variation in puzzle context
   - The model has no incentive to learn puzzle-invariant features

4. **Task difficulty**:
   - 950-class classification is extremely fine-grained
   - Adjacent cells may have nearly identical textures

### What Would Help?

1. **Multi-puzzle training**: Train on multiple puzzles simultaneously
2. **Higher puzzle resolution**: Use larger puzzle images (512x512 or 1024x1024)
3. **Coarser task**: Start with quadrant prediction before fine-grained cells
4. **Data augmentation**: Color jittering, texture distortions to force shape-based matching
5. **Contrastive learning**: Learn piece-to-region matching explicitly

## Files

- `model.py` - DualInputCellClassifier with spatial correlation and cross-attention
- `dataset.py` - PuzzleDataset returning (piece, puzzle, cell_index) tuples
- `train.py` - Training script with cross-puzzle evaluation
- `visualize.py` - Visualization utilities

## Running the Experiment

```bash
cd network
source ../venv/bin/activate

# Run with default settings (200 epochs)
python -m experiments.exp5_cross_puzzle_generalization.train --epochs 200

# Run multiple times for statistical significance
python -m experiments.exp5_cross_puzzle_generalization.train --epochs 200 --runs 3
```

## Key Insight

This experiment reveals a fundamental challenge in puzzle solving: **single-puzzle training is insufficient for learning generalizable matching**.

While the original single-input design was fatally flawed (making the task impossible), the fixed dual-input design shows that even with proper inputs, the model fails to generalize. This suggests that:

1. **The matching problem is inherently difficult** - visual similarity between piece and puzzle region requires sophisticated reasoning
2. **Diversity in training data is essential** - models need exposure to multiple puzzles to learn invariant features
3. **Architecture alone is not enough** - having the right inputs doesn't guarantee learning the right function
