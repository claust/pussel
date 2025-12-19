# CNN-Based Puzzle Piece Position and Rotation Prediction

## Problem Definition
Create a deep learning model that can predict the correct position and rotation of a single puzzle piece within its original puzzle.

## Dataset Structure
- **Complete Puzzles**: Located in `datasets/example/` directory
- **Puzzle Pieces**: Stored in `datasets/example/pieces/puzzle_XXX/` directories
- **Piece Format**: `piece_ID_xX1_yY1_xX2_yY2_rROTATION.png`
  - X1, Y1, X2, Y2: Bounding box coordinates
  - ROTATION: One of four possible rotations (0°, 90°, 180°, 270°)

## CNN-Based Approach Specification

### Model Architecture

#### 1. Feature Extraction
- **Dual Backbone Network**: Two identical CNN backbones (ResNet-50 or other timm models)
  - One backbone processes the puzzle piece image
  - One backbone processes the complete puzzle image
- **Input**: RGB images resized to 224×224 pixels
- **Feature Maps**: High-dimensional feature representations from both backbones

#### 2. Feature Fusion
- **Concatenation**: Features from both backbones are concatenated
- **Dimensionality Reduction**:
  - First layer: 1024 units with ReLU activation and 0.3 dropout
  - Second layer: 512 units with ReLU activation and 0.2 dropout
- **Output**: 512-dimensional fused feature vector

#### 3. Position Prediction Head
- **Architecture**: Two fully-connected layers
  - First layer: 256 units with ReLU activation and 0.1 dropout
  - Second layer: 4 units (x1, y1, x2, y2) with Sigmoid activation
- **Output**: Normalized coordinates (0-1) representing piece position

#### 4. Rotation Prediction Head
- **Architecture**: Two fully-connected layers
  - First layer: 256 units with ReLU activation and 0.1 dropout
  - Second layer: 4 units (one per rotation class)
- **Output**: Logits for 4-class classification (0°, 90°, 180°, 270°)

### Data Preprocessing and Augmentation

#### Preprocessing
1. Resize all puzzle images to a standard size (e.g., 512×512)
2. Extract pieces according to their bounding box coordinates
3. Normalize pixel values (0-1 or standardization)
4. Create paired datasets (piece, complete puzzle, ground truth position/rotation)

#### Augmentation (Training Only)
- Random brightness/contrast adjustments
- Slight color shifts
- Minor affine transformations (excluding rotation)
- Random cropping with padding

### Loss Functions

#### Position Loss
- **Primary Loss**: Mean Squared Error (MSE)
- **Additional Metric**: IoU (Intersection over Union) for bounding box evaluation

#### Rotation Loss
- **Primary Loss**: Cross-Entropy Loss
- **Additional Metrics**:
  - Classification Accuracy
  - Confusion Matrix for rotation classes

#### Combined Loss
- **Formula**: L_total = α * L_pos + β * L_rot
- **Default Weights**: α = 1.0, β = 1.0 (configurable)

### Training Strategy

#### Setup
- **Optimizer**: Adam with configurable learning rate (default: 1e-4)
- **Scheduler**: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 5 epochs
  - Monitors: validation total loss
- **Regularization**: Dropout at multiple levels (0.1-0.3)

#### Curriculum
1. **Stage 1**: Train on easy pieces (edge pieces, distinctive pieces)
2. **Stage 2**: Include more challenging pieces
3. **Optional Stage 3**: Fine-tuning with pieces from unseen puzzles

#### Evaluation
- **Position Metrics**:
  - Mean Squared Error
  - Mean IoU
- **Rotation Metrics**:
  - Classification Accuracy
  - Confusion Matrix
- **Combined Metrics**:
  - Total Loss (weighted combination)
  - Individual component losses

### Implementation Requirements

#### Dependencies
- PyTorch or TensorFlow
- OpenCV or PIL for image processing
- NumPy, Pandas for data handling
- Matplotlib for visualization

#### Computing Resources
- GPU with at least 8GB VRAM
- 16GB+ system RAM
- SSD storage for faster data loading

## Future Extensions

### Model Improvements
- Attention mechanisms for better feature extraction
- Multi-scale feature extraction for handling various piece sizes
- Self-supervised pretraining on puzzle dataset

### Dataset Expansion
- More diverse puzzle types and difficulty levels
- Varied lighting conditions and backgrounds
- Real-world captured puzzle pieces (vs. computer-generated)

### Application Integration
- Real-time puzzle solving assistance
- Integration with robotic systems for physical puzzle assembly
