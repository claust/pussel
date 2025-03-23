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
- **Backbone Network**: CNN architecture (ResNet-50 or EfficientNet-B3)
- **Input**: RGB puzzle piece images, resized to 224×224 pixels
- **Feature Maps**: Output from the backbone will be high-dimensional feature representations

#### 2. Position Prediction Head
- **Architecture**: Multiple fully-connected layers following the CNN backbone
- **Output**: 4 values representing normalized coordinates (x1, y1, x2, y2) of piece position
- **Activation**: Sigmoid to bound predictions between 0 and 1 (will be scaled to original image dimensions)

#### 3. Rotation Prediction Head
- **Architecture**: Fully-connected layers branching from the same backbone
- **Output**: 4-class classification (0°, 90°, 180°, 270°)
- **Activation**: Softmax to produce rotation class probabilities

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
- **Primary Loss**: Mean Squared Error (MSE) or Smooth L1 Loss
- **Formula**: L_pos = MSE(predicted_position, ground_truth_position)

#### Rotation Loss
- **Primary Loss**: Categorical Cross-Entropy
- **Formula**: L_rot = CrossEntropy(predicted_rotation, ground_truth_rotation)

#### Combined Loss
- **Formula**: L_total = α * L_pos + β * L_rot
- **Hyperparameters**: α, β to balance the contribution of each loss

### Training Strategy

#### Setup
- **Batch Size**: 32-64 depending on memory constraints
- **Optimizer**: Adam with learning rate 1e-4
- **Scheduler**: ReduceLROnPlateau or CosineAnnealingLR
- **Epochs**: 100-200 (with early stopping)

#### Curriculum
1. **Stage 1**: Train on easy pieces (edge pieces, distinctive pieces)
2. **Stage 2**: Include more challenging pieces
3. **Optional Stage 3**: Fine-tuning with pieces from unseen puzzles

### Evaluation Metrics

#### Position Accuracy
- **Mean Absolute Error**: Average pixel distance between predicted and ground truth positions
- **IOU (Intersection over Union)**: For bounding box overlap measurement

#### Rotation Accuracy
- **Classification Accuracy**: Percentage of correctly predicted rotations
- **Confusion Matrix**: To analyze rotation prediction patterns

#### Combined Metric
- **Correct Placement Rate**: Percentage of pieces with both position and rotation correct (within thresholds)
- **Placement Error**: Weighted combination of position and rotation errors

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
