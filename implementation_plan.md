# CNN-Based Puzzle Piece Prediction: POC Implementation Plan

## Project Structure
```
pussel/
└── network/
    ├── puzzle_generator.py  # Existing preprocessing script
    ├── specification.md     # Existing specification
    ├── model.py             # LightningModule with timm backbone
    ├── dataset.py           # Dataset handling with Albumentations ✓
    ├── train.py             # Training script with Lightning Trainer
    └── config.py            # Configuration parameters
```

## Implementation Tasks

### 1. Setup Project & Dependencies
- [x] Virtual environment is ready
- [x] Install required packages:
  ```
  torch torchvision
  pytorch-lightning
  timm
  albumentations
  pillow
  pandas
  matplotlib
  ```

### 2. Dataset Implementation (network/dataset.py)
- [x] Create PyTorch Dataset class that reads from metadata.csv
- [x] Implement basic transforms using Albumentations
- [x] Create DataModule for Lightning with train/val loading

### 3. Model Implementation (network/model.py)
- [ ] Create PuzzleCNN LightningModule with timm ResNet50 backbone
- [ ] Implement position prediction head (bounding box coordinates)
- [ ] Implement rotation prediction head (4-class classification)
- [ ] Define loss functions (MSE, CrossEntropy)
- [ ] Add training_step, validation_step with metrics

### 4. Training Setup (network/train.py)
- [ ] Configure Lightning Trainer
- [ ] Add callbacks (model checkpoint, early stopping)
- [ ] Implement simple logging of metrics
- [ ] Add CLI arguments for key parameters

### 5. Configuration (network/config.py)
- [ ] Model configuration (backbone, head dims)
- [ ] Training parameters
- [ ] Data paths and parameters

### 6. Initial Testing & Visualization
- [ ] Add simple inference function in model.py
- [ ] Implement basic visualization to show predictions on puzzle

## Timeline
- Dataset implementation: ✓ Done
- Model implementation: 1-2 days
- Training setup: 1 day
- Initial testing & visualization: 1 day

## Future Improvements (Post-POC)
- Experiment with different backbones
- Implement more sophisticated data augmentation
- Add detailed evaluation metrics
- Create standalone prediction script
