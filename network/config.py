#!/usr/bin/env python
"""Configuration parameters for puzzle piece prediction model."""

from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """Get default configuration parameters.

    Returns:
        Dictionary of configuration parameters
    """
    config = {
        # Data configuration
        "data": {
            "data_dir": "datasets/example",
            "batch_size": 32,
            "num_workers": 4,
            "piece_size": (128, 128),  # Reduced from 224 for faster training
            "puzzle_size": (128, 128),  # Reduced from 224 for faster training
        },
        # Model configuration
        "model": {
            "backbone_name": "resnet18",  # Changed from resnet50 for faster training
            "pretrained": True,
            "learning_rate": 3e-3,  # Increased from 1e-4 based on LR finder results
            "position_weight": 1.0,
            "rotation_weight": 1.0,
        },
        # Training configuration
        "training": {
            "max_epochs": 100,
            "early_stop_patience": 10,
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
            "experiment_name": "dual_input_model",
        },
    }

    return config
