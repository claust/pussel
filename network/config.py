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
            "data_dir": "datasets",
            "batch_size": 32,
            "num_workers": 4,
            "piece_size": (224, 224),  # Model input for pieces (saved at this size)
            "puzzle_size": (256, 256),  # Puzzle context image size
        },
        # Model configuration
        "model": {
            "backbone_name": "mobilenetv3_small_100",
            "pretrained": True,
            "learning_rate": 1e-3,  # Increased from 1e-4 based on LR finder results
            "position_weight": 1.0,
            "rotation_weight": 1.0,
            "use_spatial_correlation": True,  # Enable spatial correlation for position
        },
        # Training configuration
        "training": {
            "max_epochs": 10,
            "early_stop_patience": 4,
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
            "experiment_name": "dual_input_model",
        },
    }

    return config
