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
            "data_dir": "datasets/example/processed",
            "batch_size": 32,
            "num_workers": 4,
            "input_size": (224, 224),
        },
        # Model configuration
        "model": {
            "backbone_name": "resnet50",
            "pretrained": True,
            "learning_rate": 1e-4,
            "position_weight": 1.0,
            "rotation_weight": 1.0,
        },
        # Training configuration
        "training": {
            "max_epochs": 100,
            "early_stop_patience": 10,
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
        },
    }

    return config
