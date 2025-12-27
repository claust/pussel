"""Experiment 9: Fine-tune backbone for improved cross-puzzle generalization.

This experiment builds on exp7's DualInputRegressorWithCorrelation architecture,
but unfreezes the MobileNetV3-Small backbone with differential learning rates
to learn task-specific features.

Phase 2 of the coarse regression approach.
"""
