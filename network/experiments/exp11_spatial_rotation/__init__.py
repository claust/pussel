"""Experiment 11: Spatial Rotation Head.

This experiment fixes the architectural flaw from exp10 by using a spatial
rotation head that preserves feature map structure, allowing the model to
distinguish between rotational orientations (e.g., 0 vs 180 degrees).

Key changes:
1. SpatialRotationHead: Conv2d layers before flattening (vs pooling first)
2. Random rotation sampling: 4 samples/puzzle (vs 16 in exp10)
"""
