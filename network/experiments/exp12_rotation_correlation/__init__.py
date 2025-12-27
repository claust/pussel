"""Experiment 12: Rotation Correlation.

This experiment fixes the fundamental flaw in exp10 and exp11: the rotation
head only saw piece features, completely ignoring the puzzle. Since rotation
is a relationship between piece and puzzle (not intrinsic to the piece),
both experiments failed.

The fix: Rotation Correlation - compare piece features against puzzle features
for each rotation, selecting the rotation with the highest correlation/match.
"""
