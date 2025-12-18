#!/usr/bin/env python
"""Tests for CIoU loss implementation."""

import pytest
import torch
from torchvision.ops import complete_box_iou_loss


class TestCIoULoss:
    """Test suite for Complete IoU (CIoU) loss function."""

    def test_ciou_loss_identical_boxes(self):
        """Test that CIoU loss is zero for identical boxes."""
        pred_boxes = torch.tensor([[0.2, 0.3, 0.5, 0.6]])
        gt_boxes = torch.tensor([[0.2, 0.3, 0.5, 0.6]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        assert loss.shape == (1,)
        # Allow for floating-point precision errors
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5).all()

    def test_ciou_loss_non_overlapping_boxes(self):
        """Test CIoU loss with non-overlapping boxes (no gradient stalling)."""
        # Non-overlapping boxes (IoU = 0)
        pred_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
        gt_boxes = torch.tensor([[0.8, 0.8, 1.0, 1.0]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        assert loss.shape == (1,)
        # Loss should be positive (boxes don't overlap)
        assert loss > 0
        # Loss should be at most 2 (max possible: 1 - (-1) when worst case)
        assert loss <= 2.0

        # Verify gradients are non-zero (no gradient stalling)
        pred_boxes_grad = pred_boxes.clone().requires_grad_(True)
        loss_grad = complete_box_iou_loss(pred_boxes_grad, gt_boxes, reduction="sum")
        loss_grad.sum().backward()
        assert pred_boxes_grad.grad is not None
        assert not torch.all(pred_boxes_grad.grad == 0)

    def test_ciou_loss_partially_overlapping(self):
        """Test CIoU loss with partially overlapping boxes."""
        pred_boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        gt_boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        assert loss.shape == (1,)
        # Loss should be positive but less than 1 (partial overlap)
        assert 0 < loss < 1

    def test_ciou_loss_batch_processing(self):
        """Test that CIoU loss works with batches."""
        pred_boxes = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.2, 0.2, 0.5, 0.5],
                [0.3, 0.3, 0.6, 0.6],
            ]
        )
        gt_boxes = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.25, 0.25, 0.55, 0.55],
                [0.8, 0.8, 1.0, 1.0],
            ]
        )

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        assert loss.shape == (3,)
        # First box should have ~0 loss (identical)
        assert loss[0] < 0.01
        # Second box should have small loss (close match)
        assert 0 < loss[1] < 0.5
        # Third box should have larger loss (no overlap)
        assert loss[2] > 0.5

    def test_ciou_loss_aspect_ratio_penalty(self):
        """Test that CIoU penalizes aspect ratio differences."""
        # Same center and size, but different aspect ratios
        pred_boxes = torch.tensor([[0.2, 0.3, 0.8, 0.5]])  # Wide box
        gt_boxes = torch.tensor([[0.3, 0.2, 0.5, 0.8]])  # Tall box

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        # Loss should reflect aspect ratio mismatch
        assert loss > 0

    def test_ciou_loss_center_distance_penalty(self):
        """Test that CIoU penalizes center distance."""
        # Two boxes with same size but different centers
        pred_boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
        gt_boxes = torch.tensor([[0.3, 0.3, 0.5, 0.5]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        # Loss should reflect center distance
        assert loss > 0.5

    def test_ciou_loss_gradient_flow(self):
        """Test that gradients flow properly through CIoU loss."""
        pred_boxes = torch.tensor([[0.2, 0.3, 0.5, 0.6]], requires_grad=True)
        gt_boxes = torch.tensor([[0.25, 0.35, 0.55, 0.65]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")
        loss.sum().backward()

        assert pred_boxes.grad is not None
        assert pred_boxes.grad.shape == pred_boxes.shape
        # Gradients should be non-zero
        assert not torch.all(pred_boxes.grad == 0)

    def test_ciou_loss_numerical_stability(self):
        """Test CIoU loss numerical stability with edge cases."""
        # Very small boxes
        pred_boxes = torch.tensor([[0.49, 0.49, 0.51, 0.51]])
        gt_boxes = torch.tensor([[0.495, 0.495, 0.505, 0.505]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        assert torch.isfinite(loss).all()
        assert loss >= 0

    def test_ciou_loss_normalized_coordinates(self):
        """Test CIoU loss with normalized coordinates (0-1 range)."""
        pred_boxes = torch.tensor([[0.1, 0.2, 0.9, 0.8]])
        gt_boxes = torch.tensor([[0.15, 0.25, 0.85, 0.75]])

        loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="none")

        assert loss.shape == (1,)
        assert torch.isfinite(loss).all()
        # Should have small loss (good overlap)
        assert 0 < loss < 0.3

    def test_ciou_better_than_iou_for_non_overlapping(self):
        """Verify CIoU provides meaningful loss when IoU=0 (no overlap)."""
        # Non-overlapping boxes at different distances
        pred_boxes_close = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
        pred_boxes_far = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
        gt_boxes = torch.tensor([[0.5, 0.5, 0.7, 0.7]])

        loss_close = complete_box_iou_loss(pred_boxes_close, gt_boxes, reduction="none")
        loss_far = complete_box_iou_loss(pred_boxes_far, gt_boxes, reduction="none")

        # Both should have non-zero loss
        assert loss_close > 0
        assert loss_far > 0
        # Farther box should have higher loss
        assert loss_far > loss_close


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
