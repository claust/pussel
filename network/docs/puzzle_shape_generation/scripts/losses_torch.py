"""Differentiable loss functions for puzzle piece shape optimization."""

import torch
import torch.nn.functional as F


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute bidirectional Chamfer distance between two point sets.

    The Chamfer distance is the sum of the average nearest-neighbor distances
    in both directions. It's fully differentiable and works well for contour matching.

    Args:
        pred: (N, 2) predicted contour points.
        target: (M, 2) target contour points.
        eps: Small value for numerical stability.

    Returns:
        Scalar Chamfer distance (lower is better).
    """
    # Compute pairwise distance matrix: (N, M)
    dist_matrix = torch.cdist(pred, target, p=2)

    # Forward direction: for each predicted point, find nearest target point
    min_dists_pred_to_target = dist_matrix.min(dim=1).values  # (N,)
    forward_loss = min_dists_pred_to_target.mean()

    # Backward direction: for each target point, find nearest predicted point
    min_dists_target_to_pred = dist_matrix.min(dim=0).values  # (M,)
    backward_loss = min_dists_target_to_pred.mean()

    return forward_loss + backward_loss


def soft_hausdorff(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 20.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute smooth approximation of Hausdorff distance using soft-max.

    The Hausdorff distance is the maximum of the minimum distances, which has
    discontinuous gradients. This soft version uses a weighted soft-max
    approximation for smoother optimization.

    Args:
        pred: (N, 2) predicted contour points.
        target: (M, 2) target contour points.
        beta: Temperature for soft-max. Higher values approximate hard max more closely.
              Typical values: 10-50. Default 20 provides good balance.
        eps: Small value for numerical stability.

    Returns:
        Scalar soft Hausdorff distance (lower is better).
    """
    # Compute pairwise distance matrix: (N, M)
    dist_matrix = torch.cdist(pred, target, p=2)

    # Forward: soft-max over pred of (min distance to target)
    min_dists_pred = dist_matrix.min(dim=1).values  # (N,)
    # Subtract max for numerical stability in softmax
    min_dists_pred_stable = min_dists_pred - min_dists_pred.max().detach()
    weights_pred = F.softmax(beta * min_dists_pred_stable, dim=0)
    soft_max_pred = (weights_pred * min_dists_pred).sum()

    # Backward: soft-max over target of (min distance to pred)
    min_dists_target = dist_matrix.min(dim=0).values  # (M,)
    min_dists_target_stable = min_dists_target - min_dists_target.max().detach()
    weights_target = F.softmax(beta * min_dists_target_stable, dim=0)
    soft_max_target = (weights_target * min_dists_target).sum()

    # Return the larger of the two (standard Hausdorff is max of both directions)
    # Using soft-max here too for full differentiability
    stacked = torch.stack([soft_max_pred, soft_max_target])
    stacked_stable = stacked - stacked.max().detach()
    weights = F.softmax(beta * stacked_stable, dim=0)
    return (weights * stacked).sum()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    chamfer_weight: float = 1.0,
    hausdorff_weight: float = 0.3,
    hausdorff_beta: float = 20.0,
) -> torch.Tensor:
    """Compute combined loss from Chamfer and soft-Hausdorff distances.

    This combination provides both good average matching (Chamfer) and
    attention to worst-case deviations (Hausdorff).

    Args:
        pred: (N, 2) predicted contour points.
        target: (M, 2) target contour points.
        chamfer_weight: Weight for Chamfer distance term.
        hausdorff_weight: Weight for soft-Hausdorff distance term.
        hausdorff_beta: Temperature for soft-max in Hausdorff computation.

    Returns:
        Weighted sum of losses.
    """
    chamfer = chamfer_distance(pred, target)
    hausdorff = soft_hausdorff(pred, target, beta=hausdorff_beta)
    return chamfer_weight * chamfer + hausdorff_weight * hausdorff
