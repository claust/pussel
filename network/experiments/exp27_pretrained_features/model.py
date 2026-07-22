"""Exp27 model: frozen pretrained ViT features under the correlation heads.

The stage-0 zero-shot probe showed that frozen DINOv2-S/14 features beat the
synthetic-trained CNN 3.3x on the real-photo benchmark, and that the signal
lives in *dense* correlation (49.2% both) rather than pooled descriptors
(18.0%). This model therefore keeps the validated exp20 head shapes but:

- replaces the trained-from-scratch dual backbones with ONE shared frozen
  DINOv2-S/14 encoder (exp19's dual-vs-Siamese finding concerned *trained*
  encoders; frozen ones are identical by construction),
- adds small trainable per-branch adapters (the only backbone-side capacity),
- defaults to a dense position head: the piece's adapted token grid is
  cross-correlated against the puzzle grid as a template (the probe's winning
  shape, and critical-review item #6's dense-heatmap direction),
- scores rotation by RE-ENCODING the 4 rotated piece images instead of
  rotating a feature map: ViT features are not rotation-equivariant, so
  ``torch.rot90`` on the token grid is not a substitute for encoding the
  rotated image. The exp20 comparison net is reused unchanged.

ImageNet normalization happens INSIDE the model (registered buffers), so
every caller — harness, evaluators, backend — keeps feeding [0, 1] tensors
exactly as for exp20/exp26 and cannot get the transform wrong.

Inputs: piece 224x224 (16x16 patch tokens), puzzle 448x448 (32x32 tokens,
exactly 8x8 per cell of the 4x4 grid).
"""

import contextlib
from typing import Literal

import timm  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..exp20_realistic_pieces.model import FastBackboneModel, RotationCorrelationModule, SpatialCorrelationModule

DEFAULT_ENCODER = "vit_small_patch14_dinov2.lvd142m"
PATCH_SIZE = 14

PositionHeadType = Literal["dense", "pooled"]


class FrozenViTEncoder(nn.Module):
    """Frozen pretrained ViT that maps [0, 1] RGB batches to patch-token grids.

    Normalization constants come from the timm data config and are applied
    internally. The ViT stays in eval mode permanently and runs under
    ``no_grad`` while fully frozen (no activation graph, major memory win);
    unfreezing the last block flips it to a normal grad-tracking forward.
    """

    def __init__(self, model_name: str = DEFAULT_ENCODER) -> None:
        """Load the pretrained encoder and freeze it.

        Args:
            model_name: timm model name of a patch-14 ViT.
        """
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        self.fully_frozen = True

        cfg = timm.data.resolve_model_data_config(self.vit)
        self.register_buffer("pixel_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.num_prefix_tokens = int(getattr(self.vit, "num_prefix_tokens", 1))
        self.feature_dim = int(self.vit.num_features)

    def train(self, mode: bool = True) -> "FrozenViTEncoder":
        """Keep the ViT in eval mode regardless of the surrounding train/eval state.

        Args:
            mode: Requested mode for this module's own (non-ViT) state.

        Returns:
            self.
        """
        super().train(mode)
        self.vit.eval()
        return self

    def unfreeze_last_block(self) -> None:
        """Make the final transformer block and norm trainable (phase 2)."""
        self.fully_frozen = False
        for param in self.vit.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.vit.norm.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of [0, 1] RGB images into a spatial token grid.

        Args:
            x: Tensor of shape (B, 3, H, W) with H and W multiples of 14.

        Returns:
            Patch-token feature map of shape (B, C, H/14, W/14).
        """
        grid_h, grid_w = x.shape[2] // PATCH_SIZE, x.shape[3] // PATCH_SIZE
        context = torch.no_grad() if self.fully_frozen else contextlib.nullcontext()
        with context:
            x = (x - self.pixel_mean) / self.pixel_std
            feats = self.vit.forward_features(x)
            tokens = feats[:, self.num_prefix_tokens :, :]
        return tokens.transpose(1, 2).reshape(x.shape[0], self.feature_dim, grid_h, grid_w)


def make_adapter(in_dim: int, out_dim: int) -> nn.Module:
    """Small trainable adapter on top of the frozen features (one per branch).

    Args:
        in_dim: Frozen encoder feature dimension.
        out_dim: Adapter output dimension (the heads' feature_dim).

    Returns:
        The adapter module.
    """
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1),
        nn.GELU(),
        nn.Conv2d(out_dim, out_dim, 3, padding=1),
        nn.GELU(),
    )


class DensePositionModule(nn.Module):
    """Dense template-matching position head (the probe's winning shape).

    The piece's projected token grid is pooled to one cell's footprint and
    cross-correlated against the projected puzzle grid via grouped
    convolution; a softmax over window positions gives an attention map whose
    expectation is the predicted center. Window centers span exactly the
    valid cell-center range, so no padding is needed.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        correlation_dim: int = 128,
        dropout: float = 0.1,
        grid_size: int = 4,
    ) -> None:
        """Initialize the dense position head.

        Args:
            feature_dim: Input feature dimension (adapter output).
            correlation_dim: Projection dimension for the correlation.
            dropout: Dropout rate on the projections.
            grid_size: Puzzle grid size; the template covers one cell.
        """
        super().__init__()
        self.grid_size = grid_size
        self.piece_proj = nn.Sequential(
            nn.Conv2d(feature_dim, correlation_dim, 1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.puzzle_proj = nn.Sequential(
            nn.Conv2d(feature_dim, correlation_dim, 1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.temperature = nn.Parameter(torch.ones(1) * (correlation_dim**0.5))

    def forward(self, piece_feat_map: torch.Tensor, puzzle_feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Correlate the piece template against the puzzle and return position.

        Args:
            piece_feat_map: Piece features (B, C, h, w).
            puzzle_feat_map: Puzzle features (B, C, H, W).

        Returns:
            Tuple of (attention_map over window positions (B, 1, oh, ow),
            expected (x, y) position in [0, 1]).
        """
        batch_size, _, height, width = puzzle_feat_map.shape
        template = max(height // self.grid_size, 1)

        piece_proj = self.piece_proj(piece_feat_map)
        puzzle_proj = self.puzzle_proj(puzzle_feat_map)
        corr_dim = piece_proj.shape[1]

        # The grouped conv accumulates corr_dim * template^2 products (the
        # normalization comes after), which overflows fp16 under AMP
        # (observed: pos_loss=nan on CUDA). Compute the correlation and the
        # softmax in fp32, outside any autocast region.
        autocast_off = (
            torch.autocast(device_type="cuda", enabled=False)
            if puzzle_proj.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_off:
            kernel = F.adaptive_avg_pool2d(piece_proj.float(), (template, template))
            correlation = F.conv2d(
                puzzle_proj.float().reshape(1, batch_size * corr_dim, height, width),
                kernel.reshape(batch_size, corr_dim, template, template),
                groups=batch_size,
            ).transpose(0, 1)
            # Sum over channels, mean over the window footprint (scale-matches
            # the exp20 correlation: a channel dot product per location).
            correlation = correlation / (template * template * self.temperature.float())

        out_h, out_w = correlation.shape[2], correlation.shape[3]
        attention_flat = F.softmax(correlation.reshape(batch_size, -1), dim=-1)
        attention_map = attention_flat.view(batch_size, 1, out_h, out_w)

        device = puzzle_feat_map.device
        x_coords = (torch.arange(out_w, device=device, dtype=torch.float32) + template / 2) / width
        y_coords = (torch.arange(out_h, device=device, dtype=torch.float32) + template / 2) / height
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        attention = attention_map.squeeze(1)
        expected_x = (attention * xx).sum(dim=[1, 2])
        expected_y = (attention * yy).sum(dim=[1, 2])
        expected_position = torch.stack([expected_x, expected_y], dim=1)

        return attention_map, expected_position


def heatmap_ce_loss(
    position: torch.Tensor,
    attention_map: torch.Tensor,
    targets: torch.Tensor,
    mse_weight: float = 1.0,
) -> torch.Tensor:
    """Position loss for the dense head: window cross-entropy + position MSE.

    MSE through the expectation of a near-uniform softmax over 625 windows
    gives vanishingly small per-window gradients (measured: after 3 epochs the
    best 3x3 window held ~10% attention mass and cell accuracy crawled).
    Supervising the heatmap directly with cross-entropy against the true
    window (SiamFC-style) gives every window logit a strong signal; the MSE
    term is kept so the expectation/refinement path stays calibrated for
    sub-cell position output.

    Args:
        position: Model position output (B, 2).
        attention_map: Dense-head attention over windows (B, 1, oh, ow),
            a softmax (fp32).
        targets: True (cx, cy) in [0, 1] (B, 2).
        mse_weight: Weight of the auxiliary position MSE term.

    Returns:
        Scalar loss.
    """
    b, _, out_h, out_w = attention_map.shape
    # Recover the window/template geometry from the map shape: the puzzle
    # patch grid side H satisfies oh = H - ts + 1 with ts = H // grid (grid=4),
    # i.e. oh = 0.75*H + 1.
    grid_side = round((out_h - 1) * 4 / 3)
    template = grid_side // 4

    target_j = torch.clamp(torch.round(targets[:, 0] * grid_side - template / 2).long(), 0, out_w - 1)
    target_i = torch.clamp(torch.round(targets[:, 1] * grid_side - template / 2).long(), 0, out_h - 1)
    target_idx = target_i * out_w + target_j

    log_attention = torch.log(attention_map.reshape(b, -1).clamp_min(1e-9))
    ce = F.nll_loss(log_attention, target_idx)
    return ce + mse_weight * F.mse_loss(position, targets)


def _batchnorm_to_groupnorm(module: nn.Module) -> None:
    """Recursively replace BatchNorm2d layers with GroupNorm (in place).

    Args:
        module: Module tree to rewrite.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(min(8, child.num_features), child.num_features))
        else:
            _batchnorm_to_groupnorm(child)


class PrecomputedRotationCorrelation(RotationCorrelationModule):
    """exp20's rotation head fed with re-encoded rotated piece feature maps.

    Reuses the parent's projections, comparison net and region extraction;
    the source of the per-rotation piece maps differs (encodings of the
    rotated images, not rotations of one encoding), and the parent's
    BatchNorm layers are replaced with GroupNorm: with a frozen encoder the
    trainable adapters shift this head's input distribution every epoch, so
    BN running statistics are permanently stale (measured at epoch 9:
    48% rotation with running stats vs 97% with batch statistics).
    """

    def __init__(self, feature_dim: int = 576, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        """Initialize and swap BatchNorm for GroupNorm.

        Args:
            feature_dim: Input feature dimension.
            hidden_dim: Hidden dimension of the comparison network.
            dropout: Dropout rate.
        """
        super().__init__(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        _batchnorm_to_groupnorm(self)

    def score_rotations(
        self,
        piece_feat_maps: list[torch.Tensor],
        puzzle_feat_map: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Score the 4 rotation classes from precomputed piece feature maps.

        Args:
            piece_feat_maps: One feature map per rotation class k, from the
                piece image rotated clockwise by k*90 (matching the parent's
                ``_rotate_feature_map`` direction).
            puzzle_feat_map: Puzzle feature map (B, C, H, W).
            position: Predicted (x, y) positions (B, 2) in [0, 1].

        Returns:
            Rotation logits (B, 4).
        """
        _, _, h_piece, w_piece = piece_feat_maps[0].shape
        puzzle_region = self._extract_region(puzzle_feat_map, position, (h_piece, w_piece))
        puzzle_proj = self.puzzle_proj(puzzle_region)

        rotation_scores = []
        for rot_idx in range(4):
            piece_proj = self.piece_proj(piece_feat_maps[rot_idx])
            combined = torch.cat([piece_proj, puzzle_proj], dim=1)
            rotation_scores.append(self.comparison_net(combined))
        return torch.cat(rotation_scores, dim=1)


class FrozenFeatureModel(nn.Module):
    """Frozen DINOv2 encoder + trainable adapters + correlation heads.

    Drop-in compatible with the exp20 harness: ``forward(piece, puzzle)``
    returns ``(position, rotation_logits, attention_map)`` and
    ``positions_to_cells`` / ``get_parameter_groups`` match the
    ``FastBackboneModel`` interface.
    """

    positions_to_cells = staticmethod(FastBackboneModel.positions_to_cells)

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        adapter_dim: int = 256,
        correlation_dim: int = 128,
        rotation_hidden_dim: int = 128,
        dropout: float = 0.1,
        rotation_dropout: float = 0.2,
        position_head: PositionHeadType = "dense",
        grid_size: int = 4,
        unfreeze_last_block: bool = False,
    ) -> None:
        """Initialize the model.

        Args:
            encoder_name: timm name of the frozen patch-14 ViT.
            adapter_dim: Adapter output dimension (heads' feature_dim).
            correlation_dim: Position-correlation projection dimension.
            rotation_hidden_dim: Rotation comparison hidden dimension.
            dropout: Position head dropout.
            rotation_dropout: Rotation head dropout.
            position_head: "dense" (probe-validated default) or "pooled"
                (exp20's SpatialCorrelationModule, as an ablation).
            grid_size: Puzzle grid size (4 for the realistic 4x4 benchmark).
            unfreeze_last_block: Make the encoder's last block trainable
                (phase 2; keeps the rest frozen).
        """
        super().__init__()
        self.position_head_type: PositionHeadType = position_head
        self.encoder = FrozenViTEncoder(encoder_name)
        if unfreeze_last_block:
            self.encoder.unfreeze_last_block()

        self.piece_adapter = make_adapter(self.encoder.feature_dim, adapter_dim)
        self.puzzle_adapter = make_adapter(self.encoder.feature_dim, adapter_dim)

        self.dense_position: DensePositionModule | None = None
        self.pooled_position: SpatialCorrelationModule | None = None
        if position_head == "dense":
            self.dense_position = DensePositionModule(
                feature_dim=adapter_dim, correlation_dim=correlation_dim, dropout=dropout, grid_size=grid_size
            )
        else:
            self.pooled_position = SpatialCorrelationModule(
                feature_dim=adapter_dim, correlation_dim=correlation_dim, dropout=dropout
            )

        self.refinement = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self.rotation_correlation = PrecomputedRotationCorrelation(
            feature_dim=adapter_dim, hidden_dim=rotation_hidden_dim, dropout=rotation_dropout
        )

    def forward(self, piece: torch.Tensor, puzzle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            piece: Piece batch (B, 3, 224, 224) in [0, 1].
            puzzle: Puzzle batch (B, 3, 448, 448) in [0, 1].

        Returns:
            Tuple of (position (B, 2) in [0, 1], rotation logits (B, 4),
            attention map over the puzzle).
        """
        batch_size = piece.shape[0]

        # Encode the piece at all 4 clockwise rotations in one batch. Class k
        # corresponds to the image rotated clockwise by k*90, mirroring
        # exp20's feature-map rotation direction (rot90 with k=-1 per step).
        rotated = torch.cat([torch.rot90(piece, k=-r, dims=[2, 3]) for r in range(4)], dim=0)
        piece_maps = self.piece_adapter(self.encoder(rotated))
        piece_maps_by_rot = [piece_maps[r * batch_size : (r + 1) * batch_size] for r in range(4)]

        puzzle_map = self.puzzle_adapter(self.encoder(puzzle))

        if self.dense_position is not None:
            attention_map, expected_pos = self.dense_position(piece_maps_by_rot[0], puzzle_map)
        else:
            assert self.pooled_position is not None
            piece_vec = F.adaptive_avg_pool2d(piece_maps_by_rot[0], 1).flatten(1)
            attention_map, expected_pos = self.pooled_position(piece_vec, puzzle_map)

        refinement = self.refinement(expected_pos)
        position = torch.clamp(expected_pos + 0.1 * refinement, 0, 1)

        rotation_logits = self.rotation_correlation.score_rotations(piece_maps_by_rot, puzzle_map, position)
        return position, rotation_logits, attention_map

    def predict_cell(self, piece: torch.Tensor, puzzle: torch.Tensor, grid_size: int = 4) -> torch.Tensor:
        """Predict grid cell indices for a batch.

        Args:
            piece: Piece batch.
            puzzle: Puzzle batch.
            grid_size: Grid size.

        Returns:
            Cell indices (0 to grid_size*grid_size - 1).
        """
        position, _, _ = self.forward(piece, puzzle)
        return self.positions_to_cells(position, grid_size)

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """Parameter groups: trainable encoder params (if any) vs adapters/heads.

        Args:
            backbone_lr: Learning rate for unfrozen encoder parameters.
            head_lr: Learning rate for adapters and heads.
            weight_decay: Weight decay for both groups.

        Returns:
            AdamW-style parameter group list (empty groups omitted).
        """
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_modules: list[nn.Module] = [self.piece_adapter, self.puzzle_adapter, self.refinement]
        if self.dense_position is not None:
            head_modules.append(self.dense_position)
        if self.pooled_position is not None:
            head_modules.append(self.pooled_position)
        head_modules.append(self.rotation_correlation)
        head_params = [p for m in head_modules for p in m.parameters() if p.requires_grad]

        groups: list[dict] = []
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"}
            )
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "heads"})
        return groups


if __name__ == "__main__":
    model = FrozenFeatureModel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,} | trainable: {trainable:,}")
    piece = torch.rand(2, 3, 224, 224)
    puzzle = torch.rand(2, 3, 448, 448)
    position, rotation_logits, attention = model(piece, puzzle)
    print(
        f"position {tuple(position.shape)}, rotation {tuple(rotation_logits.shape)}, attention {tuple(attention.shape)}"
    )
    print(f"cells: {model.predict_cell(piece, puzzle).tolist()}")
