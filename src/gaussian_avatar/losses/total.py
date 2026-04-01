"""Total loss aggregation for Gaussian Avatar training."""

import torch
import torch.nn as nn

from gaussian_avatar.losses.aiap import AIAPLoss
from gaussian_avatar.losses.config import LossThresholds, LossWeights
from gaussian_avatar.losses.mask import MaskLoss
from gaussian_avatar.losses.reconstruction import ReconstructionLoss
from gaussian_avatar.losses.regularization import RegularizationLoss
from gaussian_avatar.losses.utils import (
    compute_boundary_weight_map,
    composite_premultiplied,
    composite_with_random_background,
)


class TotalLoss(nn.Module):
    """Aggregates all loss functions for training.

    Combines reconstruction losses (L1, SSIM, LPIPS), mask loss,
    regularization losses, and AIAP isometry losses into a single
    training objective.

    Args:
        weights: Loss weight configuration
        thresholds: Regularization threshold configuration
        use_fused_ssim: Whether to use fused-ssim if available
        use_cuda_knn: Whether to use CUDA KNN if available
        background_color: Optional static RGB background (each in [0, 1]).
            When provided, this fixed color is used for compositing instead
            of a random color each step.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        thresholds: LossThresholds | None = None,
        use_fused_ssim: bool = True,
        use_cuda_knn: bool = True,
        background_color: tuple[float, float, float] | None = None,
    ):
        super().__init__()

        self.weights = weights or LossWeights()
        self.thresholds = thresholds or LossThresholds()
        self.background_color = background_color

        self.reconstruction_loss = ReconstructionLoss(
            lambda_l1=self.weights.lambda_l1,
            lambda_l2=self.weights.lambda_l2,
            lambda_ssim=self.weights.lambda_ssim,
            lambda_lpips=self.weights.lambda_lpips,
            lpips_net=self.weights.lpips_net,
        )
        self.mask_loss = MaskLoss()

        self.regularization_loss = RegularizationLoss(
            lambda_position=self.weights.lambda_position_reg,
            lambda_scale=self.weights.lambda_scale_reg,
            lambda_offset=self.weights.lambda_offset_reg,
            lambda_offset_color=self.weights.lambda_offset_color_reg,
            lambda_scale_isotropy=self.weights.lambda_scale_isotropy,
            lambda_opacity_binary=self.weights.lambda_opacity_binary,
            epsilon_position=self.thresholds.epsilon_position,
            epsilon_scale=self.thresholds.epsilon_scale,
        )

        self.aiap_loss = AIAPLoss(
            k_neighbors=self.thresholds.k_neighbors,
            lambda_position=self.weights.lambda_aiap_position,
            lambda_covariance=self.weights.lambda_aiap_covariance,
            use_cuda_knn=use_cuda_knn,
        )

    def initialize_aiap(self, canonical_positions: torch.Tensor):
        """Initialize AIAP neighbor cache.

        Must be called once before training if using AIAP loss.

        Args:
            canonical_positions: (N, 3) Gaussian positions in canonical pose
        """
        self.aiap_loss.initialize(canonical_positions)

    def forward(
        self,
        rendered_rgb: torch.Tensor,
        rendered_alpha: torch.Tensor,
        target_rgb: torch.Tensor,
        target_mask: torch.Tensor,
        local_positions: torch.Tensor,
        log_scales: torch.Tensor,
        delta_position: torch.Tensor | None = None,
        delta_rotation: torch.Tensor | None = None,
        delta_scale: torch.Tensor | None = None,
        delta_color: torch.Tensor | None = None,
        canonical_positions: torch.Tensor | None = None,
        canonical_rotations: torch.Tensor | None = None,
        canonical_scales: torch.Tensor | None = None,
        posed_positions: torch.Tensor | None = None,
        posed_rotations: torch.Tensor | None = None,
        posed_scales: torch.Tensor | None = None,
        raw_opacities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            rendered_rgb: Rendered RGB image (B, 3, H, W)
            rendered_alpha: Rendered alpha (B, 1, H, W) or (B, H, W)
            target_rgb: Target RGB image (B, 3, H, W)
            target_mask: Target mask (B, 1, H, W) or (B, H, W)
            local_positions: (N, 3) or (B, N, 3) local position offsets
            log_scales: (N, 3) or (B, N, 3) log-scale values
            delta_position: Optional (N, 3) MLP position offsets
            delta_rotation: Optional (N, 3) MLP rotation offsets
            delta_scale: Optional (N, 3) MLP scale offsets
            delta_color: Optional (N, 3) MLP color offsets
            canonical_positions: Optional (N, 3) canonical Gaussian positions
            canonical_rotations: Optional (N, 4) canonical quaternions
            canonical_scales: Optional (N, 3) canonical scales
            posed_positions: Optional (N, 3) posed Gaussian positions
            posed_rotations: Optional (N, 4) posed quaternions
            posed_scales: Optional (N, 3) posed scales
            raw_opacities: Optional (N, 1) or (N,) pre-sigmoid opacity values
                for binary opacity regularization

        Returns:
            Dictionary with all individual losses and total
        """
        losses = {}

        # Use static background color if configured, otherwise random
        if self.background_color is not None:
            background = torch.tensor(
                self.background_color, device=rendered_rgb.device, dtype=rendered_rgb.dtype
            )
        else:
            background = torch.rand(3, device=rendered_rgb.device, dtype=rendered_rgb.dtype)

        # Composite over same background.
        # rendered_rgb is premultiplied alpha (renderer uses black bg),
        # so use composite_premultiplied to avoid double-applying alpha.
        # target_rgb is straight (non-premultiplied), so use the regular
        # composite_with_random_background.
        rendered_composite = composite_premultiplied(
            rendered_rgb, rendered_alpha, background
        )
        target_composite, _ = composite_with_random_background(
            target_rgb, target_mask, background
        )

        # Reconstruction loss on composited images
        if self.weights.mask_reconstruction_loss:
            recon_mask = compute_boundary_weight_map(
                target_mask,
                boundary_weight=self.weights.boundary_weight,
                kernel_size=self.weights.boundary_kernel_size,
            )
        else:
            recon_mask = None
        recon_losses = self.reconstruction_loss(
            rendered_composite, target_composite, recon_mask
        )
        losses.update(recon_losses)

        # Mask loss (normalize to 3D by removing channel dim)
        if rendered_alpha.dim() == 4:
            rendered_alpha_2d = rendered_alpha[:, 0]
        else:
            rendered_alpha_2d = rendered_alpha
        if target_mask.dim() == 4:
            target_mask_2d = target_mask[:, 0]
        else:
            target_mask_2d = target_mask

        losses["mask"] = self.mask_loss(rendered_alpha_2d, target_mask_2d)

        reg_losses = self.regularization_loss(
            local_positions,
            log_scales,
            delta_position,
            delta_rotation,
            delta_scale,
            delta_color,
            raw_opacities=raw_opacities,
        )
        losses.update(reg_losses)

        if canonical_positions is not None and posed_positions is not None:
            # Initialize AIAP if not done or Gaussian count changed (after densification)
            if not self.aiap_loss._initialized or canonical_positions.shape[0] != self.aiap_loss._initialized_count:
                self.aiap_loss.initialize(canonical_positions)

            # Compute position AIAP (always available)
            losses["aiap_position"] = self.aiap_loss.position_loss(
                canonical_positions, posed_positions
            )

            # Compute covariance AIAP only if rotations and scales are provided
            if (
                canonical_rotations is not None
                and canonical_scales is not None
                and posed_rotations is not None
                and posed_scales is not None
            ):
                losses["aiap_covariance"] = self.aiap_loss.covariance_loss(
                    canonical_rotations,
                    canonical_scales,
                    posed_rotations,
                    posed_scales,
                )
            else:
                losses["aiap_covariance"] = torch.tensor(0.0, device=rendered_rgb.device)

            losses["aiap"] = (
                self.aiap_loss.lambda_position * losses["aiap_position"]
                + self.aiap_loss.lambda_covariance * losses["aiap_covariance"]
            )
        else:
            losses["aiap_position"] = torch.tensor(0.0, device=rendered_rgb.device)
            losses["aiap_covariance"] = torch.tensor(0.0, device=rendered_rgb.device)
            losses["aiap"] = torch.tensor(0.0, device=rendered_rgb.device)

        losses["total"] = (
            losses["reconstruction"]
            + self.weights.lambda_mask * losses["mask"]
            + losses["regularization"]
            + losses["aiap"]
        )

        return losses
