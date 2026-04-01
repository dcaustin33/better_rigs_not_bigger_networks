"""Regularization losses for Gaussian parameters."""

import torch
import torch.nn as nn

from gaussian_avatar.losses.utils import get_device_from_tensors


class PositionRegularizationLoss(nn.Module):
    """Penalize Gaussians that drift far from triangle centers.

    Uses a hinge-style loss that only penalizes positions beyond a threshold,
    with quadratic growth for excessive drift.

    Args:
        epsilon: Threshold distance below which no penalty applies (meters)
    """

    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, local_positions: torch.Tensor) -> torch.Tensor:
        """Compute position regularization loss.

        Args:
            local_positions: (N, 3) or (B, N, 3) local position offsets

        Returns:
            Scalar loss value
        """
        distances = torch.norm(local_positions, dim=-1)
        excess = torch.clamp(distances - self.epsilon, min=0.0)
        return (excess**2).mean()


class ScaleRegularizationLoss(nn.Module):
    """Penalize Gaussians that grow too large.

    Uses a hinge-style loss that only penalizes linear scales beyond a threshold,
    with quadratic growth for oversized Gaussians.

    Args:
        epsilon: Maximum allowed linear scale before penalty
    """

    def __init__(self, epsilon: float = 0.15):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, log_scales: torch.Tensor) -> torch.Tensor:
        """Compute scale regularization loss.

        Args:
            log_scales: (N, 3) or (B, N, 3) log-scale values

        Returns:
            Scalar loss value
        """
        scales = torch.exp(log_scales)
        excess = torch.clamp(scales - self.epsilon, min=0.0)
        return (excess**2).mean()


class ScaleIsotropyLoss(nn.Module):
    """Penalize anisotropy in the tangent-plane (x, y) scales and encourage
    tangent-vs-normal anisotropy.

    Two components:
    1. Isotropy: penalizes (log_sx - log_sy)^2 so tangent-plane scales match.
    2. Flatness: penalizes exp(log_sz - log_sx) + exp(log_sz - log_sy)
       to encourage tangent scales larger than normal (flat Gaussians).
       Asymmetric: strongly penalizes z > x,y, gently rewards x,y > z.

    Args:
        lambda_flatness: Weight for the tangent-vs-normal flatness term
            relative to the isotropy term.
    """

    def __init__(self, lambda_flatness: float = 0.5):
        super().__init__()
        self.lambda_flatness = lambda_flatness

    def forward(self, log_scales: torch.Tensor) -> torch.Tensor:
        """Compute scale isotropy loss.

        Args:
            log_scales: (N, 3) or (B, N, 3) log-scale values where
                the last dim is (x, y, z) in the local triangle frame

        Returns:
            Scalar loss value
        """
        # Encourage x ≈ y (round in tangent plane)
        diff_xy = log_scales[..., 0] - log_scales[..., 1]
        isotropy = (diff_xy**2).mean()

        # Encourage x > z and y > z (flat against surface)
        flatness = (
            torch.exp(log_scales[..., 2] - log_scales[..., 0])
            + torch.exp(log_scales[..., 2] - log_scales[..., 1])
        ).mean()

        return isotropy + self.lambda_flatness * flatness


class OpacityBinaryLoss(nn.Module):
    """Push Gaussian opacities toward 0 or 1 using binary entropy.

    Computes -o*log(o) - (1-o)*log(1-o) which is minimized at o=0 and o=1,
    and maximized at o=0.5. This discourages semi-transparent Gaussians that
    waste rendering budget.

    Args:
        eps: Clamping epsilon for numerical stability.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, raw_opacities: torch.Tensor) -> torch.Tensor:
        """Compute binary entropy loss on opacities.

        Args:
            raw_opacities: (N, 1) or (N,) pre-sigmoid opacity values

        Returns:
            Scalar loss value
        """
        o = torch.sigmoid(raw_opacities.squeeze())
        o = o.clamp(self.eps, 1.0 - self.eps)
        return (-o * o.log() - (1.0 - o) * (1.0 - o).log()).mean()


class OffsetRegularizationLoss(nn.Module):
    """Penalize large MLP offset predictions.

    Encourages the deformation MLP to output small corrections,
    letting mesh-based deformation dominate.

    Args:
        lambda_position: Weight for position offset penalty
        lambda_rotation: Weight for rotation offset penalty
        lambda_scale: Weight for scale offset penalty
        lambda_color: Weight for color offset penalty
    """

    def __init__(
        self,
        lambda_position: float = 1.0,
        lambda_rotation: float = 1.0,
        lambda_scale: float = 1.0,
        lambda_color: float = 1.0,
    ):
        super().__init__()
        self.lambda_position = lambda_position
        self.lambda_rotation = lambda_rotation
        self.lambda_scale = lambda_scale
        self.lambda_color = lambda_color
        
    def forward(
        self,
        delta_position: torch.Tensor | None = None,
        delta_rotation: torch.Tensor | None = None,
        delta_scale: torch.Tensor | None = None,
        delta_color: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute offset regularization loss.

        Args:
            delta_position: (N, 3) position offsets
            delta_rotation: (N, 3) quaternion xyz offsets (w=1 implicit)
            delta_scale: (N, 3) log-scale offsets
            delta_color: (N, 3) color offsets
            device: Device for zero tensors when all deltas are None

        Returns:
            Dictionary with 'base' (position+rotation+scale) and 'color' losses
        """
        if device is None:
            device = get_device_from_tensors(delta_position, delta_rotation, delta_scale, delta_color)
        base_loss = torch.tensor(0.0, device=device)
        color_loss = torch.tensor(0.0, device=device)

        if delta_position is not None:
            base_loss = base_loss + self.lambda_position * (delta_position**2).mean()

        if delta_rotation is not None:
            base_loss = base_loss + self.lambda_rotation * (delta_rotation**2).mean()

        if delta_scale is not None:
            base_loss = base_loss + self.lambda_scale * (delta_scale**2).mean()

        if delta_color is not None:
            color_loss = color_loss + self.lambda_color * (delta_color**2).mean()

        return {"base": base_loss, "color": color_loss}


class RegularizationLoss(nn.Module):
    """Combined regularization losses.

    Aggregates position, scale, and offset regularization with configurable weights.

    Args:
        lambda_position: Weight for position regularization
        lambda_scale: Weight for scale regularization
        lambda_offset: Weight for offset regularization
        lambda_offset_color: Weight for color offset regularization
        lambda_scale_isotropy: Weight for tangent-plane scale isotropy
        epsilon_position: Threshold for position penalty (meters)
        epsilon_scale: Threshold for scale penalty
    """

    def __init__(
        self,
        lambda_position: float = 0.01,
        lambda_scale: float = 0.01,
        lambda_offset: float = 0.001,
        lambda_offset_color: float = 0.001,
        lambda_scale_isotropy: float = 0.0,
        lambda_opacity_binary: float = 0.0,
        epsilon_position: float = 0.05,
        epsilon_scale: float = 0.15,
    ):
        super().__init__()
        self.lambda_position = lambda_position
        self.lambda_scale = lambda_scale
        self.lambda_offset = lambda_offset
        self.lambda_offset_color = lambda_offset_color
        self.lambda_scale_isotropy = lambda_scale_isotropy
        self.lambda_opacity_binary = lambda_opacity_binary

        self.position_loss = PositionRegularizationLoss(epsilon_position)
        self.scale_loss = ScaleRegularizationLoss(epsilon_scale)
        self.scale_isotropy_loss = ScaleIsotropyLoss()
        self.opacity_binary_loss = OpacityBinaryLoss()
        self.offset_loss = OffsetRegularizationLoss(lambda_color=1.0)

    def forward(
        self,
        local_positions: torch.Tensor,
        log_scales: torch.Tensor,
        delta_position: torch.Tensor | None = None,
        delta_rotation: torch.Tensor | None = None,
        delta_scale: torch.Tensor | None = None,
        delta_color: torch.Tensor | None = None,
        raw_opacities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all regularization losses.

        Args:
            local_positions: (N, 3) or (B, N, 3) local position offsets
            log_scales: (N, 3) or (B, N, 3) log-scale values
            delta_position: Optional (N, 3) position offsets from MLP
            delta_rotation: Optional (N, 3) quaternion xyz offsets from MLP
            delta_scale: Optional (N, 3) log-scale offsets from MLP
            delta_color: Optional (N, 3) color offsets from MLP
            raw_opacities: Optional (N, 1) or (N,) pre-sigmoid opacity values

        Returns:
            Dictionary with individual losses and total
        """
        losses = {}

        losses["position_reg"] = self.position_loss(local_positions)
        losses["scale_reg"] = self.scale_loss(log_scales)
        losses["scale_isotropy"] = self.scale_isotropy_loss(log_scales)
        offset_losses = self.offset_loss(
            delta_position, delta_rotation, delta_scale, delta_color,
            device=local_positions.device,
        )
        losses["offset_reg_base"] = offset_losses["base"]
        losses["offset_reg_color"] = offset_losses["color"]
        losses["offset_reg"] = offset_losses["base"] + offset_losses["color"]

        if raw_opacities is not None and self.lambda_opacity_binary > 0:
            losses["opacity_binary"] = self.opacity_binary_loss(raw_opacities)
        else:
            losses["opacity_binary"] = torch.tensor(0.0, device=local_positions.device)

        losses["regularization"] = (
            self.lambda_position * losses["position_reg"]
            + self.lambda_scale * losses["scale_reg"]
            + self.lambda_scale_isotropy * losses["scale_isotropy"]
            + self.lambda_offset * losses["offset_reg_base"]
            + self.lambda_offset_color * losses["offset_reg_color"]
            + self.lambda_opacity_binary * losses["opacity_binary"]
        )

        return losses
