"""Mask/silhouette loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLoss(nn.Module):
    """Binary cross-entropy mask/silhouette loss.

    Supervises foreground/background separation using ground truth masks.

    Args:
        reduction: How to reduce the loss ("mean", "sum")
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        rendered_alpha: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mask loss.

        Args:
            rendered_alpha: Rendered alpha/opacity (B, H, W) or (H, W), range [0, 1]
            target_mask: Ground truth mask, same shape, range [0, 1]

        Returns:
            Scalar loss value
        """
        # Clamp to avoid log(0) in BCE computation
        rendered_alpha = rendered_alpha.clamp(1e-6, 1 - 1e-6)

        return F.binary_cross_entropy(
            rendered_alpha,
            target_mask,
            reduction=self.reduction,
        )
