"""Reconstruction losses for image quality supervision."""

import torch
import torch.nn as nn

from gaussian_avatar.losses.utils import ensure_batch_dim


class L1Loss(nn.Module):
    """L1 pixel-wise reconstruction loss.

    Computes the mean absolute difference between rendered and target images,
    optionally weighted by a foreground mask.

    Args:
        reduction: How to reduce the loss ("mean", "sum", "none")
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            rendered: Rendered image (B, 3, H, W) or (3, H, W), values in [0, 1]
            target: Target image, same shape as rendered
            mask: Optional foreground mask (B, 1, H, W) or (1, H, W)

        Returns:
            Scalar loss value (or per-pixel if reduction="none")
        """
        diff = torch.abs(rendered - target)

        if mask is not None:
            if mask.dim() == rendered.dim() - 1:
                mask = mask.unsqueeze(-3)
            diff = diff * mask

            if self.reduction == "mean":
                return diff.sum() / (mask.sum() * rendered.shape[-3] + 1e-8)

        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        return diff


class L2Loss(nn.Module):
    """L2 pixel-wise reconstruction loss.

    Computes the mean squared difference between rendered and target images,
    optionally weighted by a foreground mask.

    Args:
        reduction: How to reduce the loss ("mean", "sum", "none")
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute L2 loss.

        Args:
            rendered: Rendered image (B, 3, H, W) or (3, H, W), values in [0, 1]
            target: Target image, same shape as rendered
            mask: Optional foreground mask (B, 1, H, W) or (1, H, W)

        Returns:
            Scalar loss value (or per-pixel if reduction="none")
        """
        diff = (rendered - target) ** 2

        if mask is not None:
            if mask.dim() == rendered.dim() - 1:
                mask = mask.unsqueeze(-3)
            diff = diff * mask

            if self.reduction == "mean":
                return diff.sum() / (mask.sum() * rendered.shape[-3] + 1e-8)

        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        return diff


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss.

    Uses fused-ssim for GPU acceleration when available,
    falls back to pytorch-msssim otherwise.

    Args:
        window_size: Size of the Gaussian window (default: 11)
        use_fused: Whether to use fused-ssim if available
    """

    def __init__(self, window_size: int = 11, use_fused: bool = True):
        super().__init__()
        self.window_size = window_size
        self.use_fused = use_fused
        self._fused_available: bool | None = None

    def _check_fused_available(self) -> bool:
        """Check if fused-ssim is available (cached)."""
        if self._fused_available is None:
            try:
                from fused_ssim import fused_ssim  # noqa: F401

                self._fused_available = True
            except ImportError:
                self._fused_available = False
        return self._fused_available

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            rendered: Rendered image (B, 3, H, W) or (3, H, W), values in [0, 1]
            target: Target image, same shape as rendered
            mask: Optional foreground mask (currently ignored by SSIM)

        Returns:
            Scalar loss value (1 - SSIM)
        """
        rendered, _ = ensure_batch_dim(rendered)
        target, _ = ensure_batch_dim(target)

        if self.use_fused and self._check_fused_available() and rendered.is_cuda:
            from fused_ssim import fused_ssim

            ssim_val = fused_ssim(rendered, target)
        else:
            from pytorch_msssim import ssim

            ssim_val = ssim(
                rendered,
                target,
                data_range=1.0,
                size_average=True,
                win_size=self.window_size,
            )

        return 1.0 - ssim_val


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity loss.

    Uses pre-trained AlexNet features to measure perceptual similarity.
    Network weights are frozen during training.

    Args:
        net: Network backbone ("alex", "vgg", "squeeze")
        normalize: Whether inputs are in [0, 1] (True) or [-1, 1] (False)
    """

    def __init__(self, net: str = "alex", normalize: bool = True):
        super().__init__()
        import lpips

        self.lpips = lpips.LPIPS(net=net)
        self.normalize = normalize

        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute LPIPS loss.

        Args:
            rendered: Rendered image (B, 3, H, W) or (3, H, W), values in [0, 1]
            target: Target image, same shape as rendered
            mask: Optional mask (currently ignored)

        Returns:
            Scalar loss value
        """
        rendered, _ = ensure_batch_dim(rendered)
        target, _ = ensure_batch_dim(target)

        if next(self.lpips.parameters()).device != rendered.device:
            self.lpips = self.lpips.to(rendered.device)

        if self.normalize:
            rendered = rendered * 2.0 - 1.0
            target = target * 2.0 - 1.0

        return self.lpips(rendered, target, normalize=False).mean()


class ReconstructionLoss(nn.Module):
    """Combined reconstruction loss with L1, L2, SSIM, and LPIPS.

    Aggregates L1, L2, SSIM, and LPIPS losses with configurable weights.

    Args:
        lambda_l1: Weight for L1 loss
        lambda_l2: Weight for L2 loss
        lambda_ssim: Weight for SSIM loss
        lambda_lpips: Weight for LPIPS loss (set to 0 to skip LPIPS)
        lpips_net: LPIPS backbone network ("alex", "vgg", "squeeze")
        use_fused_ssim: Whether to use fused-ssim if available
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_l2: float = 0.0,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.1,
        lpips_net: str = "alex",
        use_fused_ssim: bool = True,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips

        self.l1_loss = L1Loss()
        self.l2_loss = L2Loss()
        self.ssim_loss = SSIMLoss(use_fused=use_fused_ssim)
        self.lpips_loss = LPIPSLoss(net=lpips_net) if lambda_lpips > 0 else None

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined reconstruction loss.

        Args:
            rendered: Rendered image (B, 3, H, W)
            target: Target image
            mask: Optional foreground mask

        Returns:
            Dictionary with individual losses and total
        """
        losses = {}

        losses["l1"] = self.l1_loss(rendered, target, mask)
        losses["l2"] = self.l2_loss(rendered, target, mask)
        losses["ssim"] = self.ssim_loss(rendered, target, mask)

        if self.lpips_loss is not None:
            losses["lpips"] = self.lpips_loss(rendered, target, mask)
        else:
            losses["lpips"] = torch.tensor(0.0, device=rendered.device)

        losses["reconstruction"] = (
            self.lambda_l1 * losses["l1"]
            + self.lambda_l2 * losses["l2"]
            + self.lambda_ssim * losses["ssim"]
            + self.lambda_lpips * losses["lpips"]
        )

        return losses
