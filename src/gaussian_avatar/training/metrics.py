"""Evaluation metrics for model quality assessment.

Provides PSNR, SSIM, and LPIPS metrics for comparing rendered images
against ground truth during evaluation.
"""

import logging
import torch
from torch import Tensor
import lpips
from pytorch_msssim import ssim as pytorch_ssim


MAX_PSNR = 100.0  # Cap to avoid inf in aggregate metrics


def compute_psnr(pred: Tensor, target: Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    For batched inputs, computes per-image PSNR and returns the mean.
    Zero-MSE images are clamped to MAX_PSNR (100 dB) to prevent inf
    from propagating into aggregate metrics.

    Args:
        pred: Predicted image (B, 3, H, W) or (3, H, W), values in [0, 1]
        target: Target image, same shape as pred

    Returns:
        PSNR value in dB (higher is better), clamped to MAX_PSNR
    """
    if pred.dim() == 3:
        mse = ((pred - target) ** 2).mean()
        if mse == 0:
            return MAX_PSNR
        return min((10 * torch.log10(1.0 / mse)).item(), MAX_PSNR)

    # Batched: compute per-image MSE, then per-image PSNR, then average
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])  # (B,)
    psnr = torch.where(
        mse == 0,
        torch.tensor(MAX_PSNR),
        10 * torch.log10(1.0 / mse),
    )
    return min(psnr.mean().item(), MAX_PSNR)


def compute_ssim(pred: Tensor, target: Tensor) -> float:
    """Compute Structural Similarity Index.

    Args:
        pred: Predicted image (B, 3, H, W) or (3, H, W), values in [0, 1]
        target: Target image, same shape as pred

    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    ssim_val = pytorch_ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True,
    )
    return ssim_val.item()


def compute_ssim_map(pred: Tensor, target: Tensor) -> Tensor:
    """Compute per-pixel SSIM error map.

    Returns 1 - SSIM at each pixel so that high values indicate regions
    where structural similarity is *lost* (analogous to LPIPS distance).

    Args:
        pred: Predicted image (3, H, W) or (B, 3, H, W), values in [0, 1]
        target: Target image, same shape as pred

    Returns:
        SSIM error map (H, W) in [0, 1] where 1 = maximum dissimilarity
    """
    from pytorch_msssim.ssim import _fspecial_gauss_1d
    from pytorch_msssim.ssim import gaussian_filter as gf

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    win_size = 11
    win_sigma = 1.5
    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([pred.shape[1]] + [1] * (len(pred.shape) - 1))
    win = win.to(pred.device, dtype=pred.dtype)

    mu1 = gf(pred, win)
    mu2 = gf(target, win)
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gf(pred * pred, win) - mu1_sq
    sigma2_sq = gf(target * target, win) - mu2_sq
    sigma12 = gf(pred * target, win) - mu1_mu2
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    # Average over channels → (B, H, W), take first batch element
    ssim_spatial = ssim_map.mean(dim=1)[0]  # (H, W)

    # Convert to error: 1 - ssim, clamped to [0, 1]
    return (1.0 - ssim_spatial).clamp(0, 1)


class LPIPSMetric:
    """LPIPS perceptual distance metric for evaluation.

    Uses pre-trained network features to measure perceptual similarity.
    Lower values indicate more similar images.

    Args:
        device: Device to run the model on ("cuda" or "cpu")
        net: Backbone network for LPIPS ("alex" or "vgg")
    """

    def __init__(self, device: str = "cuda", net: str = "vgg"):
        self._net = net
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        self._spatial_model = None

    def _get_spatial_model(self):
        """Lazily initialize the spatial LPIPS model (shares weights concept)."""
        if self._spatial_model is None:
            device = next(self.model.parameters()).device
            self._spatial_model = lpips.LPIPS(net=self._net, spatial=True).to(device)
            self._spatial_model.eval()
        return self._spatial_model

    def _prepare_inputs(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Validate and prepare inputs for LPIPS computation.

        Args:
            pred: Predicted image (B, 3, H, W) or (3, H, W), values in [0, 1]
            target: Target image, same shape as pred

        Returns:
            (pred, target) in [-1, 1] range with batch dimension
        """
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        device = next(self.model.parameters()).device
        if pred.device != device:
            pred = pred.to(device)
            target = target.to(device)

        # Clamp to handle tiny floating-point overshoots from rendering
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        # LPIPS expects [-1, 1] range
        pred = pred * 2 - 1
        target = target * 2 - 1
        return pred, target

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Compute LPIPS distance.

        Args:
            pred: Predicted image (B, 3, H, W) or (3, H, W), values in [0, 1]
            target: Target image, same shape as pred

        Returns:
            LPIPS distance (lower is better)
        """
        pred, target = self._prepare_inputs(pred, target)
        with torch.no_grad():
            return self.model(pred, target).mean().item()

    def spatial_map(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute per-pixel LPIPS spatial distance map.

        Args:
            pred: Predicted image (3, H, W), values in [0, 1]
            target: Target image (3, H, W), values in [0, 1]

        Returns:
            Spatial LPIPS map (H, W) with per-pixel perceptual distances
        """
        pred, target = self._prepare_inputs(pred, target)
        spatial_model = self._get_spatial_model()
        with torch.no_grad():
            # spatial=True returns (B, 1, H, W)
            spatial = spatial_model(pred, target)
        return spatial.squeeze(0).squeeze(0)  # (H, W)
