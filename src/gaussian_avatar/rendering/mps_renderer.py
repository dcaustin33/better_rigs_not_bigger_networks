"""MPS (Apple Silicon) Gaussian splatting renderer wrapping opensplat-metal.

This module provides:
- MPSRenderer: MPS-accelerated Gaussian splatting using opensplat-metal
- MPS_AVAILABLE: Boolean indicating if opensplat-metal is available
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gaussian_avatar.rendering.camera import Camera

try:
    import opensplat_metal

    MPS_AVAILABLE = True
except ImportError:
    opensplat_metal = None
    MPS_AVAILABLE = False

# SH band 0 normalization constant: 0.5 * sqrt(1/pi)
_C0 = 0.28209479177387814


def _check_mps_available() -> None:
    """Raise ImportError with helpful message if opensplat-metal is not available."""
    if not MPS_AVAILABLE:
        raise ImportError(
            "opensplat-metal is not available. Install with:\n"
            "  uv sync --extra mps\n"
            "Note: opensplat-metal requires an Apple Silicon Mac with MPS support."
        )


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    """Convert post-sigmoid RGB values to SH band-0 coefficients.

    Args:
        rgb: RGB colors in [0, 1] range (N, 3)

    Returns:
        SH band-0 coefficients (N, 1, 3)
    """
    return ((rgb - 0.5) / _C0).unsqueeze(1)


class MPSRenderer(nn.Module):
    """MPS-accelerated Gaussian splatting renderer using opensplat-metal.

    Provides the same output interface as GaussianRenderer but renders on
    Apple Silicon GPUs via Metal. Supports both RGB and spherical harmonics
    color modes.

    Args:
        background_color: Default background RGB (default: white)
        sh_degree: Spherical harmonics degree. None for post-activation
            RGB colors (default), or an int for SH coefficients.

    Raises:
        ImportError: If opensplat-metal is not available
    """

    def __init__(
        self,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sh_degree: int | None = None,
    ):
        _check_mps_available()
        super().__init__()
        self.register_buffer(
            "background",
            torch.tensor(background_color, dtype=torch.float32),
        )
        self.sh_degree = sh_degree

    def forward(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        colors: Tensor,
        opacities: Tensor,
        camera: Camera,
        background: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Render Gaussians to image via opensplat-metal.

        Args:
            means: Global positions (N, 3)
            quats: Quaternions in wxyz format (N, 4)
            scales: Linear scales (N, 3)
            colors: RGB colors (N, 3) or SH coefficients (N, K, 3)
            opacities: Opacity values (N,)
            camera: Camera parameters
            background: Optional override background RGB (3,)

        Returns:
            Dictionary containing:
                - rgb: Rendered RGB image (3, H, W)
                - alpha: Alpha/opacity map (H, W)
        """
        bg = background if background is not None else self.background
        device = means.device

        # Ensure all tensors are contiguous float32 on MPS
        means = means.to(device=device, dtype=torch.float32).contiguous()
        quats = quats.to(device=device, dtype=torch.float32).contiguous()
        scales = scales.to(device=device, dtype=torch.float32).contiguous()
        opacities = opacities.to(device=device, dtype=torch.float32).contiguous()
        bg = bg.to(device=device, dtype=torch.float32).contiguous()

        # Build SH coefficients based on color mode
        if self.sh_degree is not None:
            # SH mode: colors is (N, K, 3) raw SH coefficients
            sh_coeffs = colors.to(device=device, dtype=torch.float32).contiguous()
            sh_degree = self.sh_degree
        else:
            # RGB mode: convert post-sigmoid RGB to SH band-0
            rgb = colors.to(device=device, dtype=torch.float32).contiguous()
            sh_coeffs = rgb_to_sh0(rgb)
            sh_degree = 0

        viewmat = camera.viewmat.to(device=device, dtype=torch.float32).contiguous()
        K = camera.intrinsic.to(device=device, dtype=torch.float32).contiguous()

        rendered = opensplat_metal.render(
            means=means,
            quats=quats,
            scales=scales,
            sh_coeffs=sh_coeffs,
            opacities=opacities,
            viewmat=viewmat,
            K=K,
            img_height=camera.height,
            img_width=camera.width,
            sh_degree=sh_degree,
            background=bg,
        )

        # opensplat-metal returns (H, W, 3) → permute to (3, H, W)
        rgb_out = rendered.permute(2, 0, 1)

        # opensplat-metal doesn't expose per-pixel alpha directly;
        # estimate from difference with background
        bg_img = bg.view(3, 1, 1).expand_as(rgb_out)
        diff = (rgb_out - bg_img).abs().max(dim=0).values
        alpha = (diff > 1e-4).float()

        return {
            "rgb": rgb_out,
            "alpha": alpha,
        }
