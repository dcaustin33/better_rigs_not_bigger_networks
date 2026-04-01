"""Gaussian splatting renderer wrapping gsplat.

This module provides:
- GaussianRenderer: CUDA-accelerated Gaussian splatting using gsplat
- MockRenderer: CPU-compatible mock for testing without gsplat
- GSPLAT_AVAILABLE: Boolean indicating if gsplat is available
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from gaussian_avatar.rendering.camera import Camera

try:
    from gsplat import rasterization

    GSPLAT_AVAILABLE = True
except ImportError:
    rasterization = None
    GSPLAT_AVAILABLE = False


def _check_gsplat_available() -> None:
    """Raise ImportError with helpful message if gsplat is not available."""
    if not GSPLAT_AVAILABLE:
        raise ImportError(
            "gsplat is not available. Install with CUDA support:\n"
            "  uv sync --extra cuda\n"
            "Or install manually:\n"
            "  pip install gsplat\n"
            "Note: gsplat requires a CUDA-capable GPU."
        )


class MockRenderer(nn.Module):
    """Mock renderer for CPU testing.

    Provides the same interface as GaussianRenderer but returns
    placeholder tensors. Useful for testing training loop structure
    without requiring CUDA.

    Args:
        allow_forward: If True, return zeros instead of raising.
            If False (default), raise RuntimeError on forward.
    """

    def __init__(self, allow_forward: bool = False):
        super().__init__()
        self.allow_forward = allow_forward
        self.register_buffer(
            "background",
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        )

    def forward(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        colors: Tensor,
        opacities: Tensor,
        camera: Camera,
        background: Tensor | None = None,
        absgrad: bool = False,
    ) -> dict[str, Tensor]:
        """Render placeholder outputs.

        Args:
            means: Global positions (N, 3)
            quats: Quaternions in wxyz format (N, 4)
            scales: Linear scales (N, 3)
            colors: RGB colors (N, 3)
            opacities: Opacity values (N,)
            camera: Camera parameters
            background: Optional background RGB (3,), ignored
            absgrad: Accepted for interface compatibility, ignored

        Returns:
            Dictionary with placeholder outputs:
                - rgb: (3, H, W) zeros
                - alpha: (H, W) zeros
                - viewspace_points: (N, 2) zeros with requires_grad=True
                - visibility_filter: (N,) all True
                - radii: (N,) ones

        Raises:
            RuntimeError: If allow_forward=False
        """
        if not self.allow_forward:
            raise RuntimeError(
                "MockRenderer.forward() called but allow_forward=False. "
                "Actual rendering requires gsplat and CUDA. "
                "Set allow_forward=True for testing with placeholder outputs."
            )

        H, W = camera.height, camera.width
        device = means.device
        N = means.shape[0]

        return {
            "rgb": torch.zeros(3, H, W, device=device),
            "alpha": torch.zeros(H, W, device=device),
            "viewspace_points": torch.zeros(N, 2, device=device, requires_grad=True),
            "visibility_filter": torch.ones(N, dtype=torch.bool, device=device),
            "radii": torch.ones(N, device=device),
        }


class GaussianRenderer(nn.Module):
    """Wrapper for gsplat rasterization.

    Handles conversion from GaussianAvatar output format to gsplat
    input format and returns rendered images with auxiliary info.

    Args:
        background_color: Default background RGB (default: black)
        sh_degree: Spherical harmonics degree. None for post-activation
            RGB colors (default), or an int for SH coefficients.

    Raises:
        ImportError: If gsplat is not available (requires CUDA)
    """

    def __init__(
        self,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sh_degree: int | None = None,
    ):
        _check_gsplat_available()
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
        absgrad: bool = False,
    ) -> dict[str, Tensor]:
        """Render Gaussians to image.

        Args:
            means: Global positions (N, 3)
            quats: Quaternions in wxyz format (N, 4)
            scales: Linear scales (N, 3)
            colors: RGB colors (N, 3)
            opacities: Opacity values (N,)
            camera: Camera parameters
            background: Optional override background RGB (3,)
            absgrad: If True, gsplat computes absolute-value gradients
                for means2d during backward, stored as means2d.absgrad.
                Used for improved densification signal. Default False.

        Returns:
            Dictionary containing:
                - rgb: Rendered RGB image (3, H, W)
                - alpha: Alpha/opacity map (H, W)
                - viewspace_points: 2D positions for densification (N, 2)
                - visibility_filter: Boolean mask of visible Gaussians (N,)
                - radii: Screen-space radii (N,)
        """
        bg = background if background is not None else self.background

        # gsplat expects batched inputs for viewmats and Ks
        viewmat = camera.viewmat.unsqueeze(0)  # (1, 4, 4)
        K = camera.intrinsic.unsqueeze(0)  # (1, 3, 3)

        renders, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=camera.width,
            height=camera.height,
            sh_degree=self.sh_degree,
            backgrounds=bg.unsqueeze(0),  # (1, D=3) for C=1
            packed=False,  # Workaround for gsplat 1.5.3 bug #764
            absgrad=absgrad,
        )

        rgb = renders[0].permute(2, 0, 1)
        alpha = alphas[0, :, :, 0]
        # retain_grad on the FULL means2d tensor before slicing, because
        # the sliced view is a dead autograd branch (gsplat uses means2d
        # internally for rasterization, so gradients only flow through
        # the unsliced tensor).
        means2d = info["means2d"]
        if means2d.requires_grad:
            means2d.retain_grad()
        viewspace_points = means2d[0]
        radii = info["radii"][0]  # (N,) or (N, 2)
        if radii.dim() == 2:
            # gsplat 1.5+ returns per-axis radii (N, 2); reduce to (N,)
            visibility_filter = (radii > 0).any(dim=-1)
            radii = radii.amax(dim=-1)
        else:
            visibility_filter = radii > 0

        return {
            "rgb": rgb,
            "alpha": alpha,
            "viewspace_points": viewspace_points,
            "visibility_filter": visibility_filter,
            "radii": radii,
            "_means2d": means2d,
        }
