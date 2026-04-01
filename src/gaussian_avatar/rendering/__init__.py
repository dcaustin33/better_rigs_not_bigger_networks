"""Rendering module for Gaussian splatting.

This module provides:
- Camera: Dataclass for camera parameters used in gsplat rendering
- GaussianRenderer: Wrapper for gsplat rasterization (requires CUDA)
- MPSRenderer: Wrapper for opensplat-metal rasterization (requires Apple Silicon)
- MockRenderer: CPU-compatible mock renderer for testing
"""

from gaussian_avatar.rendering.camera import Camera
from gaussian_avatar.rendering.mps_renderer import (
    MPSRenderer,
    MPS_AVAILABLE,
)
from gaussian_avatar.rendering.renderer import (
    GaussianRenderer,
    MockRenderer,
    GSPLAT_AVAILABLE,
)

__all__ = [
    "Camera",
    "GaussianRenderer",
    "MPSRenderer",
    "MockRenderer",
    "GSPLAT_AVAILABLE",
    "MPS_AVAILABLE",
]
