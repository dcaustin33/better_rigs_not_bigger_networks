"""Diagnostic multi-view rendering for Gaussian avatar inspection.

Provides orbit camera construction and multi-view grid rendering so that
raw Gaussian splats can be examined from arbitrary viewpoints during
evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from gaussian_avatar.rendering.camera import Camera


def _rotation_y(angle_deg: float, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Build a 3x3 rotation matrix around the Y axis.

    Args:
        angle_deg: Rotation angle in degrees.
        dtype: Tensor dtype.
        device: Tensor device.

    Returns:
        (3, 3) rotation matrix.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        dtype=dtype,
        device=device,
    )


def _rotation_x(angle_deg: float, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Build a 3x3 rotation matrix around the X axis.

    Args:
        angle_deg: Rotation angle in degrees.
        dtype: Tensor dtype.
        device: Tensor device.

    Returns:
        (3, 3) rotation matrix.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[1, 0, 0], [0, c, -s], [0, s, c]],
        dtype=dtype,
        device=device,
    )


def _orbit_extrinsic(
    base_extrinsic: Tensor,
    center: Tensor,
    rotation_3x3: Tensor,
) -> Tensor:
    """Compute a new extrinsic by orbiting the camera around *center*.

    The orbit is implemented as a scene rotation (by ``-R``) so that the
    resulting camera views the same subject from a different angle.

    Args:
        base_extrinsic: (4, 4) original world-to-camera transform.
        center: (3,) world-space point to orbit around.
        rotation_3x3: (3, 3) rotation to apply (positive = camera moves in
            that direction around center).

    Returns:
        (4, 4) new extrinsic.
    """
    R_inv = rotation_3x3.T  # inverse scene rotation
    M = torch.eye(4, dtype=base_extrinsic.dtype, device=base_extrinsic.device)
    M[:3, :3] = R_inv
    M[:3, 3] = center - R_inv @ center
    return base_extrinsic @ M


def create_orbit_cameras(
    base_camera: Camera,
    center: Tensor,
    azimuths: list[float] | None = None,
    include_top: bool = True,
) -> list[tuple[str, Camera]]:
    """Create cameras orbiting around *center* at various azimuths.

    The base camera is treated as the 0-degree reference.  Additional
    cameras are created by rotating around the world Y axis.  An optional
    top-down view is added by rotating around the world X axis.

    Args:
        base_camera: The original dataset camera (used as 0-degree view).
        center: (3,) world-space point to orbit around (e.g. Gaussian
            centroid).
        azimuths: List of azimuth angles in degrees.  Defaults to
            ``[0, 60, 120, 180, 240, 300]``.
        include_top: If True, append a top-down view (~70 deg elevation).

    Returns:
        List of ``(label, Camera)`` pairs.
    """
    if azimuths is None:
        azimuths = [0, 60, 120, 180, 240, 300]

    dtype = base_camera.extrinsic.dtype
    device = base_camera.extrinsic.device

    cameras: list[tuple[str, Camera]] = []
    for az in azimuths:
        R = _rotation_y(az, dtype=dtype, device=device)
        ext = _orbit_extrinsic(base_camera.extrinsic, center, R)
        label = f"{int(az)}deg"
        cameras.append((
            label,
            Camera(
                intrinsic=base_camera.intrinsic,
                extrinsic=ext,
                width=base_camera.width,
                height=base_camera.height,
            ),
        ))

    if include_top:
        R = _rotation_x(-70.0, dtype=dtype, device=device)
        ext = _orbit_extrinsic(base_camera.extrinsic, center, R)
        cameras.append((
            "top",
            Camera(
                intrinsic=base_camera.intrinsic,
                extrinsic=ext,
                width=base_camera.width,
                height=base_camera.height,
            ),
        ))

    return cameras


def render_multiview_grid(
    renderer,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    colors: Tensor,
    opacities: Tensor,
    base_camera: Camera,
    center: Tensor | None = None,
    azimuths: list[float] | None = None,
    include_top: bool = True,
) -> Image.Image:
    """Render Gaussians from multiple viewpoints and return a labelled grid.

    Args:
        renderer: ``GaussianRenderer`` (or ``MockRenderer``).
        means: (N, 3) global Gaussian positions.
        quats: (N, 4) quaternions (wxyz).
        scales: (N, 3) linear scales.
        colors: (N, 3) RGB colours.
        opacities: (N,) opacity values.
        base_camera: The dataset camera for the current frame.
        center: (3,) orbit centre.  If *None*, uses the centroid of *means*.
        azimuths: Azimuth angles to render.  Defaults to
            ``[0, 60, 120, 180, 240, 300]``.
        include_top: Whether to include a top-down view.

    Returns:
        PIL RGB Image containing the labelled view grid.
    """
    if center is None:
        center = means.mean(dim=0).detach()

    orbit_cams = create_orbit_cameras(
        base_camera, center, azimuths=azimuths, include_top=include_top,
    )

    panels: list[tuple[str, np.ndarray]] = []
    for label, cam in orbit_cams:
        out = renderer(
            means=means,
            quats=quats,
            scales=scales,
            colors=colors,
            opacities=opacities,
            camera=cam,
        )
        rgb = out["rgb"]  # (3, H, W)
        rgb_np = (rgb.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
        panels.append((label, rgb_np))

    return _compose_grid(panels)


def _compose_grid(
    panels: list[tuple[str, np.ndarray]],
    max_cols: int = 4,
    label_height: int = 24,
) -> Image.Image:
    """Arrange labelled image panels into a grid.

    Args:
        panels: List of ``(label, HxWx3 uint8 array)`` pairs.
        max_cols: Maximum number of columns in the grid.
        label_height: Pixel height reserved for the text label above each
            panel.

    Returns:
        Composite PIL RGB Image.
    """
    if not panels:
        return Image.new("RGB", (1, 1))

    n = len(panels)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    H, W, _ = panels[0][1].shape
    cell_h = H + label_height
    cell_w = W

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(32, 32, 32))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, (label, img_np) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * cell_h

        # Draw label
        draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255), font=font)

        # Paste rendered image below label
        panel_img = Image.fromarray(img_np)
        grid.paste(panel_img, (x0, y0 + label_height))

    return grid
