"""Shared utilities for loss functions."""

import torch
import torch.nn.functional as F


def compute_boundary_weight_map(
    mask: torch.Tensor,
    boundary_weight: float,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Compute a per-pixel weight map that upweights silhouette boundaries.

    Extracts boundary pixels via morphological dilation minus erosion
    (using max-pool), then builds a weight map:
        weight = mask * (1 + boundary_weight * boundary)

    Interior foreground pixels get weight 1, boundary pixels get
    weight (1 + boundary_weight), background stays 0.

    Args:
        mask: Binary foreground mask (B, 1, H, W), values in [0, 1]
        boundary_weight: Extra weight multiplier for boundary pixels.
            0 recovers the plain binary mask.
        kernel_size: Dilation/erosion kernel size. Controls the width of the
            boundary band in pixels (~kernel_size - 1 pixels wide).

    Returns:
        Weight map (B, 1, H, W) with boundary pixels upweighted
    """
    if boundary_weight == 0.0:
        return mask

    pad = kernel_size // 2

    # Morphological dilation via max-pool
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)

    # Morphological erosion via negated max-pool
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)

    # Boundary band: 1 at silhouette edges, 0 elsewhere
    boundary = dilated - eroded

    # Upweight boundary pixels within the foreground mask
    weight_map = mask * (1.0 + boundary_weight * boundary)

    return weight_map


def composite_with_random_background(
    image: torch.Tensor,
    alpha: torch.Tensor,
    background: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Composite straight (non-premultiplied) image over background.

    Use this for ground-truth images whose RGB has not been
    pre-composited.  For renderer output (premultiplied alpha from a
    black-background render), use ``composite_premultiplied`` instead.

    Args:
        image: Straight-alpha RGB image (B, 3, H, W)
        alpha: Alpha channel (B, 1, H, W) or (B, H, W)
        background: Optional background color (3,) or (B, 3, 1, 1).
                    If None, generates random color.

    Returns:
        Tuple of (composited_image, background_color)
    """
    # Auto-unsqueeze 3D alpha (B, H, W) -> (B, 1, H, W)
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)
    if image.dim() != 4 or alpha.dim() != 4:
        raise ValueError(
            f"Expected 4D image and alpha tensors, got image.dim()={image.dim()}, "
            f"alpha.dim()={alpha.dim()}. Reshape to (B, C, H, W) before calling."
        )

    if background is None:
        background = torch.rand(3, device=image.device, dtype=image.dtype)

    if background.dim() == 1:
        background = background.view(1, 3, 1, 1)

    composited = image * alpha + background * (1.0 - alpha)

    return composited, background


def composite_premultiplied(
    premultiplied_rgb: torch.Tensor,
    alpha: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    """Composite a premultiplied-alpha image over a background.

    The renderer outputs premultiplied RGB when using a black background
    (i.e. ``rgb = accumulated_color``).  Re-compositing onto a new
    background only requires adding the background contribution; do NOT
    multiply by alpha again.

    Args:
        premultiplied_rgb: Premultiplied RGB (B, 3, H, W)
        alpha: Alpha channel (B, 1, H, W) or (B, H, W)
        background: Background color (3,) or (B, 3, 1, 1)

    Returns:
        Composited image (B, 3, H, W)
    """
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(1)
    if background.dim() == 1:
        background = background.view(1, 3, 1, 1)

    return premultiplied_rgb + background * (1.0 - alpha)


def ensure_batch_dim(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Add batch dimension if missing.

    Args:
        tensor: Input tensor of shape (..., H, W) or (B, ..., H, W)

    Returns:
        Tuple of (tensor with batch dim, was_batched flag)
    """
    was_batched = tensor.dim() == 4
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor, was_batched


def get_device_from_tensors(*tensors: torch.Tensor | None) -> torch.device:
    """Get device from first non-None tensor.

    Args:
        *tensors: Variable number of tensors (some may be None)

    Returns:
        Device of the first non-None tensor, or CPU if all are None
    """
    for t in tensors:
        if t is not None:
            return t.device
    return torch.device("cpu")
