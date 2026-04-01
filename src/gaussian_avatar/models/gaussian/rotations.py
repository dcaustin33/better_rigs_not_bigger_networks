"""Quaternion rotation utilities for Gaussian transformations.

All quaternions use wxyz ordering: q = (w, x, y, z) where w is the scalar component.
This matches PyTorch3D and most academic literature conventions.
"""

import torch
from torch import Tensor


def normalize_quaternion(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize quaternions to unit length.

    Args:
        q: Quaternions (..., 4) in wxyz format
        eps: Small value for numerical stability

    Returns:
        Unit quaternions (..., 4) in wxyz format
    """
    norm = q.norm(dim=-1, keepdim=True)
    return q / norm.clamp(min=eps)


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Convert quaternions to rotation matrices.

    For unit quaternion q = (w, x, y, z):
    R = [[1-2(y²+z²), 2(xy-wz),   2(xz+wy)  ],
         [2(xy+wz),   1-2(x²+z²), 2(yz-wx)  ],
         [2(xz-wy),   2(yz+wx),   1-2(x²+y²)]]

    Args:
        q: Quaternions (..., 4) in wxyz format (does not need to be normalized)

    Returns:
        Rotation matrices (..., 3, 3)
    """
    q = normalize_quaternion(q)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )

    return R


def matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert rotation matrices to quaternions using Shepperd's method.

    This method is numerically stable for all rotation matrices by choosing
    the largest diagonal element to compute the quaternion.

    Args:
        R: Rotation matrices (..., 3, 3)

    Returns:
        Quaternions (..., 4) in wxyz format with positive w (when possible)
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    n = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    d0 = R[:, 0, 0]
    d1 = R[:, 1, 1]
    d2 = R[:, 2, 2]

    q = torch.zeros(n, 4, dtype=R.dtype, device=R.device)

    mask_w = trace > 0
    if mask_w.any():
        s = torch.sqrt(1.0 + trace[mask_w]) * 2  # s = 4w
        q[mask_w, 0] = 0.25 * s
        q[mask_w, 1] = (R[mask_w, 2, 1] - R[mask_w, 1, 2]) / s
        q[mask_w, 2] = (R[mask_w, 0, 2] - R[mask_w, 2, 0]) / s
        q[mask_w, 3] = (R[mask_w, 1, 0] - R[mask_w, 0, 1]) / s

    mask_x = (~mask_w) & (d0 > d1) & (d0 > d2)
    if mask_x.any():
        s = torch.sqrt(1.0 + d0[mask_x] - d1[mask_x] - d2[mask_x]) * 2  # s = 4x
        q[mask_x, 0] = (R[mask_x, 2, 1] - R[mask_x, 1, 2]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (R[mask_x, 0, 1] + R[mask_x, 1, 0]) / s
        q[mask_x, 3] = (R[mask_x, 0, 2] + R[mask_x, 2, 0]) / s

    mask_y = (~mask_w) & (~mask_x) & (d1 > d2)
    if mask_y.any():
        s = torch.sqrt(1.0 + d1[mask_y] - d0[mask_y] - d2[mask_y]) * 2  # s = 4y
        q[mask_y, 0] = (R[mask_y, 0, 2] - R[mask_y, 2, 0]) / s
        q[mask_y, 1] = (R[mask_y, 0, 1] + R[mask_y, 1, 0]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (R[mask_y, 1, 2] + R[mask_y, 2, 1]) / s

    mask_z = (~mask_w) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + d2[mask_z] - d0[mask_z] - d1[mask_z]) * 2  # s = 4z
        q[mask_z, 0] = (R[mask_z, 1, 0] - R[mask_z, 0, 1]) / s
        q[mask_z, 1] = (R[mask_z, 0, 2] + R[mask_z, 2, 0]) / s
        q[mask_z, 2] = (R[mask_z, 1, 2] + R[mask_z, 2, 1]) / s
        q[mask_z, 3] = 0.25 * s

    q = normalize_quaternion(q)

    q = torch.where(q[:, 0:1] < 0, -q, q)

    return q.reshape(*batch_shape, 4)


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Compose two quaternions (Hamilton product).

    Result represents rotation q2 followed by rotation q1.
    For vectors: v' = q1 * q2 * v * (q1*q2)^-1

    Args:
        q1: First quaternions (..., 4) in wxyz format
        q2: Second quaternions (..., 4) in wxyz format

    Returns:
        Composed quaternions (..., 4) in wxyz format (normalized)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = torch.stack([w, x, y, z], dim=-1)
    return normalize_quaternion(result)
