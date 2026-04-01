"""Input encoding functions for the deformation MLP.

This module provides:
- Sinusoidal positional encoding for spatial coordinates
- Axis-angle to 6D rotation representation conversion
- Full MLP input assembly combining positions and pose
"""

import math

import torch
from torch import Tensor


def hann_window_weight(num_frequencies: int, alpha: float, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
    """Compute Hann window weights for frequency band annealing.

    Implements a truncated Hann window that slides across frequency bands as
    alpha increases. Bands below alpha are fully active, bands above are
    inactive, and the band at alpha smoothly transitions.

    From Nerfies (Park et al. 2021): w_j(alpha) = (1 - cos(pi * clamp(alpha - j, 0, 1))) / 2

    Args:
        num_frequencies: Number of frequency bands L
        alpha: Window position in [0, L]. Controls how many bands are active.
        device: Tensor device
        dtype: Tensor dtype

    Returns:
        Per-band weights (L,) in [0, 1]
    """
    bands = torch.arange(num_frequencies, device=device, dtype=dtype or torch.float32)
    x = torch.clamp(alpha - bands, 0.0, 1.0)
    return (1.0 - torch.cos(math.pi * x)) / 2.0


def positional_encoding(x: Tensor, num_frequencies: int = 6, alpha: float | None = None) -> Tensor:
    """Apply sinusoidal positional encoding with optional frequency annealing.

    Maps 3D coordinates to higher-dimensional representation using sinusoidal
    functions at increasing frequencies, enabling the MLP to learn high-frequency
    spatial variations.

    When alpha is provided, applies Hann window annealing to progressively
    enable higher-frequency bands during training (coarse-to-fine).

    Args:
        x: Input coordinates (*, 3)
        num_frequencies: Number of frequency bands L
        alpha: Hann window position for frequency annealing. When None, all
            bands have full weight. When provided, bands are weighted by
            w_j = (1 - cos(pi * clamp(alpha - j, 0, 1))) / 2.

    Returns:
        Encoded features (*, 3 + 6*L) where output order is:
        [x, y, z, sin(π·x), sin(π·y), sin(π·z), cos(π·x), cos(π·y), cos(π·z),
         sin(2π·x), ..., cos(2^(L-1)·π·z)]

    Formula:
        PE(p) = [p, w_0·sin(π·p), w_0·cos(π·p), w_1·sin(2π·p), w_1·cos(2π·p), ...,
                 w_{L-1}·sin(2^(L-1)·π·p), w_{L-1}·cos(2^(L-1)·π·p)]
    """
    freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=x.dtype, device=x.device)
    freq_bands = freq_bands * torch.pi

    x_scaled = x.unsqueeze(-2) * freq_bands.unsqueeze(-1)  # (*, L, 3)
    sin_features = torch.sin(x_scaled)
    cos_features = torch.cos(x_scaled)

    if alpha is not None:
        weights = hann_window_weight(num_frequencies, alpha, device=x.device, dtype=x.dtype)
        # weights: (L,) → (L, 1) for broadcasting with (*, L, 3)
        w = weights.unsqueeze(-1)
        sin_features = sin_features * w
        cos_features = cos_features * w

    batch_shape = x.shape[:-1]
    sin_cos = torch.stack([sin_features, cos_features], dim=-2)
    encoded = sin_cos.reshape(*batch_shape, num_frequencies * 6)

    return torch.cat([x, encoded], dim=-1)


def encode_mlp_input(
    posed_positions: Tensor,
    pose: Tensor,
    num_frequencies: int = 6,
) -> Tensor:
    """Assemble full MLP input from posed positions and pose.

    Encodes posed positions with positional encoding, converts pose to 6D
    representation, and broadcasts pose to all Gaussians.

    Args:
        posed_positions: (N, 3) Gaussian positions after deformation
        pose: (J*3,) axis-angle pose vector (72 for SMPL, 165 for SMPL-X)
        num_frequencies: Number of positional encoding frequency bands

    Returns:
        (N, D) MLP input where D = (3+6*L) + J*6
    """
    N = posed_positions.shape[0]
    dtype = posed_positions.dtype

    pe_posed = positional_encoding(posed_positions, num_frequencies)

    pose_6d = axis_angle_to_6d(pose.reshape(-1, 3))
    pose_broadcast = pose_6d.flatten().unsqueeze(0).expand(N, -1).to(dtype=dtype)

    return torch.cat([pe_posed, pose_broadcast], dim=-1)


def axis_angle_to_6d(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle rotations to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation matrix,
    which provides a continuous parameterization without singularities.

    Args:
        axis_angle: Axis-angle rotations (*, 3) where magnitude is angle and
                    direction is rotation axis

    Returns:
        6D representation (*, 6) containing the first two columns of the
        rotation matrix, flattened as [r11, r21, r31, r12, r22, r32]

    Note:
        Uses Rodrigues formula for axis-angle to rotation matrix conversion:
        R = I + sin(θ)·K + (1 - cos(θ))·K²
        where K is the skew-symmetric matrix of the unit rotation axis.
    """
    eps = 1e-8
    theta = axis_angle.norm(dim=-1, keepdim=True)
    safe_theta = theta.clamp(min=eps)
    k = axis_angle / safe_theta

    # Skew-symmetric matrix K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=-1),
        torch.stack([kz, zeros, -kx], dim=-1),
        torch.stack([-ky, kx, zeros], dim=-1),
    ], dim=-2)

    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    I = I.expand(*axis_angle.shape[:-1], 3, 3)
    K_squared = torch.matmul(K, K)
    R = I + sin_theta * K + (1 - cos_theta) * K_squared

    # Near-zero angles return identity to avoid numerical issues
    is_small = (theta < eps).squeeze(-1)
    if is_small.any():
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                   dtype=axis_angle.dtype, device=axis_angle.device)
        col1 = R[..., :, 0]
        col2 = R[..., :, 1]
        result = torch.cat([col1, col2], dim=-1)
        result = torch.where(is_small.unsqueeze(-1), identity_6d, result)
        return result

    col1 = R[..., :, 0]
    col2 = R[..., :, 1]
    return torch.cat([col1, col2], dim=-1)


def euler_xyz_to_matrix(euler_xyz: Tensor) -> Tensor:
    """Convert Euler XYZ rotations to rotation matrices.

    Applies rotations in the order R = Rz @ Ry @ Rx (extrinsic XYZ convention),
    where euler_xyz[..., 0] is the X rotation, [..1] is Y, [..2] is Z.

    Args:
        euler_xyz: Euler XYZ angles in radians (*, 3)

    Returns:
        Rotation matrices (*, 3, 3)
    """
    x = euler_xyz[..., 0]
    y = euler_xyz[..., 1]
    z = euler_xyz[..., 2]

    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    # R = Rz @ Ry @ Rx, computed element-wise for efficiency
    # Row 0
    r00 = cy * cz
    r01 = sx * sy * cz - cx * sz
    r02 = cx * sy * cz + sx * sz
    # Row 1
    r10 = cy * sz
    r11 = sx * sy * sz + cx * cz
    r12 = cx * sy * sz - sx * cz
    # Row 2
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)

    return R


def euler_xyz_to_6d(euler_xyz: Tensor) -> Tensor:
    """Convert Euler XYZ rotations to 6D rotation representation.

    The 6D representation consists of the first two columns of the rotation matrix,
    providing a continuous parameterization without singularities. This is the
    Euler XYZ counterpart to axis_angle_to_6d(), used for MHR pose encoding.

    Args:
        euler_xyz: Euler XYZ angles in radians (*, 3)

    Returns:
        6D representation (*, 6) containing the first two columns of the
        rotation matrix, flattened as [r11, r21, r31, r12, r22, r32]
    """
    R = euler_xyz_to_matrix(euler_xyz)
    col1 = R[..., :, 0]
    col2 = R[..., :, 1]
    return torch.cat([col1, col2], dim=-1)
