"""Deformation module for pose-dependent Gaussian property corrections.

This module provides:
- Positional encoding for spatial coordinates
- Pose encoding (axis-angle and Euler XYZ to 6D rotation representation)
- DeformationMLP for predicting property offsets
"""

from gaussian_avatar.models.deformation.encoding import (
    positional_encoding,
    axis_angle_to_6d,
    euler_xyz_to_matrix,
    euler_xyz_to_6d,
)
from gaussian_avatar.models.deformation.mlp import (
    DeformationMLP,
    construct_quaternion_offset,
)

__all__ = [
    "positional_encoding",
    "axis_angle_to_6d",
    "euler_xyz_to_matrix",
    "euler_xyz_to_6d",
    "DeformationMLP",
    "construct_quaternion_offset",
]
