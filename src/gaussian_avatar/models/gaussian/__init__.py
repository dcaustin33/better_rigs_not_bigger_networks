"""Gaussian representation and parameter management.

This module provides:
- GaussianModel: Parameter storage and local-to-global transformation
- Rotation utilities for quaternion operations
"""

from gaussian_avatar.models.gaussian.model import GaussianModel
from gaussian_avatar.models.gaussian.rotations import (
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)

__all__ = [
    "GaussianModel",
    "normalize_quaternion",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_multiply",
]
