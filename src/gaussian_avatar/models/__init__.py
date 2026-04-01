"""Models module for Gaussian Avatar.

This module provides:
- Mesh: LBS-deformable triangle mesh abstractions (BaseMesh, SMPLMesh)
- Gaussian: Gaussian parameter storage and local-to-global transformation
- Avatar: Top-level model combining mesh, Gaussians, and deformation
"""

from gaussian_avatar.models.mesh import BaseMesh, SMPLMesh
from gaussian_avatar.models.gaussian import (
    GaussianModel,
    normalize_quaternion,
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)
from gaussian_avatar.models.avatar import GaussianAvatar

__all__ = [
    # Mesh
    "BaseMesh",
    "SMPLMesh",
    # Gaussian
    "GaussianModel",
    "normalize_quaternion",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_multiply",
    # Avatar
    "GaussianAvatar",
]
