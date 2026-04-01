"""Mesh abstraction layer for LBS-deformable triangle meshes.

This module provides:
- BaseMesh: Abstract base class defining the mesh interface
- SMPLMesh: SMPL/SMPL-X implementation using smplx library
- MHRMesh: MHR (Momentum Human Rig) implementation (optional, requires MHR assets)
- MHR_AVAILABLE: Boolean flag indicating if MHR support is available
- create_mesh(): Factory function to create mesh instances by model type
- Triangle computation utilities (centroids, local frames, stretches)
"""

import torch

from gaussian_avatar.models.mesh.base import BaseMesh
from gaussian_avatar.models.mesh.smpl import SMPLMesh

try:
    from gaussian_avatar.models.mesh.mhr import MHRMesh

    MHR_AVAILABLE = True
except ImportError:
    MHR_AVAILABLE = False

__all__ = ["BaseMesh", "SMPLMesh", "MHR_AVAILABLE", "create_mesh"]
if MHR_AVAILABLE:
    __all__.append("MHRMesh")


def create_mesh(
    model_type: str,
    model_path: str,
    gender: str = "neutral",
    num_betas: int = 10,
    lod: int = 1,
    device: torch.device = torch.device("cpu"),
) -> BaseMesh:
    """Create a mesh instance based on model type string.

    Factory function that dispatches to the appropriate mesh class
    based on the model_type parameter.

    Args:
        model_type: One of "smpl", "smplx", or "mhr"
        model_path: Path to the model files directory
        gender: Model gender ("male", "female", "neutral"). Used by SMPL/SMPL-X only.
        num_betas: Number of shape coefficients. Used by SMPL/SMPL-X only.
        lod: Level-of-detail (0-6). Used by MHR only.
        device: Device to place tensors on

    Returns:
        A BaseMesh subclass instance

    Raises:
        ValueError: If model_type is not recognized
        RuntimeError: If MHR is requested but not available
    """
    if model_type in ("smpl", "smplx"):
        return SMPLMesh(
            model_path=model_path,
            model_type=model_type,
            gender=gender,
            num_betas=num_betas,
            device=device,
        )
    elif model_type == "mhr":
        if not MHR_AVAILABLE:
            raise RuntimeError(
                "MHR mesh requested but MHR dependencies are not available. "
                "Ensure the MHR TorchScript model is accessible and all "
                "required packages are installed."
            )
        return MHRMesh(
            model_path=model_path,
            lod=lod,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be 'smpl', 'smplx', or 'mhr'."
        )
