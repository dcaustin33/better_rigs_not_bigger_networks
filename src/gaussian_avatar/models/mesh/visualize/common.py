"""Common utilities for mesh visualization scripts.

This module provides shared constants, functions, and utilities used across
the mesh visualization scripts to avoid code duplication.
"""

import numpy as np
import torch

from gaussian_avatar.models.mesh.smpl import SMPLMesh


# Model paths
SMPL_MODELS_ROOT = "mesh_models"
SMPLX_MODELS_ROOT = "mesh_models/models"
DATASET_ROOT = "datasets/people_snapshot_public"


def create_mesh(model_type: str, gender: str = "neutral") -> SMPLMesh:
    """Create a mesh instance for the specified model type.

    Args:
        model_type: Either 'smpl' or 'smplx'
        gender: Model gender ('neutral', 'male', or 'female')

    Returns:
        SMPLMesh instance
    """
    if model_type == "smpl":
        model_path = SMPL_MODELS_ROOT
    else:
        model_path = SMPLX_MODELS_ROOT

    return SMPLMesh(
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        num_betas=10,
        device=torch.device("cpu"),
    )


def compute_mesh_bounds(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """Compute bounding box and center for mesh vertices.

    Args:
        vertices: Vertex positions (V, 3) or stacked (N, V, 3)

    Returns:
        Tuple of (max_range, mid_x, mid_y, mid_z) for axis setup
    """
    if vertices.ndim == 3:
        vertices = vertices.reshape(-1, 3)

    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2

    return max_range, mid_x, mid_y, mid_z


def setup_3d_axes(
    ax,
    vertices: np.ndarray,
    scale: float = 1.0,
) -> None:
    """Configure 3D axes with equal aspect ratio based on mesh bounds.

    Args:
        ax: Matplotlib 3D axes object
        vertices: Vertex positions for computing bounds
        scale: Scale factor for the bounding box (default: 1.0)
    """
    max_range, mid_x, mid_y, mid_z = compute_mesh_bounds(vertices)
    max_range *= scale

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def add_common_parser_args(parser) -> None:
    """Add common CLI arguments to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--model-type",
        choices=["smpl", "smplx"],
        default="smplx",
        help="Model type to visualize (default: smplx)",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Model gender (default: neutral)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save visualization to file instead of showing",
    )
