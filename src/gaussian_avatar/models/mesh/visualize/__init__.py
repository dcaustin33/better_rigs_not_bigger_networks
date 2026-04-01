"""Mesh visualization utilities.

Standalone scripts for visualizing SMPL/SMPL-X meshes.

Usage:
    uv run python -m gaussian_avatar.models.mesh.visualize.canonical --model-type smplx
    uv run python -m gaussian_avatar.models.mesh.visualize.poses --model-type smplx --synthetic
    uv run python -m gaussian_avatar.models.mesh.visualize.normals --model-type smplx
    uv run python -m gaussian_avatar.models.mesh.visualize.smpl_vs_smplx

Note: These utilities require optional dependencies (matplotlib, trimesh, pyglet).
Install with: uv sync --extra test
"""

# Lazy imports to avoid requiring matplotlib at package import time
__all__ = [
    "common",
    "canonical",
    "poses",
    "normals",
    "smpl_vs_smplx",
]
