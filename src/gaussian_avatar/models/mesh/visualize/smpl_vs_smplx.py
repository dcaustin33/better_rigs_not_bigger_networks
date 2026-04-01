#!/usr/bin/env python3
"""Side-by-side comparison of SMPL and SMPL-X models.

This script shows the differences between SMPL and SMPL-X models with
the same body pose applied to both, highlighting mesh resolution and
joint structure differences.

Usage:
    uv run python -m gaussian_avatar.models.mesh.visualize.smpl_vs_smplx
    uv run python -m gaussian_avatar.models.mesh.visualize.smpl_vs_smplx --save comparison.png

Note: SMPL requires chumpy which is incompatible with Python 3.12+.
On Python 3.12+, this script will only show SMPL-X.
"""

import argparse
import sys

import numpy as np
import torch

from gaussian_avatar.models.mesh.smpl import SMPLMesh
from gaussian_avatar.models.mesh.visualize.common import (
    compute_mesh_bounds,
    SMPL_MODELS_ROOT,
    SMPLX_MODELS_ROOT,
)


def create_meshes(gender: str = "neutral") -> dict:
    """Create SMPL and SMPL-X mesh instances.

    Args:
        gender: Model gender

    Returns:
        Dictionary with 'smpl' and/or 'smplx' mesh instances
    """
    meshes = {}

    try:
        meshes["smplx"] = SMPLMesh(
            model_path=SMPLX_MODELS_ROOT,
            model_type="smplx",
            gender=gender,
            num_betas=10,
            device=torch.device("cpu"),
        )
        print(f"Loaded SMPL-X: {meshes['smplx'].num_vertices:,} vertices, "
              f"{meshes['smplx'].get_faces().shape[0]:,} faces, "
              f"{meshes['smplx'].num_joints} joints")
    except Exception as e:
        print(f"Failed to load SMPL-X: {e}")

    try:
        meshes["smpl"] = SMPLMesh(
            model_path=SMPL_MODELS_ROOT,
            model_type="smpl",
            gender=gender,
            num_betas=10,
            device=torch.device("cpu"),
        )
        print(f"Loaded SMPL: {meshes['smpl'].num_vertices:,} vertices, "
              f"{meshes['smpl'].get_faces().shape[0]:,} faces, "
              f"{meshes['smpl'].num_joints} joints")
    except Exception as e:
        print(f"SMPL not available: {e}")
        if sys.version_info >= (3, 12):
            print("(SMPL requires chumpy which is incompatible with Python 3.12+)")

    return meshes


def create_comparison_pose() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a pose that shows visible deformation.

    Returns:
        Tuple of (smpl_pose, smpl_shape, smplx_pose, smplx_shape)
    """
    # Create a pose with visible arm and leg movement
    smpl_pose = torch.zeros(72)  # SMPL: 72 dims
    smplx_pose = torch.zeros(66)  # SMPL-X: 66 dims

    # Bend right arm at elbow (different indices for SMPL vs SMPL-X)
    # SMPL: body_pose starts at index 3, right elbow is joint 19
    smpl_pose[3 + 19 * 3] = 1.5  # Bend elbow ~85 degrees

    # SMPL-X: Right elbow would be at different index in body_pose
    # Using similar joint for body (mapping may differ slightly)
    smplx_pose[3 + 18 * 3] = 1.5  # Approximate equivalent

    # Raise left arm slightly
    smpl_pose[3 + 16 * 3 + 2] = -0.5  # Left shoulder Z rotation
    smplx_pose[3 + 15 * 3 + 2] = -0.5  # SMPL-X equivalent

    # Bend one leg
    smpl_pose[3 + 4 * 3] = 0.5  # Left knee bend
    smplx_pose[3 + 3 * 3] = 0.5  # SMPL-X equivalent

    # Standard shape (neutral)
    smpl_shape = torch.zeros(10)
    smplx_shape = torch.zeros(10)

    return smpl_pose, smpl_shape, smplx_pose, smplx_shape


def visualize_comparison(
    meshes: dict,
    save_path: str | None = None,
) -> None:
    """Create side-by-side visualization of available meshes.

    Args:
        meshes: Dictionary of mesh name -> SMPLMesh instance
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    num_meshes = len(meshes)
    if num_meshes == 0:
        print("No meshes available to visualize")
        return

    smpl_pose, smpl_shape, smplx_pose, smplx_shape = create_comparison_pose()

    fig = plt.figure(figsize=(8 * num_meshes, 10))

    all_vertices = []
    for name, mesh in meshes.items():
        if name == "smpl":
            verts = mesh.forward(smpl_pose, smpl_shape).numpy()
        else:
            verts = mesh.forward(smplx_pose, smplx_shape).numpy()
        all_vertices.append(verts)

    all_verts = np.concatenate(all_vertices, axis=0)
    max_range, mid_x, mid_y, mid_z = compute_mesh_bounds(all_verts)
    max_range *= 1.1  # Add margin

    for idx, (name, mesh) in enumerate(meshes.items()):
        ax = fig.add_subplot(1, num_meshes, idx + 1, projection='3d')

        if name == "smpl":
            vertices = mesh.forward(smpl_pose, smpl_shape).numpy()
        else:
            vertices = mesh.forward(smplx_pose, smplx_shape).numpy()

        faces = mesh.get_faces().numpy()
        mesh_faces = vertices[faces]

        collection = Poly3DCollection(
            mesh_faces,
            alpha=0.7,
            facecolor='lightblue' if name == 'smplx' else 'lightyellow',
            edgecolor='gray',
            linewidth=0.05,
        )
        ax.add_collection3d(collection)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        title = f"{name.upper()}\n"
        title += f"Vertices: {mesh.num_vertices:,}\n"
        title += f"Faces: {faces.shape[0]:,}\n"
        title += f"Joints: {mesh.num_joints}"
        ax.set_title(title)

        ax.view_init(elev=10, azim=160)

    plt.suptitle("SMPL vs SMPL-X Comparison\n(Same pose applied to both)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()


def print_comparison_table(meshes: dict) -> None:
    """Print a table comparing mesh properties."""
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"{'Property':<25} {'SMPL':<15} {'SMPL-X':<15}")
    print("-" * 60)

    props = [
        ("Vertices", "num_vertices"),
        ("Joints", "num_joints"),
    ]

    for prop_name, attr in props:
        smpl_val = getattr(meshes.get("smpl"), attr, "N/A") if "smpl" in meshes else "N/A"
        smplx_val = getattr(meshes.get("smplx"), attr, "N/A") if "smplx" in meshes else "N/A"

        if isinstance(smpl_val, int):
            smpl_val = f"{smpl_val:,}"
        if isinstance(smplx_val, int):
            smplx_val = f"{smplx_val:,}"

        print(f"{prop_name:<25} {str(smpl_val):<15} {str(smplx_val):<15}")

    smpl_faces = meshes["smpl"].get_faces().shape[0] if "smpl" in meshes else "N/A"
    smplx_faces = meshes["smplx"].get_faces().shape[0] if "smplx" in meshes else "N/A"
    if isinstance(smpl_faces, int):
        smpl_faces = f"{smpl_faces:,}"
    if isinstance(smplx_faces, int):
        smplx_faces = f"{smplx_faces:,}"
    print(f"{'Faces':<25} {smpl_faces:<15} {smplx_faces:<15}")

    # Pose dimensions
    print(f"{'Pose dims (body only)':<25} {'72':<15} {'66':<15}")

    # Features
    print("-" * 60)
    print("Features:")
    print(f"  {'Body model':<23} {'Yes':<15} {'Yes':<15}")
    print(f"  {'Hands':<23} {'No':<15} {'Yes (MANO)':<15}")
    print(f"  {'Face':<23} {'No':<15} {'Yes (FLAME)':<15}")
    print(f"  {'Python 3.12+ support':<23} {'No (chumpy)':<15} {'Yes':<15}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SMPL and SMPL-X models side by side"
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
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Only print comparison table, no visualization",
    )

    args = parser.parse_args()

    print(f"Loading models ({args.gender})...")
    meshes = create_meshes(args.gender)

    if not meshes:
        print("Error: No models could be loaded")
        sys.exit(1)

    print_comparison_table(meshes)

    if not args.table_only:
        print("\nCreating visualization...")
        visualize_comparison(meshes, args.save)


if __name__ == "__main__":
    main()
