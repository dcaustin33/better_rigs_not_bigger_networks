#!/usr/bin/env python3
"""Visualize canonical A-pose mesh for SMPL or SMPL-X models.

This script renders the canonical A-pose mesh using matplotlib or trimesh,
allowing visual verification that the A-pose is correctly computed with
arms at approximately 45 degrees from the body.

Usage:
    uv run python -m gaussian_avatar.models.mesh.visualize.canonical --model-type smplx
    uv run python -m gaussian_avatar.models.mesh.visualize.canonical --model-type smpl --save output.png
"""

import argparse
import sys

import numpy as np

from gaussian_avatar.models.mesh.visualize.common import (
    create_mesh,
    setup_3d_axes,
)


def visualize_with_matplotlib(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str,
    save_path: str | None = None,
) -> None:
    """Visualize mesh using matplotlib 3D scatter/trisurf.

    Args:
        vertices: Vertex positions (V, 3)
        faces: Triangle indices (F, 3)
        title: Plot title
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_faces = vertices[faces]
    collection = Poly3DCollection(
        mesh_faces,
        alpha=0.7,
        facecolor='lightblue',
        edgecolor='gray',
        linewidth=0.1,
    )
    ax.add_collection3d(collection)

    setup_3d_axes(ax, vertices)
    ax.set_title(title)
    ax.view_init(elev=0, azim=180)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def visualize_with_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str,
    save_path: str | None = None,
) -> None:
    """Visualize mesh using trimesh.

    Args:
        vertices: Vertex positions (V, 3)
        faces: Triangle indices (F, 3)
        title: Plot title (shown in window title)
        save_path: If provided, export mesh to this path
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if save_path:
        if save_path.endswith('.png'):
            scene = mesh.scene()
            scene.set_camera(angles=[0, np.pi, 0], distance=2.5)
            png = scene.save_image(resolution=[1024, 1024])
            with open(save_path, 'wb') as f:
                f.write(png)
            print(f"Saved image to {save_path}")
        else:
            mesh.export(save_path)
            print(f"Exported mesh to {save_path}")
    else:
        print(f"Showing: {title}")
        print("Use mouse to rotate, scroll to zoom, press 'q' to quit")
        mesh.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize canonical A-pose mesh"
    )
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
        "--backend",
        choices=["matplotlib", "trimesh"],
        default="matplotlib",
        help="Visualization backend (default: matplotlib)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save visualization to file instead of showing",
    )

    args = parser.parse_args()

    print(f"Loading {args.model_type.upper()} model ({args.gender})...")
    try:
        mesh = create_mesh(args.model_type, args.gender)
    except Exception as e:
        print(f"Error loading model: {e}")
        if args.model_type == "smpl":
            print("\nNote: SMPL requires chumpy which is incompatible with Python 3.12+")
            print("Try using --model-type smplx instead")
        sys.exit(1)

    vertices = mesh.get_canonical_vertices().numpy()
    faces = mesh.get_faces().numpy()

    title = f"{args.model_type.upper()} Canonical A-Pose ({args.gender})\n"
    title += f"Vertices: {mesh.num_vertices:,}, Faces: {faces.shape[0]:,}, Joints: {mesh.num_joints}"

    print(f"Mesh loaded: {mesh.num_vertices:,} vertices, {faces.shape[0]:,} faces")

    if args.backend == "matplotlib":
        visualize_with_matplotlib(vertices, faces, title, args.save)
    else:
        visualize_with_trimesh(vertices, faces, title, args.save)


if __name__ == "__main__":
    main()
