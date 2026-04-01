#!/usr/bin/env python3
"""Visualize triangle normals from centroids.

This script draws arrows from triangle centroids in the normal direction,
verifying that the local frame computation produces correct triangle normals.

Usage:
    uv run python -m gaussian_avatar.models.mesh.visualize.normals --model-type smplx
    uv run python -m gaussian_avatar.models.mesh.visualize.normals --model-type smplx --sample-ratio 0.01
    uv run python -m gaussian_avatar.models.mesh.visualize.normals --model-type smplx --save normals.png
"""

import argparse
import sys

import numpy as np

from gaussian_avatar.models.mesh.utils import (
    get_triangle_centroids,
    get_triangle_local_frames,
)
from gaussian_avatar.models.mesh.visualize.common import (
    create_mesh,
    setup_3d_axes,
)


def visualize_normals_matplotlib(
    vertices: np.ndarray,
    faces: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    sample_ratio: float = 0.01,
    normal_scale: float = 0.02,
    title: str = "Triangle Normals",
    save_path: str | None = None,
) -> None:
    """Visualize mesh with normal arrows using matplotlib.

    Args:
        vertices: Vertex positions (V, 3)
        faces: Triangle indices (F, 3)
        centroids: Triangle centroids (F, 3)
        normals: Triangle normals (F, 3)
        sample_ratio: Fraction of triangles to show normals for
        normal_scale: Scale factor for normal arrow length
        title: Plot title
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_faces = vertices[faces]
    collection = Poly3DCollection(
        mesh_faces,
        alpha=0.3,
        facecolor='lightgray',
        edgecolor='darkgray',
        linewidth=0.05,
    )
    ax.add_collection3d(collection)

    num_faces = faces.shape[0]
    num_samples = max(1, int(num_faces * sample_ratio))
    sample_indices = np.random.choice(num_faces, num_samples, replace=False)

    sampled_centroids = centroids[sample_indices]
    sampled_normals = normals[sample_indices]

    ax.quiver(
        sampled_centroids[:, 0],
        sampled_centroids[:, 1],
        sampled_centroids[:, 2],
        sampled_normals[:, 0] * normal_scale,
        sampled_normals[:, 1] * normal_scale,
        sampled_normals[:, 2] * normal_scale,
        color='red',
        arrow_length_ratio=0.3,
        linewidth=0.8,
        alpha=0.8,
    )

    setup_3d_axes(ax, vertices)
    ax.set_title(f"{title}\n({num_samples} normals shown, {sample_ratio*100:.1f}% sample)")
    ax.view_init(elev=20, azim=135)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def visualize_local_frames_matplotlib(
    vertices: np.ndarray,
    faces: np.ndarray,
    centroids: np.ndarray,
    frames: np.ndarray,
    sample_ratio: float = 0.005,
    axis_scale: float = 0.02,
    title: str = "Local Frames",
    save_path: str | None = None,
) -> None:
    """Visualize full local coordinate frames (X, Y, Z axes).

    Args:
        vertices: Vertex positions (V, 3)
        faces: Triangle indices (F, 3)
        centroids: Triangle centroids (F, 3)
        frames: Local frame rotation matrices (F, 3, 3)
        sample_ratio: Fraction of triangles to show frames for
        axis_scale: Scale factor for axis arrow length
        title: Plot title
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_faces = vertices[faces]
    collection = Poly3DCollection(
        mesh_faces,
        alpha=0.2,
        facecolor='lightgray',
        edgecolor='darkgray',
        linewidth=0.02,
    )
    ax.add_collection3d(collection)

    num_faces = faces.shape[0]
    num_samples = max(1, int(num_faces * sample_ratio))
    sample_indices = np.random.choice(num_faces, num_samples, replace=False)

    sampled_centroids = centroids[sample_indices]
    sampled_frames = frames[sample_indices]

    # Draw X, Y, Z axes for each sampled frame
    colors = ['red', 'green', 'blue']  # X=red, Y=green, Z=blue
    axis_names = ['X', 'Y', 'Z']

    for axis_idx, (color, name) in enumerate(zip(colors, axis_names)):
        axis_dirs = sampled_frames[:, :, axis_idx]  # (N, 3)

        ax.quiver(
            sampled_centroids[:, 0],
            sampled_centroids[:, 1],
            sampled_centroids[:, 2],
            axis_dirs[:, 0] * axis_scale,
            axis_dirs[:, 1] * axis_scale,
            axis_dirs[:, 2] * axis_scale,
            color=color,
            arrow_length_ratio=0.3,
            linewidth=0.6,
            alpha=0.8,
            label=f'{name}-axis',
        )

    setup_3d_axes(ax, vertices)
    ax.set_title(f"{title}\n({num_samples} frames shown)")
    ax.legend(loc='upper left')

    ax.view_init(elev=20, azim=135)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize triangle normals and local frames"
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
        "--sample-ratio",
        type=float,
        default=0.01,
        help="Fraction of triangles to show normals for (default: 0.01)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.02,
        help="Scale factor for arrow length (default: 0.02)",
    )
    parser.add_argument(
        "--mode",
        choices=["normals", "frames"],
        default="normals",
        help="Visualization mode: 'normals' for Z-axis only, 'frames' for all axes",
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
        sys.exit(1)

    vertices = mesh.get_canonical_vertices()
    faces = mesh.get_faces()

    print(f"Mesh loaded: {mesh.num_vertices:,} vertices, {faces.shape[0]:,} faces")

    print("Computing triangle properties...")
    centroids = get_triangle_centroids(vertices, faces)
    frames = get_triangle_local_frames(vertices, faces)

    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    centroids_np = centroids.numpy()
    frames_np = frames.numpy()

    # Extract normals (Z-axis, column 2)
    normals_np = frames_np[:, :, 2]

    # Verify normals are unit length
    norms = np.linalg.norm(normals_np, axis=1)
    print(f"Normal lengths: min={norms.min():.6f}, max={norms.max():.6f}")

    title = f"{args.model_type.upper()} Triangle {'Normals' if args.mode == 'normals' else 'Local Frames'}"

    if args.mode == "normals":
        visualize_normals_matplotlib(
            vertices_np,
            faces_np,
            centroids_np,
            normals_np,
            sample_ratio=args.sample_ratio,
            normal_scale=args.scale,
            title=title,
            save_path=args.save,
        )
    else:
        visualize_local_frames_matplotlib(
            vertices_np,
            faces_np,
            centroids_np,
            frames_np,
            sample_ratio=args.sample_ratio,
            axis_scale=args.scale,
            title=title,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
