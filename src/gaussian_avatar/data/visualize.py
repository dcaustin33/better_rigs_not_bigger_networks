"""Visualization utilities for SMPL mesh and image data.

This module provides functions for projecting SMPL meshes onto images
and visualizing the alignment between reconstructed meshes and video frames.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from gaussian_avatar.data.people_snapshot import PeopleSnapshotDataset
from gaussian_avatar.models.mesh.smpl import SMPLMesh

def project_vertices(
    vertices: torch.Tensor,
    translation: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> np.ndarray:
    """Project 3D vertices to 2D pixel coordinates.

    Applies root translation, then world-to-camera transform via extrinsic,
    then perspective projection via intrinsic matrix.

    Args:
        vertices: (V, 3) 3D vertices in world space
        translation: (3,) root translation
        extrinsic: (4, 4) world-to-camera transform
        intrinsic: (3, 3) camera intrinsic matrix

    Returns:
        (V, 2) 2D pixel coordinates
    """
    world_verts = vertices + translation

    ones = torch.ones(world_verts.shape[0], 1, device=world_verts.device)
    world_verts_h = torch.cat([world_verts, ones], dim=1)

    # World-to-camera transform: (V, 4) @ (4, 4).T -> (V, 4)
    cam_coords = world_verts_h @ extrinsic.T

    cam_xyz = cam_coords[:, :3]

    # Perspective projection: (V, 3) @ (3, 3).T -> (V, 3)
    projected = cam_xyz @ intrinsic.T

    uv = projected[:, :2] / projected[:, 2:3]

    return uv.cpu().numpy()

def _get_mesh_edges(faces: torch.Tensor) -> np.ndarray:
    """Extract unique edges from face connectivity.

    Args:
        faces: (F, 3) triangle face indices

    Returns:
        (E, 2) array of edge vertex pairs
    """
    faces_np = faces.cpu().numpy()

    # Each triangle has 3 edges
    edges = np.vstack([
        faces_np[:, [0, 1]],
        faces_np[:, [1, 2]],
        faces_np[:, [2, 0]],
    ])

    # Sort edge vertices to ensure consistent ordering
    edges = np.sort(edges, axis=1)

    # Remove duplicates
    edges = np.unique(edges, axis=0)

    return edges

def visualize_sample(
    subject: str,
    frame_idx: int,
    data_root: str = "datasets/people_snapshot_public",
    model_path: str = "mesh_models",
    gender: Optional[str] = None,
    image_size: Tuple[int, int] = (512, 512),
    wireframe_color: str = "cyan",
    wireframe_alpha: float = 0.5,
    wireframe_linewidth: float = 0.3,
    show_original: bool = True,
    show_mask: bool = True,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Visualize SMPL mesh projection overlaid on dataset image.

    Creates a multi-panel visualization showing the mesh wireframe
    overlaid on the image, the original image, and the segmentation mask.

    Args:
        subject: Subject name (e.g., "male-3-casual")
        frame_idx: Frame index to visualize
        data_root: Path to PeopleSnapshot dataset root
        model_path: Path to SMPL model files
        gender: SMPL model gender ("male" or "female"). If None, inferred from subject name.
        image_size: Target image size (H, W)
        wireframe_color: Color for mesh wireframe
        wireframe_alpha: Transparency for wireframe (0-1)
        wireframe_linewidth: Line width for wireframe edges
        show_original: Whether to show original image alongside overlay
        show_mask: Whether to show the segmentation mask
        save_path: If provided, save figure to this path
        figsize: Figure size in inches. If None, auto-calculated based on columns.

    Returns:
        matplotlib Figure object
    """
    dataset = PeopleSnapshotDataset(
        data_root=data_root,
        subjects=subject,
        split="all",
        image_size=image_size,
        undistort=True,
    )

    sample = None
    for idx in range(len(dataset)):
        s = dataset[idx]
        if s["frame_idx"] == frame_idx:
            sample = s
            break

    if sample is None:
        raise ValueError(f"Frame {frame_idx} not found for subject {subject}")

    return visualize_sample_from_data(
        sample=sample,
        model_path=model_path,
        gender=gender,
        wireframe_color=wireframe_color,
        wireframe_alpha=wireframe_alpha,
        wireframe_linewidth=wireframe_linewidth,
        show_original=show_original,
        show_mask=show_mask,
        save_path=save_path,
        figsize=figsize,
    )

def visualize_sample_by_index(
    idx: int,
    dataset: PeopleSnapshotDataset,
    model_path: str = "mesh_models",
    gender: Optional[str] = None,
    wireframe_color: str = "cyan",
    wireframe_alpha: float = 0.5,
    wireframe_linewidth: float = 0.3,
    show_original: bool = True,
    show_mask: bool = True,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Visualize SMPL mesh projection for a dataset sample by index.

    Convenience wrapper that takes a dataset object and index directly.

    Args:
        idx: Dataset sample index
        dataset: PeopleSnapshotDataset instance
        model_path: Path to SMPL model files
        gender: SMPL model gender ("male" or "female"). If None, inferred from subject name.
        wireframe_color: Color for mesh wireframe
        wireframe_alpha: Transparency for wireframe (0-1)
        wireframe_linewidth: Line width for wireframe edges
        show_original: Whether to show original image alongside overlay
        show_mask: Whether to show the segmentation mask
        save_path: If provided, save figure to this path
        figsize: Figure size in inches. If None, auto-calculated based on columns.

    Returns:
        matplotlib Figure object
    """
    sample = dataset[idx]

    return visualize_sample_from_data(
        sample=sample,
        model_path=model_path,
        gender=gender,
        wireframe_color=wireframe_color,
        wireframe_alpha=wireframe_alpha,
        wireframe_linewidth=wireframe_linewidth,
        show_original=show_original,
        show_mask=show_mask,
        save_path=save_path,
        figsize=figsize,
    )

def _infer_gender(subject: str) -> str:
    """Infer gender from subject name.

    Args:
        subject: Subject name (e.g., "male-3-casual", "female-1-sport")

    Returns:
        Gender string ("male" or "female")
    """
    if subject.startswith("male"):
        return "male"
    elif subject.startswith("female"):
        return "female"
    else:
        return "male"  # Default fallback

def visualize_sample_from_data(
    sample: dict,
    model_path: str = "mesh_models",
    gender: Optional[str] = None,
    wireframe_color: str = "cyan",
    wireframe_alpha: float = 0.5,
    wireframe_linewidth: float = 0.3,
    show_original: bool = True,
    show_mask: bool = True,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """Visualize SMPL mesh projection from a sample dictionary.

    Core visualization function that renders mesh wireframe overlaid on image.

    Args:
        sample: Dataset sample dictionary with keys: image, mask, pose, betas, trans,
                intrinsic, extrinsic, frame_idx, subject
        model_path: Path to SMPL model files
        gender: SMPL model gender ("male" or "female"). If None, inferred from subject name.
        wireframe_color: Color for mesh wireframe
        wireframe_alpha: Transparency for wireframe (0-1)
        wireframe_linewidth: Line width for wireframe edges
        show_original: Whether to show original image alongside overlay
        show_mask: Whether to show the segmentation mask
        save_path: If provided, save figure to this path
        figsize: Figure size in inches. If None, auto-calculated based on columns.

    Returns:
        matplotlib Figure object
    """
    image = sample["image"]  # (3, H, W)
    mask = sample.get("mask")  # (H, W) or None
    pose = sample["pose"]  # (72,)
    betas = sample["betas"]  # (10,)
    trans = sample["trans"]  # (3,)
    intrinsic = sample["intrinsic"]  # (3, 3)
    extrinsic = sample["extrinsic"]  # (4, 4)
    frame_idx = sample["frame_idx"]
    subject = sample["subject"]

    if gender is None:
        gender = _infer_gender(subject)

    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = np.transpose(image, (1, 2, 0))

    mask_np = None
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

    mesh = SMPLMesh(
        model_path=model_path,
        model_type="smpl",
        gender=gender,
    )

    vertices = mesh.forward(pose, betas)
    faces = mesh.get_faces()

    uv = project_vertices(vertices, trans, extrinsic, intrinsic)

    edges = _get_mesh_edges(faces)

    edge_coords = uv[edges]  # (E, 2, 2)

    ncols = 1  # mesh overlay always shown
    if show_original:
        ncols += 1
    if show_mask and mask_np is not None:
        ncols += 1

    if figsize is None:
        figsize = (6 * ncols, 6)

    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    if ncols == 1:
        axes = [axes]

    col_idx = 0

    ax = axes[col_idx]
    ax.imshow(image_np)

    # Draw wireframe using LineCollection for efficiency
    lines = LineCollection(
        edge_coords,
        colors=wireframe_color,
        alpha=wireframe_alpha,
        linewidths=wireframe_linewidth,
    )
    ax.add_collection(lines)

    ax.set_xlim(0, image_np.shape[1])
    ax.set_ylim(image_np.shape[0], 0)  # Flip y-axis for image coordinates
    ax.set_title(f"{subject} - Frame {frame_idx} (mesh overlay)")
    ax.axis("off")
    col_idx += 1

    if show_original:
        ax = axes[col_idx]
        ax.imshow(image_np)
        ax.set_title(f"{subject} - Frame {frame_idx} (original)")
        ax.axis("off")
        col_idx += 1

    if show_mask and mask_np is not None:
        ax = axes[col_idx]
        ax.imshow(mask_np, cmap="gray")
        ax.set_title(f"{subject} - Frame {frame_idx} (mask)")
        ax.axis("off")
        col_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

def main():
    """CLI entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize SMPL mesh projection on PeopleSnapshot images"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="male-3-casual",
        help="Subject name (default: male-3-casual)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=10,
        help="Frame index (default: 10)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/people_snapshot_public",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mesh_models",
        help="Path to SMPL model files",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Target image size (H W)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save figure to this path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figure (useful with --save)",
    )

    args = parser.parse_args()

    fig = visualize_sample(
        subject=args.subject,
        frame_idx=args.frame,
        data_root=args.data_root,
        model_path=args.model_path,
        image_size=tuple(args.image_size),
        save_path=args.save,
    )

    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
