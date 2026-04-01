#!/usr/bin/env python3
"""Animate mesh through sample poses to verify smooth deformation.

This script loads poses from PeopleSnapshot dataset or generates synthetic
poses to visualize mesh deformation and verify correct LBS implementation.

Usage:
    # Generate synthetic animation (bending/rotating joints)
    uv run python -m gaussian_avatar.models.mesh.visualize.poses --model-type smplx --synthetic

    # Load poses from PeopleSnapshot dataset
    uv run python -m gaussian_avatar.models.mesh.visualize.poses --model-type smplx --dataset male-1-casual

    # Save animation to video
    uv run python -m gaussian_avatar.models.mesh.visualize.poses --synthetic --save output.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gaussian_avatar.models.mesh.smpl import SMPLMesh
from gaussian_avatar.models.mesh.visualize.common import (
    create_mesh,
    compute_mesh_bounds,
    DATASET_ROOT,
)


def generate_synthetic_poses(
    num_frames: int = 60,
    pose_dims: int = 66,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic poses for animation.

    Creates a sequence of poses that smoothly animate various joints
    to test deformation quality.

    Args:
        num_frames: Number of frames to generate
        pose_dims: Pose dimensionality (66 for SMPL-X, 72 for SMPL)

    Returns:
        Tuple of (poses, shapes) tensors
    """
    poses = torch.zeros(num_frames, pose_dims)
    shapes = torch.zeros(num_frames, 10)

    t = torch.linspace(0, 2 * np.pi, num_frames)

    # Animate spine (joints 1-3 in body_pose, dims 3-12)
    # Spine forward/backward bend
    poses[:, 3] = 0.3 * torch.sin(t)  # Spine1 X rotation

    # Animate shoulders (joints 15-16 in SMPL-X body_pose)
    if pose_dims >= 66:  # SMPL-X
        # Left shoulder (body_pose joint 15)
        left_shoulder_start = 3 + 15 * 3
        if left_shoulder_start + 2 < pose_dims:
            poses[:, left_shoulder_start] = 0.3 * torch.sin(t)  # X rotation

        # Right shoulder (body_pose joint 16)
        right_shoulder_start = 3 + 16 * 3
        if right_shoulder_start + 2 < pose_dims:
            poses[:, right_shoulder_start] = -0.3 * torch.sin(t)  # X rotation (opposite)

    # Animate hips (body_pose joints 0-1)
    # Left hip
    poses[:, 3 + 0 * 3] = 0.2 * torch.sin(t + np.pi / 4)  # X rotation
    # Right hip
    poses[:, 3 + 1 * 3] = 0.2 * torch.sin(t - np.pi / 4)  # X rotation

    # Slight global rotation
    poses[:, 1] = 0.1 * torch.sin(t / 2)  # Global Y rotation

    return poses, shapes


def load_peoplesnapshot_poses(
    subject: str,
    max_frames: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load poses from PeopleSnapshot dataset.

    Args:
        subject: Subject identifier (e.g., 'male-1-casual')
        max_frames: Maximum number of frames to load (None for all)

    Returns:
        Tuple of (poses, shapes) tensors
        Poses are in SMPL format (72 dims)

    Requires:
        h5py package must be installed: pip install h5py
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to load PeopleSnapshot poses. "
            "Install with: pip install h5py"
        )

    subject_path = Path(DATASET_ROOT) / subject
    poses_file = subject_path / "reconstructed_poses.hdf5"

    if not poses_file.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_file}")

    with h5py.File(poses_file, "r") as f:
        poses_np = f["pose"][:]  # (T, 72)
        betas_np = f["betas"][:]  # (10,)
        trans_np = f["trans"][:]  # (T, 3) - translation, we'll ignore this

    # Limit frames if requested
    if max_frames is not None:
        poses_np = poses_np[:max_frames]

    poses = torch.from_numpy(poses_np).float()
    # Repeat betas for all frames
    shapes = torch.from_numpy(betas_np).float().unsqueeze(0).expand(poses.shape[0], -1)

    return poses, shapes


def convert_smpl_to_smplx_pose(smpl_pose: torch.Tensor) -> torch.Tensor:
    """Convert SMPL pose (72 dims) to SMPL-X pose (66 dims).

    SMPL-X body_pose has 21 joints instead of 23, so we take a subset.

    Args:
        smpl_pose: SMPL pose tensor (T, 72) or (72,)

    Returns:
        SMPL-X compatible pose tensor (T, 66) or (66,)
    """
    is_batched = smpl_pose.dim() == 2

    if not is_batched:
        smpl_pose = smpl_pose.unsqueeze(0)

    # SMPL: global_orient (3) + body_pose (69 = 23 joints * 3)
    # SMPL-X: global_orient (3) + body_pose (63 = 21 joints * 3)
    global_orient = smpl_pose[:, :3]
    body_pose_smpl = smpl_pose[:, 3:72]  # 69 dims

    # Take first 21 joints of body pose (skip last 2 joints)
    body_pose_smplx = body_pose_smpl[:, :63]

    result = torch.cat([global_orient, body_pose_smplx], dim=1)

    if not is_batched:
        result = result.squeeze(0)

    return result


def animate_with_matplotlib(
    mesh: SMPLMesh,
    poses: torch.Tensor,
    shapes: torch.Tensor,
    save_path: Optional[str] = None,
    fps: int = 30,
) -> None:
    """Create animated visualization using matplotlib.

    Args:
        mesh: SMPLMesh instance
        poses: Pose sequence (T, pose_dims)
        shapes: Shape sequence (T, num_betas)
        save_path: If provided, save animation to this path
        fps: Frames per second for animation
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    faces = mesh.get_faces().numpy()
    num_frames = poses.shape[0]

    print(f"Computing {num_frames} frames...")
    all_vertices = []
    for i in range(num_frames):
        verts = mesh.forward(poses[i], shapes[i]).numpy()
        all_vertices.append(verts)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    all_verts = np.stack(all_vertices)
    max_range, mid_x, mid_y, mid_z = compute_mesh_bounds(all_verts)
    max_range *= 1.1  # Add margin

    def init():
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=0, azim=180)
        return []

    collection = None

    def update(frame):
        nonlocal collection

        if collection is not None:
            collection.remove()

        vertices = all_vertices[frame]
        mesh_faces = vertices[faces]

        collection = Poly3DCollection(
            mesh_faces,
            alpha=0.7,
            facecolor='lightblue',
            edgecolor='gray',
            linewidth=0.05,
        )
        ax.add_collection3d(collection)
        ax.set_title(f"{mesh.model_type.upper()} Pose Animation - Frame {frame + 1}/{num_frames}")
        return [collection]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
        elif save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps, dpi=80)
        else:
            anim.save(save_path, fps=fps)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Animate mesh through poses"
    )
    parser.add_argument(
        "--model-type",
        choices=["smpl", "smplx"],
        default="smplx",
        help="Model type to use (default: smplx)",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Model gender (default: neutral)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic poses instead of dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="PeopleSnapshot subject to load (e.g., 'male-1-casual')",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum frames to animate (default: 120)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Animation framerate (default: 30)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save animation to file (e.g., output.mp4 or output.gif)",
    )

    args = parser.parse_args()

    if not args.synthetic and args.dataset is None:
        parser.error("Either --synthetic or --dataset must be specified")

    print(f"Loading {args.model_type.upper()} model ({args.gender})...")
    try:
        mesh = create_mesh(args.model_type, args.gender)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    pose_dims = 72 if args.model_type == "smpl" else 66

    if args.synthetic:
        print(f"Generating synthetic animation ({args.max_frames} frames)...")
        poses, shapes = generate_synthetic_poses(args.max_frames, pose_dims)
    else:
        print(f"Loading poses from PeopleSnapshot/{args.dataset}...")
        try:
            poses, shapes = load_peoplesnapshot_poses(args.dataset, args.max_frames)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Convert SMPL poses to SMPL-X if needed
        if args.model_type == "smplx" and poses.shape[1] == 72:
            print("Converting SMPL poses (72 dim) to SMPL-X format (66 dim)...")
            poses = convert_smpl_to_smplx_pose(poses)

    print(f"Animating {poses.shape[0]} frames...")

    animate_with_matplotlib(mesh, poses, shapes, args.save, args.fps)


if __name__ == "__main__":
    main()
