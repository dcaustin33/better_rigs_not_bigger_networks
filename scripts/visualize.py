#!/usr/bin/env python
"""Visualization script for trained Gaussian Avatar models.

This script loads a trained checkpoint and renders visualizations, including:
- Specific poses from the dataset
- Video sequences through the dataset
- Novel poses with custom joint angles

Example usage:
    # Render specific frames from the dataset
    uv run python scripts/visualize.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual \\
        --frames 0 10 20 30

    # Render a video sequence through all test frames
    uv run python scripts/visualize.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual \\
        --video --output-dir output/videos

    # Render novel poses (A-pose and T-pose)
    uv run python scripts/visualize.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --novel-poses a-pose t-pose \\
        --output-dir output/novel_poses
"""

import argparse
from pathlib import Path

import torch
from torch import Tensor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Visualize trained Gaussian Avatar model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to people_snapshot_public dataset (required for dataset poses)",
    )
    data_group.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Subject names to visualize",
    )
    data_group.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Dataset split to use",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        default="mesh_models",
        help="Path to SMPL/SMPL-X model files root directory",
    )
    model_group.add_argument(
        "--model-type",
        type=str,
        default="smpl",
        choices=["smpl", "smplx"],
        help="Model type (smpl or smplx)",
    )
    model_group.add_argument(
        "--gender",
        type=str,
        default="male",
        choices=["male", "female", "neutral"],
        help="SMPL gender",
    )

    render_group = parser.add_argument_group("Rendering Mode")
    render_group.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="Specific frame indices to render from the dataset",
    )
    render_group.add_argument(
        "--video",
        action="store_true",
        help="Render all frames as a video sequence",
    )
    render_group.add_argument(
        "--novel-poses",
        type=str,
        nargs="+",
        default=None,
        choices=["a-pose", "t-pose", "arms-up", "arms-down", "squat"],
        help="Render novel predefined poses",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="output/visualize",
        help="Directory for visualization outputs",
    )
    output_group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Output image size (height width)",
    )
    output_group.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video output",
    )
    output_group.add_argument(
        "--save-alpha",
        action="store_true",
        help="Also save alpha/opacity channel",
    )

    return parser.parse_args()


def create_novel_pose(
    pose_name: str,
    model_type: str = "smpl",
) -> tuple[Tensor, Tensor]:
    """Create a predefined novel pose.

    Args:
        pose_name: Name of the pose ('a-pose', 't-pose', etc.)
        model_type: 'smpl' or 'smplx' to determine pose dimension

    Returns:
        Tuple of (pose, betas) tensors
    """
    if model_type == "smpl":
        pose_dim = 72
        left_shoulder_idx = 48
        right_shoulder_idx = 51
        left_elbow_idx = 55
        right_elbow_idx = 58
        left_knee_idx = 12
        right_knee_idx = 15
    else:
        pose_dim = 66
        # SMPL-X: 3 global orient + 21 body joints * 3
        # Body joint indices need +3 offset for global orientation
        left_shoulder_idx = 48   # 3 + 15*3
        right_shoulder_idx = 51  # 3 + 16*3
        left_elbow_idx = 54      # 3 + 17*3
        right_elbow_idx = 57     # 3 + 18*3
        left_knee_idx = 15       # 3 + 4*3
        right_knee_idx = 18      # 3 + 5*3

    pose = torch.zeros(pose_dim)
    betas = torch.zeros(10)

    if pose_name == "a-pose":
        # A-pose: arms at ~45 degrees (default canonical pose)
        pose[left_shoulder_idx + 2] = 0.7  # z-rotation for arm angle
        pose[right_shoulder_idx + 2] = -0.7
    elif pose_name == "t-pose":
        # T-pose: arms straight out horizontally
        pose[left_shoulder_idx + 2] = 1.57  # ~90 degrees
        pose[right_shoulder_idx + 2] = -1.57
    elif pose_name == "arms-up":
        # Arms raised above head
        pose[left_shoulder_idx + 2] = 2.5
        pose[right_shoulder_idx + 2] = -2.5
        pose[left_shoulder_idx + 1] = 0.3
        pose[right_shoulder_idx + 1] = -0.3
    elif pose_name == "arms-down":
        # Arms down by sides
        pose[left_shoulder_idx + 2] = 0.0
        pose[right_shoulder_idx + 2] = 0.0
    elif pose_name == "squat":
        # Squatting pose
        pose[left_knee_idx] = 1.2
        pose[right_knee_idx] = 1.2
        pose[left_shoulder_idx + 2] = 0.5
        pose[right_shoulder_idx + 2] = -0.5

    return pose, betas


def create_default_camera(
    height: int,
    width: int,
    device: torch.device,
) -> "Camera":
    """Create a default camera for novel pose rendering.

    Args:
        height: Image height
        width: Image width
        device: Torch device

    Returns:
        Camera instance
    """
    from gaussian_avatar.rendering import Camera

    # Create intrinsic matrix (approximate values for a typical camera)
    fx = fy = max(height, width) * 1.5
    cx = width / 2
    cy = height / 2

    intrinsic = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )

    # Create extrinsic matrix (camera looking at origin from z=3)
    extrinsic = torch.eye(4, dtype=torch.float32, device=device)
    extrinsic[2, 3] = 3.0  # Camera at z=3

    return Camera.from_dataset_sample(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        height=height,
        width=width,
    )


def save_tensor_as_image(tensor: Tensor, path: Path) -> None:
    """Save a tensor as an image file.

    Args:
        tensor: Image tensor (C, H, W) or (H, W) in [0, 1]
        path: Output file path
    """
    from PIL import Image

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    img_np = (tensor.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()

    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)

    Image.fromarray(img_np).save(path)


def render_frame(
    avatar,
    renderer,
    pose: Tensor,
    betas: Tensor,
    camera,
    device: torch.device,
    trans: Tensor | None = None,
) -> dict[str, Tensor]:
    """Render a single frame.

    Args:
        avatar: GaussianAvatar model
        renderer: GaussianRenderer or MockRenderer
        pose: Pose tensor
        betas: Shape coefficients tensor
        camera: Camera instance
        device: Torch device
        trans: Optional root translation tensor

    Returns:
        Dictionary with 'rgb' and 'alpha' tensors
    """
    pose = pose.to(device)
    betas = betas.to(device)

    with torch.no_grad():
        avatar_output = avatar(pose, betas)
        if trans is not None:
            avatar_output["means"] = avatar_output["means"] + trans.to(device)

        render_output = renderer(
            means=avatar_output["means"],
            quats=avatar_output["quats"],
            scales=avatar_output["scales"],
            colors=avatar_output["colors"],
            opacities=avatar_output["opacities"],
            camera=camera,
        )

    return {
        "rgb": render_output["rgb"],
        "alpha": render_output.get("alpha"),
    }


def render_dataset_frames(
    avatar,
    renderer,
    dataset,
    frame_indices: list[int],
    output_dir: Path,
    save_alpha: bool,
    device: torch.device,
) -> list[Path]:
    """Render specific frames from a dataset.

    Args:
        avatar: GaussianAvatar model
        renderer: GaussianRenderer or MockRenderer
        dataset: PeopleSnapshotDataset
        frame_indices: List of frame indices to render
        output_dir: Output directory
        save_alpha: Whether to save alpha channel
        device: Torch device

    Returns:
        List of saved image paths
    """
    from gaussian_avatar.rendering import Camera

    saved_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in frame_indices:
        if idx >= len(dataset):
            print(f"Warning: Frame index {idx} out of range, skipping")
            continue

        sample = dataset[idx]

        H, W = sample["image"].shape[1:]
        camera = Camera.from_dataset_sample(
            intrinsic=sample["intrinsic"].to(device),
            extrinsic=sample["extrinsic"].to(device),
            height=H,
            width=W,
        )

        result = render_frame(
            avatar, renderer, sample["pose"], sample["betas"], camera, device,
            trans=sample.get("trans"),
        )

        subject = sample.get("subject", "unknown")
        frame_idx = sample.get("frame_idx", idx)
        prefix = f"{subject}_frame_{frame_idx:04d}"

        rgb_path = output_dir / f"{prefix}_render.png"
        save_tensor_as_image(result["rgb"], rgb_path)
        saved_paths.append(rgb_path)
        print(f"Saved: {rgb_path}")

        if save_alpha and result["alpha"] is not None:
            alpha_path = output_dir / f"{prefix}_alpha.png"
            save_tensor_as_image(result["alpha"], alpha_path)
            saved_paths.append(alpha_path)

        gt_path = output_dir / f"{prefix}_gt.png"
        save_tensor_as_image(sample["image"], gt_path)
        saved_paths.append(gt_path)

    return saved_paths


def render_video_sequence(
    avatar,
    renderer,
    dataset,
    output_dir: Path,
    fps: int,
    save_alpha: bool,
    device: torch.device,
) -> Path:
    """Render all dataset frames as a video.

    Args:
        avatar: GaussianAvatar model
        renderer: GaussianRenderer or MockRenderer
        dataset: PeopleSnapshotDataset
        output_dir: Output directory
        fps: Frames per second
        save_alpha: Whether to save alpha video
        device: Torch device

    Returns:
        Path to saved video file
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video output. Install with: pip install imageio[ffmpeg]"
        )

    from gaussian_avatar.rendering import Camera

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_rgb = []
    frames_alpha = [] if save_alpha else None

    print(f"Rendering {len(dataset)} frames...")
    for idx in range(len(dataset)):
        sample = dataset[idx]

        H, W = sample["image"].shape[1:]
        camera = Camera.from_dataset_sample(
            intrinsic=sample["intrinsic"].to(device),
            extrinsic=sample["extrinsic"].to(device),
            height=H,
            width=W,
        )

        result = render_frame(
            avatar, renderer, sample["pose"], sample["betas"], camera, device,
            trans=sample.get("trans"),
        )

        rgb_np = (result["rgb"].clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
        frames_rgb.append(rgb_np)

        if save_alpha and result["alpha"] is not None:
            alpha_np = (
                (result["alpha"].clamp(0, 1) * 255).byte().cpu().numpy()
            )
            frames_alpha.append(alpha_np)

        if (idx + 1) % 50 == 0:
            print(f"Rendered {idx + 1}/{len(dataset)} frames")

    sample = dataset[0]
    subject = sample.get("subject", "avatar")

    video_path = output_dir / f"{subject}_render.mp4"
    imageio.mimsave(video_path, frames_rgb, fps=fps)
    print(f"Saved video: {video_path}")

    if save_alpha and frames_alpha:
        alpha_path = output_dir / f"{subject}_alpha.mp4"
        imageio.mimsave(alpha_path, frames_alpha, fps=fps)
        print(f"Saved alpha video: {alpha_path}")

    return video_path


def render_novel_poses(
    avatar,
    renderer,
    pose_names: list[str],
    output_dir: Path,
    image_size: tuple[int, int],
    model_type: str,
    save_alpha: bool,
    device: torch.device,
) -> list[Path]:
    """Render predefined novel poses.

    Args:
        avatar: GaussianAvatar model
        renderer: GaussianRenderer or MockRenderer
        pose_names: List of pose names to render
        output_dir: Output directory
        image_size: Output image size (height, width)
        model_type: 'smpl' or 'smplx'
        save_alpha: Whether to save alpha channel
        device: Torch device

    Returns:
        List of saved image paths
    """
    saved_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = image_size
    camera = create_default_camera(height, width, device)

    for pose_name in pose_names:
        pose, betas = create_novel_pose(pose_name, model_type)

        result = render_frame(avatar, renderer, pose, betas, camera, device)

        rgb_path = output_dir / f"{pose_name}_render.png"
        save_tensor_as_image(result["rgb"], rgb_path)
        saved_paths.append(rgb_path)
        print(f"Saved: {rgb_path}")

        if save_alpha and result["alpha"] is not None:
            alpha_path = output_dir / f"{pose_name}_alpha.png"
            save_tensor_as_image(result["alpha"], alpha_path)
            saved_paths.append(alpha_path)

    return saved_paths


def run_visualization(args: argparse.Namespace) -> None:
    """Run visualization based on arguments.

    Args:
        args: Parsed command line arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import components (deferred to allow --help without heavy imports)
    from gaussian_avatar.models import GaussianAvatar
    from gaussian_avatar.models.mesh import SMPLMesh
    from gaussian_avatar.rendering import (
        GaussianRenderer,
        MockRenderer,
        GSPLAT_AVAILABLE,
    )
    from gaussian_avatar.training.checkpoint import load_checkpoint

    print(f"Loading {args.model_type.upper()} mesh from: {args.model_path}")
    mesh = SMPLMesh(
        model_path=args.model_path,
        model_type=args.model_type,
        gender=args.gender,
        device=device,
    )

    print("Creating GaussianAvatar...")
    avatar = GaussianAvatar(
        mesh=mesh,
        num_gaussians=mesh.get_faces().shape[0],
    )

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_info = load_checkpoint(
        path=checkpoint_path,
        avatar=avatar,
        device=device,
    )
    iteration = checkpoint_info.get("iteration", 0)
    print(f"Loaded checkpoint from iteration {iteration}")

    if GSPLAT_AVAILABLE:
        print("Using GaussianRenderer (gsplat)")
        renderer = GaussianRenderer()
    else:
        print("WARNING: gsplat not available, using MockRenderer")
        renderer = MockRenderer(allow_forward=True)

    avatar.eval()

    if args.novel_poses:
        print(f"\nRendering novel poses: {args.novel_poses}")
        render_novel_poses(
            avatar=avatar,
            renderer=renderer,
            pose_names=args.novel_poses,
            output_dir=output_dir,
            image_size=tuple(args.image_size),
            model_type=args.model_type,
            save_alpha=args.save_alpha,
            device=device,
        )

    elif args.data_root is None:
        raise ValueError(
            "Either --data-root (for dataset poses) or --novel-poses must be provided"
        )

    else:
        from gaussian_avatar.data import PeopleSnapshotDataset

        subjects = args.subjects
        if subjects is None:
            raise ValueError("--subjects is required when using --data-root")

        print(f"\nLoading dataset from: {args.data_root}")
        print(f"Subjects: {subjects}")
        dataset = PeopleSnapshotDataset(
            data_root=args.data_root,
            subjects=subjects if len(subjects) > 1 else subjects[0],
            split=args.split,
            image_size=tuple(args.image_size),
            undistort=True,
        )
        print(f"Dataset samples: {len(dataset)}")

        if args.video:
            print("\nRendering video sequence...")
            render_video_sequence(
                avatar=avatar,
                renderer=renderer,
                dataset=dataset,
                output_dir=output_dir,
                fps=args.fps,
                save_alpha=args.save_alpha,
                device=device,
            )

        elif args.frames is not None:
            print(f"\nRendering frames: {args.frames}")
            render_dataset_frames(
                avatar=avatar,
                renderer=renderer,
                dataset=dataset,
                frame_indices=args.frames,
                output_dir=output_dir,
                save_alpha=args.save_alpha,
                device=device,
            )

        else:
            print("\nNo frames specified, rendering first 5 frames")
            render_dataset_frames(
                avatar=avatar,
                renderer=renderer,
                dataset=dataset,
                frame_indices=list(range(min(5, len(dataset)))),
                output_dir=output_dir,
                save_alpha=args.save_alpha,
                device=device,
            )

    print(f"\nVisualization complete. Outputs saved to: {output_dir}")


def main() -> None:
    """Main visualization entry point."""
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
