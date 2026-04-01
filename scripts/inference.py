#!/usr/bin/env python
"""Inference script for trained Gaussian Avatar models.

Loads a trained checkpoint and renders frames from pose parameters,
with device selection for CUDA (gsplat), MPS (opensplat-metal), or CPU
(MockRenderer placeholder). Supports single-frame and batch directory modes.

Example usage:
    # Single frame
    uv run python scripts/inference.py \
        --checkpoint output/run17/final.pt \
        --pose-file datasets/mhr_example/000001.npz \
        --device mps --output output/inference.png

    # Batch directory — renders all .npz files, outputs renders/ and comparisons/
    uv run python scripts/inference.py \
        --checkpoint output/run17/final.pt \
        --pose-dir datasets/mhr_example/ \
        --output output/batch_results/

    # Batch with separate image directory
    uv run python scripts/inference.py \
        --checkpoint output/run17/final.pt \
        --pose-dir poses/ --image-dir images/ \
        --output output/batch_results/
"""

from __future__ import annotations

import argparse
import math
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch

# 0th-order SH basis constant: 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479177387814

# Default model paths per model type (where to find mesh model files)
_DEFAULT_MODEL_PATHS = {
    "mhr": "mesh_models/mhr/assets",
    "smpl": "mesh_models/smpl",
    "smplx": "mesh_models/models/smplx",
}


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render a Gaussian Avatar from a trained checkpoint and pose file(s)."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to trained .pt checkpoint",
    )

    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument(
        "--pose-file", type=str,
        help="Path to single .npz pose file",
    )
    pose_group.add_argument(
        "--pose-dir", type=str,
        help="Directory of .npz pose files (batch mode). Each .npz is matched to an "
             "image with the same stem name for side-by-side output.",
    )

    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory containing source images (batch mode). Defaults to --pose-dir. "
             "Images are matched to pose files by stem name (e.g. 000001.npz <-> 000001.png).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Rendering device (default: auto-detect best available)",
    )
    parser.add_argument(
        "--output", type=str, default="output/inference.png",
        help="Output path: image file for single mode, directory for batch mode",
    )
    parser.add_argument(
        "--image-width", type=int, default=512,
        help="Render width in pixels (default: 512)",
    )
    parser.add_argument(
        "--image-height", type=int, default=512,
        help="Render height in pixels (default: 512)",
    )
    parser.add_argument(
        "--focal-length", type=float, default=None,
        help="Override focal length in pixels (default: estimated from mesh vertices)",
    )
    parser.add_argument(
        "--camera-position", type=str, default=None,
        help="Camera position as X,Y,Z in mesh space (cm, Y-up). Default: from pred_cam_t in pose file",
    )
    parser.add_argument(
        "--background", type=str, default="1,1,1",
        help="Background color as R,G,B floats (default: 1,1,1 white)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override path to mesh model files (default: use checkpoint config or 'mesh_models')",
    )
    parser.add_argument(
        "--use-pose-betas", action="store_true",
        help="Use betas from the pose file instead of the training checkpoint betas",
    )
    parser.add_argument(
        "--max-stretch", type=float, default=None,
        help="Override max triangle stretch factor (default: use checkpoint value). "
             "e.g. run27 uses 1.2",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Log Gaussian scale statistics before and after posing (per-dimension)",
    )
    parser.add_argument(
        "--save-ply", action="store_true",
        help="Save posed Gaussians as standard 3DGS-compatible .ply files alongside renders. "
             "Compatible with SIBR viewer, SuperSplat, Luma, antimatter15/splat, etc.",
    )

    parser.add_argument(
        "--interpolate", action="store_true",
        help="Render interpolation from canonical rest pose to target pose(s)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=60,
        help="Number of interpolation frames from canonical to target (default: 60)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Video frame rate (default: 30)",
    )
    parser.add_argument(
        "--easing", type=str, default="smoothstep",
        choices=["linear", "smoothstep", "cosine"],
        help="Easing function for interpolation (default: smoothstep)",
    )
    parser.add_argument(
        "--hold-frames", type=int, default=15,
        help="Frames to hold on the final pose after interpolation (default: 15)",
    )
    parser.add_argument(
        "--from-pose", type=str, default="datasets/mhr_example/000001.npz",
        help="Starting pose file for pose-to-pose interpolation. "
             "Overrides canonical rest pose when file exists. "
             "(default: datasets/mhr_example/000001.npz)",
    )
    return parser.parse_args()


def find_matching_image(stem: str, image_dir: Path) -> Path | None:
    """Find an image file in image_dir whose stem matches the given name.

    Args:
        stem: File stem to match (e.g. '000001')
        image_dir: Directory to search

    Returns:
        Path to the matching image, or None if not found
    """
    for ext in _IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def auto_detect_device() -> str:
    """Detect the best available rendering device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pose(pose_path: str) -> dict[str, np.ndarray]:
    """Load pose parameters from an NPZ file.

    Args:
        pose_path: Path to .npz file

    Returns:
        Dictionary with 'pose', 'betas', 'trans' numpy arrays and 'raw' NPZ
    """
    data = np.load(pose_path, allow_pickle=True)

    pose = np.array(data["mhr_model_params"], dtype=np.float32)
    betas = np.array(data["shape_params"], dtype=np.float32)
    cam_t = np.array(data["pred_cam_t"], dtype=np.float32)

    # Convert pred_cam_t from camera space (m, Y-down) to mesh space (cm, Y-up)
    trans = cam_t * np.array([100.0, -100.0, -100.0], dtype=np.float32)

    return {"pose": pose, "betas": betas, "trans": trans, "cam_t": cam_t, "raw": data}


def estimate_focal_from_vertices(
    vertices: np.ndarray, trans: np.ndarray,
    width: int, height: int, fill_fraction: float = 0.85,
) -> float:
    """Estimate focal length that frames the mesh body in the render viewport.

    Projects mesh vertices through the MHR extrinsic (diag 0.01, -0.01, -0.01)
    and computes the focal length that makes the body fill ``fill_fraction``
    of the image. Resolution-independent — works correctly at any render size.

    Args:
        vertices: (V, 3) mesh vertices in mesh space (cm, Y-up)
        trans: (3,) translation in mesh space (cm, Y-up)
        width: Render width in pixels
        height: Render height in pixels
        fill_fraction: How much of the frame the body should fill (0-1)

    Returns:
        Focal length in pixels
    """
    translated = vertices + trans[np.newaxis, :]

    # Apply MHR extrinsic: diag(0.01, -0.01, -0.01, 1.0)
    x_cam = 0.01 * translated[:, 0]
    y_cam = -0.01 * translated[:, 1]
    z_cam = -0.01 * translated[:, 2]

    valid = z_cam > 0.01
    if valid.sum() == 0:
        return 1000.0

    max_x_ratio = np.abs(x_cam[valid] / z_cam[valid]).max()
    max_y_ratio = np.abs(y_cam[valid] / z_cam[valid]).max()

    fx = fill_fraction * (width / 2) / max_x_ratio if max_x_ratio > 1e-6 else 10000.0
    fy = fill_fraction * (height / 2) / max_y_ratio if max_y_ratio > 1e-6 else 10000.0

    return min(fx, fy)


def estimate_focal_from_keypoints(
    kp_2d: np.ndarray, kp_3d: np.ndarray, cam_t: np.ndarray,
    cx: float, cy: float,
) -> float | None:
    """Estimate focal length from 2D/3D keypoint correspondence.

    Uses the perspective projection model:
        x_pixel = focal * (x_3d + tx) / (z_3d + tz) + cx

    Note: cx/cy must match the coordinate system of kp_2d (the original
    image's principal point), not the render resolution.

    Args:
        kp_2d: (K, 2) projected 2D keypoints in pixels
        kp_3d: (K, 3) 3D keypoints
        cam_t: (3,) camera translation
        cx: Principal point x (pixels)
        cy: Principal point y (pixels)

    Returns:
        Estimated focal length in pixels, or None if estimation fails
    """
    kp_tr = kp_3d + cam_t[np.newaxis, :]
    depth = kp_tr[:, 2]
    valid = depth > 0.1

    focals = []
    for axis in [0, 1]:
        center = cx if axis == 0 else cy
        dv = kp_tr[valid, axis]
        dp = kp_2d[valid, axis] - center
        mask = np.abs(dv) > 1e-4
        if mask.sum() > 0:
            f_est = dp[mask] * depth[valid][mask] / dv[mask]
            focals.extend(f_est.tolist())

    return float(np.median(focals)) if focals else None


def resolve_focal_length(
    pose_data: dict, width: int, height: int, cli_focal: float | None,
) -> float:
    """Determine focal length from CLI override, pose file, or mesh vertices.

    Priority order:
    1. CLI --focal-length override
    2. ``focal_length`` key in the NPZ (from sam3d, already calibrated)
    3. Estimate from ``pred_vertices`` for optimal framing (resolution-independent)
    4. Default 1000.0

    Args:
        pose_data: Dict from load_pose() including 'raw' NPZ data and 'trans'
        width: Render width in pixels
        height: Render height in pixels
        cli_focal: CLI-provided focal length override (None if not set)

    Returns:
        Focal length in pixels
    """
    if cli_focal is not None:
        return cli_focal

    raw = pose_data["raw"]

    if "focal_length" in raw:
        return float(raw["focal_length"])

    if "pred_vertices" in raw:
        return estimate_focal_from_vertices(
            np.array(raw["pred_vertices"], dtype=np.float32),
            pose_data["trans"],
            width, height,
        )

    return 1000.0


def build_mhr_camera(
    focal: float, width: int, height: int, device: torch.device,
    cam_t: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "Camera":
    """Build a Camera for MHR rendering.

    Extrinsic: R = diag(0.01, -0.01, -0.01) converts MHR mesh coordinates
    (cm, Y-up) to camera coordinates (m, Y-down). The cam_t translation is
    in camera space (meters, Y-down) and goes directly into the extrinsic.

    Args:
        focal: Focal length in pixels
        width: Image width
        height: Image height
        device: Torch device
        cam_t: Camera translation in camera space (m, Y-down). This is
            pred_cam_t from the pose file by default.

    Returns:
        Camera instance
    """
    from gaussian_avatar.rendering import Camera

    cx, cy = width / 2.0, height / 2.0

    intrinsic = torch.tensor([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    tx, ty, tz = float(cam_t[0]), float(cam_t[1]), float(cam_t[2])

    extrinsic = torch.tensor([
        [0.01, 0.0, 0.0, tx],
        [0.0, -0.01, 0.0, ty],
        [0.0, 0.0, -0.01, tz],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    return Camera(
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        width=width,
        height=height,
    )


def build_deformation_config(config_dict: dict) -> "DeformationConfig":
    """Reconstruct a DeformationConfig from a checkpoint config dict.

    Args:
        config_dict: Deformation config as a plain dict from checkpoint

    Returns:
        DeformationConfig instance
    """
    from gaussian_avatar.configs import DeformationConfig

    valid_fields = {f.name for f in fields(DeformationConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    return DeformationConfig(**filtered)


def create_renderer(device_str: str, background: tuple[float, float, float], sh_degree: int | None):
    """Create the appropriate renderer for the target device.

    Args:
        device_str: 'cuda', 'mps', or 'cpu'
        background: Background RGB color
        sh_degree: SH degree or None for RGB mode

    Returns:
        Renderer instance
    """
    if device_str == "cuda":
        from gaussian_avatar.rendering import GaussianRenderer
        return GaussianRenderer(background_color=background, sh_degree=sh_degree)
    elif device_str == "mps":
        from gaussian_avatar.rendering import MPSRenderer
        return MPSRenderer(background_color=background, sh_degree=sh_degree)
    else:
        from gaussian_avatar.rendering import MockRenderer
        print("WARNING: CPU rendering uses MockRenderer (placeholder zeros).")
        return MockRenderer(allow_forward=True)


def load_avatar_from_checkpoint(
    checkpoint_path: str, model_path_override: str | None = None,
    max_stretch_override: float | None = None,
) -> tuple:
    """Load a GaussianAvatar from a trained checkpoint.

    Reconstructs the mesh, avatar model, and loads weights from the
    checkpoint. Handles model path resolution and Gaussian count resizing.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        model_path_override: Override mesh model files path (None uses checkpoint config)

    Returns:
        Tuple of (avatar, metadata) where metadata is a dict with
        model_type, lod, use_sh, max_sh_degree, config.
    """
    from gaussian_avatar.models import GaussianAvatar
    from gaussian_avatar.models.mesh import create_mesh

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    model_type = checkpoint.get("model_type", config.get("model", {}).get("model_type", "mhr"))
    lod = checkpoint.get("lod", config.get("model", {}).get("lod", 1))
    model_config = config.get("model", {})
    gender = model_config.get("gender", "male")

    # Resolve model path: CLI override > checkpoint config (if local) > default
    if model_path_override is not None:
        model_path = model_path_override
    else:
        ckpt_model_path = model_config.get("model_path", "mesh_models")
        if Path(ckpt_model_path).exists():
            model_path = ckpt_model_path
        else:
            model_path = _DEFAULT_MODEL_PATHS.get(model_type, "mesh_models")
            print(f"Checkpoint model_path '{ckpt_model_path}' not found locally, using '{model_path}'")

    num_gaussians = config.get("num_gaussians", 100000)
    disable_offsets = config.get("disable_offsets", False)
    use_sh = checkpoint.get("use_sh", config.get("use_sh", False))
    max_sh_degree = config.get("max_sh_degree", 3)
    max_stretch = config.get("max_stretch", 10.0)
    if max_stretch_override is not None:
        print(f"Overriding max_stretch: {max_stretch} -> {max_stretch_override}")
        max_stretch = max_stretch_override
    max_local_offset = config.get("max_local_offset", None)
    deformation_config = build_deformation_config(config.get("deformation", {}))

    print(f"Creating {model_type.upper()} mesh (lod={lod})...")
    mesh = create_mesh(
        model_type=model_type,
        model_path=model_path,
        gender=gender,
        lod=lod,
        device="cpu",
    )

    avatar = GaussianAvatar(
        mesh=mesh,
        num_gaussians=num_gaussians,
        deformation_config=deformation_config,
        disable_offsets=disable_offsets,
        use_sh=use_sh,
        max_sh_degree=max_sh_degree,
        max_stretch=max_stretch,
        max_local_offset=max_local_offset,
    )

    # Resize Gaussians if checkpoint has different count (from densification)
    state_dict = checkpoint["avatar_state_dict"]
    pos_key = "gaussian_model._local_positions"
    if pos_key in state_dict:
        ckpt_num = state_dict[pos_key].shape[0]
        if ckpt_num != avatar.gaussian_model.num_gaussians:
            avatar.gaussian_model.resize(ckpt_num)

    avatar.load_state_dict(state_dict)
    avatar.eval()
    print(f"Loaded avatar with {avatar.gaussian_model.num_gaussians} Gaussians")

    return avatar, {
        "model_type": model_type,
        "lod": lod,
        "use_sh": use_sh,
        "max_sh_degree": max_sh_degree,
        "config": config,
        "training_betas": checkpoint.get("training_betas"),
    }


def compute_scale_stats(scales: torch.Tensor) -> dict[str, dict[str, float]]:
    """Compute comprehensive statistics for Gaussian scales.

    Args:
        scales: (N, 3) scale tensor (linear space, positive)

    Returns:
        Dict keyed by dimension name ('x', 'y', 'z', 'all'), each containing
        mean, std, min, max, q25, median, q75.
    """
    stats = {}
    dim_names = ["x", "y", "z"]
    for i, name in enumerate(dim_names):
        col = scales[:, i]
        stats[name] = {
            "mean": col.mean().item(),
            "std": col.std().item(),
            "min": col.min().item(),
            "max": col.max().item(),
            "q25": col.quantile(0.25).item(),
            "median": col.median().item(),
            "q75": col.quantile(0.75).item(),
        }
    flat = scales.reshape(-1)
    stats["all"] = {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "q25": flat.quantile(0.25).item(),
        "median": flat.median().item(),
        "q75": flat.quantile(0.75).item(),
    }
    return stats


def print_scale_stats(label: str, stats: dict[str, dict[str, float]]) -> None:
    """Print scale statistics in a formatted table.

    Args:
        label: Section label (e.g. 'Before Posing (Canonical)')
        stats: Output from compute_scale_stats()
    """
    print(f"\n{'=' * 70}")
    print(f"  Scale Statistics: {label}")
    print(f"{'=' * 70}")
    header = f"  {'dim':<5} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'q25':>10} {'median':>10} {'q75':>10}"
    print(header)
    print(f"  {'-' * 75}")
    for dim_name in ["x", "y", "z", "all"]:
        s = stats[dim_name]
        print(
            f"  {dim_name:<5} {s['mean']:>10.6f} {s['std']:>10.6f} {s['min']:>10.6f} "
            f"{s['max']:>10.6f} {s['q25']:>10.6f} {s['median']:>10.6f} {s['q75']:>10.6f}"
        )
    print()


def compute_offset_stats(offsets: torch.Tensor) -> dict[str, dict[str, float]]:
    """Compute statistics for Gaussian-to-triangle centroid offsets.

    In the local triangle frame (orthonormal), the offset norm equals the
    global-space distance between the Gaussian and its parent centroid.

    Args:
        offsets: (N, 3) offset tensor in local triangle coordinates.
            x/y are tangent to the face, z is along the face normal.

    Returns:
        Dict keyed by dimension name ('x', 'y', 'z', 'dist'), each containing
        mean, std, min, max, q25, median, q75.
    """
    stats = {}
    dim_names = ["x", "y", "z"]
    for i, name in enumerate(dim_names):
        col = offsets[:, i]
        stats[name] = {
            "mean": col.mean().item(),
            "std": col.std().item(),
            "min": col.min().item(),
            "max": col.max().item(),
            "q25": col.quantile(0.25).item(),
            "median": col.median().item(),
            "q75": col.quantile(0.75).item(),
        }
    dists = offsets.norm(dim=-1)
    stats["dist"] = {
        "mean": dists.mean().item(),
        "std": dists.std().item(),
        "min": dists.min().item(),
        "max": dists.max().item(),
        "q25": dists.quantile(0.25).item(),
        "median": dists.median().item(),
        "q75": dists.quantile(0.75).item(),
    }
    return stats


def print_offset_stats(label: str, stats: dict[str, dict[str, float]]) -> None:
    """Print offset statistics in a formatted table.

    Args:
        label: Section label (e.g. 'Local Offsets (no MLP)')
        stats: Output from compute_offset_stats()
    """
    print(f"\n{'=' * 70}")
    print(f"  Offset Statistics: {label}")
    print(f"{'=' * 70}")
    header = f"  {'dim':<5} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'q25':>10} {'median':>10} {'q75':>10}"
    print(header)
    print(f"  {'-' * 75}")
    for dim_name in ["x", "y", "z", "dist"]:
        s = stats[dim_name]
        print(
            f"  {dim_name:<5} {s['mean']:>10.6f} {s['std']:>10.6f} {s['min']:>10.6f} "
            f"{s['max']:>10.6f} {s['q25']:>10.6f} {s['median']:>10.6f} {s['q75']:>10.6f}"
        )
    print()


def render_frame(
    avatar,
    pose_data: dict,
    betas: torch.Tensor,
    renderer,
    render_device: torch.device,
    bg_tensor: torch.Tensor,
    width: int,
    height: int,
    focal: float,
    cam_t,
    log_scales: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Run avatar forward pass and render a single frame.

    Args:
        avatar: GaussianAvatar model (eval mode)
        pose_data: Dict from load_pose()
        betas: Shape parameters tensor
        renderer: Renderer instance
        render_device: Torch device for rendering
        bg_tensor: Background color tensor on render_device
        width: Render width
        height: Render height
        focal: Focal length in pixels
        cam_t: Camera translation (pred_cam_t or override)
        log_scales: If True, return scale statistics alongside the image

    Returns:
        RGB image as (H, W, 3) uint8 numpy array. If log_scales=True,
        returns (image, scale_log) where scale_log is a dict with
        'canonical' and 'posed' statistics.
    """
    pose = torch.tensor(pose_data["pose"], dtype=torch.float32)

    with torch.no_grad():
        avatar_output = avatar.forward(pose, betas)

    means = avatar_output["means"].to(render_device)
    quats = avatar_output["quats"].to(render_device)
    scales = avatar_output["scales"].to(render_device)
    colors = avatar_output["colors"].to(render_device)
    opacities = avatar_output["opacities"].to(render_device)

    scale_log = None
    if log_scales:
        canonical = avatar_output["canonical_scales"].detach().cpu()
        posed = avatar_output["posed_scales"].detach().cpu()
        local_pos = avatar_output["local_positions"].detach().cpu()
        pos_offsets = avatar_output["offsets"]["position"].detach().cpu()
        scale_log = {
            "canonical": compute_scale_stats(canonical),
            "posed": compute_scale_stats(posed),
            "offsets_local": compute_offset_stats(local_pos),
            "offsets_posed": compute_offset_stats(local_pos + pos_offsets),
        }

    # Match training convention: add mesh-space translation to Gaussian
    # means and use a zero-translation camera.  The MHR extrinsic is
    # diag(0.01, -0.01, -0.01) — a scale+flip, NOT an orthonormal
    # rotation.  Putting cam_t in the extrinsic works for projection but
    # breaks SH view-direction computation in renderers that extract the
    # camera center via R^T*t (correct only for orthonormal R).  With
    # t=0 the camera center is (0,0,0) regardless, matching training.
    cam_t_np = np.array([float(cam_t[0]), float(cam_t[1]), float(cam_t[2])], dtype=np.float32)
    trans = cam_t_np * np.array([100.0, -100.0, -100.0], dtype=np.float32)
    means = means + torch.tensor(trans, dtype=torch.float32, device=render_device)
    camera = build_mhr_camera(focal, width, height, render_device, cam_t=(0.0, 0.0, 0.0))

    with torch.no_grad():
        result = renderer(
            means=means,
            quats=quats,
            scales=scales,
            colors=colors,
            opacities=opacities,
            camera=camera,
            background=bg_tensor,
        )

    rgb = result["rgb"]  # (3, H, W)
    rgb_np = rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = (rgb_np * 255).astype(np.uint8)

    if log_scales:
        return img, scale_log
    return img


def make_comparison(source_img: np.ndarray, render_img: np.ndarray) -> np.ndarray:
    """Create a side-by-side comparison image (source | render).

    The source image is resized to match the render height, preserving aspect ratio.

    Args:
        source_img: Source image as (H, W, 3) uint8 RGB
        render_img: Rendered image as (H, W, 3) uint8 RGB

    Returns:
        Side-by-side image as (H, W_total, 3) uint8 RGB
    """
    rh, rw = render_img.shape[:2]
    sh, sw = source_img.shape[:2]

    scale = rh / sh
    new_sw = int(sw * scale)
    source_resized = cv2.resize(source_img, (new_sw, rh), interpolation=cv2.INTER_AREA)

    return np.concatenate([source_resized, render_img], axis=1)


def get_canonical_pose(model_type: str) -> np.ndarray:
    """Construct the canonical rest pose for the given model type.

    MHR canonical is T-pose (all zeros). SMPL/SMPL-X canonical is A-pose
    with shoulders rotated ±45 degrees from T-pose.

    Args:
        model_type: 'mhr', 'smpl', or 'smplx'

    Returns:
        Canonical pose vector as numpy array
    """
    from gaussian_avatar.constants import (
        A_POSE_ANGLE,
        MHR_POSE_DIM,
        SMPL_LEFT_SHOULDER_IDX,
        SMPL_POSE_DIM,
        SMPL_RIGHT_SHOULDER_IDX,
        SMPLX_LEFT_SHOULDER_IDX,
        SMPLX_POSE_DIM,
        SMPLX_RIGHT_SHOULDER_IDX,
    )

    if model_type == "mhr":
        return np.zeros(MHR_POSE_DIM, dtype=np.float32)
    elif model_type == "smpl":
        pose = np.zeros(SMPL_POSE_DIM, dtype=np.float32)
        pose[SMPL_LEFT_SHOULDER_IDX * 3 + 2] = -A_POSE_ANGLE
        pose[SMPL_RIGHT_SHOULDER_IDX * 3 + 2] = A_POSE_ANGLE
        return pose
    elif model_type == "smplx":
        pose = np.zeros(SMPLX_POSE_DIM, dtype=np.float32)
        pose[SMPLX_LEFT_SHOULDER_IDX * 3 + 2] = -A_POSE_ANGLE
        pose[SMPLX_RIGHT_SHOULDER_IDX * 3 + 2] = A_POSE_ANGLE
        return pose
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def resolve_start_pose(args, model_type: str) -> np.ndarray:
    """Get the starting pose for interpolation.

    Uses --from-pose file if it exists, otherwise falls back to the
    canonical rest pose for the model type.

    Args:
        args: Parsed CLI arguments (needs args.from_pose)
        model_type: 'mhr', 'smpl', or 'smplx'

    Returns:
        Starting pose vector as numpy array
    """
    if args.from_pose is not None:
        from_path = Path(args.from_pose)
        if from_path.exists():
            pose_data = load_pose(str(from_path))
            print(f"Using start pose from: {from_path}")
            return pose_data["pose"]
        else:
            print(f"Warning: --from-pose '{args.from_pose}' not found, using canonical pose")
    return get_canonical_pose(model_type)


def ease_t(t: float, method: str) -> float:
    """Apply easing function to interpolation parameter.

    Args:
        t: Linear interpolation parameter in [0, 1]
        method: 'linear', 'smoothstep', or 'cosine'

    Returns:
        Eased parameter in [0, 1]
    """
    if method == "linear":
        return t
    elif method == "smoothstep":
        return 3 * t * t - 2 * t * t * t
    elif method == "cosine":
        return (1 - math.cos(math.pi * t)) / 2
    else:
        raise ValueError(f"Unknown easing method: {method}")


def interpolate_poses(
    canonical: np.ndarray,
    target: np.ndarray,
    num_frames: int,
    easing: str,
    hold_frames: int,
) -> list[np.ndarray]:
    """Generate interpolated poses from canonical to target.

    Args:
        canonical: Canonical rest pose vector
        target: Target pose vector
        num_frames: Number of interpolation frames
        easing: Easing method name
        hold_frames: Frames to hold on the final target pose

    Returns:
        List of pose arrays, length = num_frames + hold_frames
    """
    poses = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        alpha = ease_t(t, easing)
        pose = canonical * (1 - alpha) + target * alpha
        poses.append(pose)
    for _ in range(hold_frames):
        poses.append(target.copy())
    return poses


def write_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    """Write a list of RGB frames to an MP4 video.

    Args:
        frames: List of (H, W, 3) uint8 RGB images
        output_path: Output .mp4 file path
        fps: Frames per second
    """
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def resolve_cam_t(args, pose_data: dict):
    """Resolve camera translation from CLI override or pose file.

    Args:
        args: Parsed CLI arguments
        pose_data: Dict from load_pose()

    Returns:
        Camera translation for build_mhr_camera
    """
    if args.camera_position is not None:
        cam_parts = [float(x) for x in args.camera_position.split(",")]
        if len(cam_parts) != 3:
            raise ValueError(f"--camera-position must be X,Y,Z (3 values), got {len(cam_parts)}")
        return (
            0.01 * (-cam_parts[0]),
            -0.01 * (-cam_parts[1]),
            -0.01 * (-cam_parts[2]),
        )
    return pose_data["cam_t"]


def get_posed_gaussians(
    avatar,
    pose_data: dict,
    betas: torch.Tensor,
    cam_t,
) -> dict[str, torch.Tensor]:
    """Run avatar forward pass and return posed Gaussian properties on CPU.

    Applies the same mesh-space translation as render_frame so positions
    are in the same world coordinate system as rendered images.

    Args:
        avatar: GaussianAvatar model (eval mode)
        pose_data: Dict from load_pose()
        betas: Shape parameters tensor
        cam_t: Camera translation (pred_cam_t from pose file or CLI override)

    Returns:
        Dict with 'means' (N,3), 'quats' (N,4 wxyz), 'scales' (N,3 linear),
        'colors' (N,3 RGB [0,1] or N,K,3 SH), 'opacities' (N,) [0,1].
    """
    pose = torch.tensor(pose_data["pose"], dtype=torch.float32)
    with torch.no_grad():
        avatar_output = avatar.forward(pose, betas)

    means = avatar_output["means"]
    cam_t_np = np.array(
        [float(cam_t[0]), float(cam_t[1]), float(cam_t[2])], dtype=np.float32,
    )
    trans = cam_t_np * np.array([100.0, -100.0, -100.0], dtype=np.float32)
    means = means + torch.tensor(trans, dtype=means.dtype, device=means.device)

    return {
        "means": means.detach().cpu(),
        "quats": avatar_output["quats"].detach().cpu(),
        "scales": avatar_output["scales"].detach().cpu(),
        "colors": avatar_output["colors"].detach().cpu(),
        "opacities": avatar_output["opacities"].detach().cpu(),
    }


def save_gaussians_ply(
    path: str | Path,
    gaussians: dict[str, torch.Tensor],
    use_sh: bool = False,
    max_sh_degree: int = 0,
) -> None:
    """Save posed Gaussians to standard 3DGS-compatible binary PLY.

    Writes the format expected by the original graphdeco-inria/gaussian-splatting
    viewer (SIBR), SuperSplat, Luma, antimatter15/splat, Polycam, and gsplat.

    Conventions in the PLY file:
        - Scales: log-space (viewers apply exp())
        - Opacities: logit / inverse-sigmoid (viewers apply sigmoid())
        - Colors: 0th-order SH DC coefficients (viewers convert via 0.5 + SH_C0 * f_dc)
        - Rotations: wxyz quaternions (unnormalized)
        - Normals: zeros (placeholder, not used by any viewer)

    Args:
        path: Output .ply file path
        gaussians: Dict from get_posed_gaussians() with keys 'means', 'quats',
            'scales', 'colors', 'opacities'
        use_sh: Whether colors are SH coefficients (True) or RGB (False)
        max_sh_degree: Maximum SH degree (0-3). Only used when use_sh=True.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = gaussians["means"].shape[0]

    # Positions: direct world-space
    xyz = gaussians["means"].numpy().astype(np.float32)

    # Normals: zeros (required placeholder)
    normals = np.zeros((n, 3), dtype=np.float32)

    # Scales: linear → log
    log_scales = (
        torch.log(gaussians["scales"].clamp(min=1e-8)).numpy().astype(np.float32)
    )

    # Opacities: [0,1] → logit
    ops = gaussians["opacities"].clamp(1e-6, 1 - 1e-6)
    logit_ops = torch.log(ops / (1 - ops)).numpy().astype(np.float32).reshape(-1, 1)

    # Rotations: already wxyz
    rots = gaussians["quats"].numpy().astype(np.float32)

    # Colors → SH coefficients
    if use_sh:
        sh = gaussians["colors"].numpy().astype(np.float32)  # (N, K, 3)
        f_dc = sh[:, 0, :]  # (N, 3)
        if sh.shape[1] > 1:
            # Channel-first layout: (N, K-1, 3) → (N, 3, K-1) → (N, 3*(K-1))
            f_rest = np.ascontiguousarray(
                np.transpose(sh[:, 1:, :], (0, 2, 1))
            ).reshape(n, -1)
        else:
            f_rest = np.zeros((n, 0), dtype=np.float32)
    else:
        # RGB [0,1] → SH DC coefficients
        rgb = gaussians["colors"].clamp(0, 1).numpy().astype(np.float32)
        f_dc = (rgb - 0.5) / SH_C0
        f_rest = np.zeros((n, 0), dtype=np.float32)

    num_f_rest = f_rest.shape[1]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(num_f_rest):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.extend([
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ])
    header = "\n".join(header_lines) + "\n"

    all_props = [xyz, normals, f_dc]
    if num_f_rest > 0:
        all_props.append(f_rest)
    all_props.extend([logit_ops, log_scales, rots])

    data = np.concatenate(all_props, axis=1).astype(np.float32)

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())

    print(f"Saved PLY: {path} ({n:,} Gaussians, {path.stat().st_size / 1024 / 1024:.1f} MB)")


def resolve_betas(args, metadata: dict, pose_data: dict) -> torch.Tensor:
    """Resolve betas from training checkpoint or pose file.

    Args:
        args: Parsed CLI arguments
        metadata: Avatar metadata from load_avatar_from_checkpoint
        pose_data: Dict from load_pose()

    Returns:
        Betas tensor
    """
    training_betas = metadata["training_betas"]
    if not args.use_pose_betas and training_betas is not None:
        first_subject = next(iter(training_betas))
        return torch.tensor(training_betas[first_subject], dtype=torch.float32)
    return torch.tensor(pose_data["betas"], dtype=torch.float32)


def main() -> None:
    args = parse_args()

    bg_parts = [float(x) for x in args.background.split(",")]
    if len(bg_parts) != 3:
        raise ValueError(f"--background must be R,G,B (3 values), got {len(bg_parts)}")
    background = tuple(bg_parts)

    device_str = args.device if args.device else auto_detect_device()
    print(f"Using device: {device_str}")

    # Load avatar from checkpoint (once)
    print(f"Loading checkpoint: {args.checkpoint}")
    avatar, metadata = load_avatar_from_checkpoint(
        args.checkpoint, args.model_path, max_stretch_override=args.max_stretch,
    )
    use_sh = metadata["use_sh"]
    max_sh_degree = metadata["max_sh_degree"]

    # Create renderer (once)
    render_device = torch.device(device_str)
    sh_degree = max_sh_degree if use_sh else None
    renderer = create_renderer(device_str, background, sh_degree)
    bg_tensor = torch.tensor(background, dtype=torch.float32, device=render_device)

    if args.pose_dir is not None:
        _run_batch(args, avatar, metadata, renderer, render_device, bg_tensor)
    else:
        _run_single(args, avatar, metadata, renderer, render_device, bg_tensor)


def _run_single(args, avatar, metadata, renderer, render_device, bg_tensor) -> None:
    """Render a single pose file (original behavior)."""
    print(f"Loading pose: {args.pose_file}")
    pose_data = load_pose(args.pose_file)
    betas = resolve_betas(args, metadata, pose_data)

    focal = resolve_focal_length(pose_data, args.image_width, args.image_height, args.focal_length)
    cam_t = resolve_cam_t(args, pose_data)
    print(f"Focal length: {focal:.1f}px")

    print("Rendering...")
    result = render_frame(
        avatar, pose_data, betas, renderer, render_device, bg_tensor,
        args.image_width, args.image_height, focal, cam_t,
        log_scales=args.log,
    )

    if args.log:
        rgb_uint8, scale_log = result
        print_scale_stats("Before Posing (Canonical)", scale_log["canonical"])
        print_scale_stats("After Posing", scale_log["posed"])
        print_offset_stats("Local (no MLP)", scale_log["offsets_local"])
        print_offset_stats("Posed (local + MLP)", scale_log["offsets_posed"])
    else:
        rgb_uint8 = result

    if args.interpolate:
        start_pose = resolve_start_pose(args, metadata["model_type"])
        _run_interpolation_single(
            args, avatar, metadata, pose_data, betas, focal, cam_t,
            renderer, render_device, bg_tensor, rgb_uint8, start_pose,
        )
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")

    if args.save_ply:
        gaussians = get_posed_gaussians(avatar, pose_data, betas, cam_t)
        ply_path = output_path.with_suffix(".ply")
        save_gaussians_ply(ply_path, gaussians, metadata["use_sh"], metadata["max_sh_degree"])


def _run_interpolation_single(
    args, avatar, metadata, pose_data, betas, focal, cam_t,
    renderer, render_device, bg_tensor, final_img, start_pose,
) -> None:
    """Render interpolation to target pose (single-pose mode).

    Creates a directory with individual frames, an MP4 video, and the
    final target-pose image.

    Args:
        args: Parsed CLI arguments
        avatar: GaussianAvatar model
        metadata: Avatar metadata dict
        pose_data: Dict from load_pose()
        betas: Shape parameters tensor
        focal: Focal length in pixels
        cam_t: Camera translation
        renderer: Renderer instance
        render_device: Torch device
        bg_tensor: Background color tensor
        final_img: Already-rendered target-pose image (H, W, 3) uint8
        start_pose: Starting pose vector (from --from-pose or canonical)
    """
    output_dir = Path(args.output).with_suffix("")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    target = pose_data["pose"]
    interp_poses = interpolate_poses(
        start_pose, target, args.num_frames, args.easing, args.hold_frames,
    )

    total = len(interp_poses)
    rendered_frames = []
    ply_dir = output_dir / "ply" if args.save_ply else None
    if ply_dir is not None:
        ply_dir.mkdir(parents=True, exist_ok=True)

    for i, pose in enumerate(interp_poses):
        print(f"  Interpolation frame {i + 1}/{total}", end="\r")
        frame_pose_data = {**pose_data, "pose": pose}
        img = render_frame(
            avatar, frame_pose_data, betas, renderer, render_device, bg_tensor,
            args.image_width, args.image_height, focal, cam_t,
        )
        rendered_frames.append(img)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if ply_dir is not None:
            gaussians = get_posed_gaussians(avatar, frame_pose_data, betas, cam_t)
            save_gaussians_ply(
                ply_dir / f"frame_{i:04d}.ply", gaussians,
                metadata["use_sh"], metadata["max_sh_degree"],
            )
    print()

    video_path = output_dir / "interpolation.mp4"
    write_video(rendered_frames, video_path, args.fps)
    print(f"Saved video: {video_path}")

    final_path = output_dir / "final.png"
    cv2.imwrite(str(final_path), cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"Saved final: {final_path}")


def _run_batch(args, avatar, metadata, renderer, render_device, bg_tensor) -> None:
    """Render all .npz files in a directory with side-by-side comparisons."""
    pose_dir = Path(args.pose_dir)
    image_dir = Path(args.image_dir) if args.image_dir else pose_dir
    output_dir = Path(args.output)

    render_dir = output_dir / "renders"
    comparison_dir = output_dir / "comparisons"
    ply_dir = output_dir / "ply" if args.save_ply else None
    render_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    if ply_dir is not None:
        ply_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(pose_dir.glob("*.npz"))
    if not pose_files:
        print(f"No .npz files found in {pose_dir}")
        return

    print(f"Found {len(pose_files)} pose files in {pose_dir}")

    # Resolve betas once from first pose file (or training checkpoint)
    first_pose_data = load_pose(str(pose_files[0]))
    betas = resolve_betas(args, metadata, first_pose_data)

    # Resolve start pose once for all interpolations
    start_pose = resolve_start_pose(args, metadata["model_type"]) if args.interpolate else None
    all_interp_frames = []

    for i, pose_file in enumerate(pose_files):
        stem = pose_file.stem
        print(f"[{i + 1}/{len(pose_files)}] {stem}")

        pose_data = load_pose(str(pose_file))
        cam_t = resolve_cam_t(args, pose_data)

        # Use source image dimensions when available so pred_cam_t framing is correct
        source_path = find_matching_image(stem, image_dir)
        if source_path is not None:
            source_bgr = cv2.imread(str(source_path))
            source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            render_h, render_w = source_bgr.shape[:2]
        else:
            source_rgb = None
            render_w, render_h = args.image_width, args.image_height

        focal = resolve_focal_length(
            pose_data, render_w, render_h, args.focal_length,
        )

        frame_result = render_frame(
            avatar, pose_data, betas, renderer, render_device, bg_tensor,
            render_w, render_h, focal, cam_t,
            log_scales=args.log,
        )

        if args.log:
            rgb_uint8, scale_log = frame_result
            print_scale_stats(f"{stem} — Before Posing (Canonical)", scale_log["canonical"])
            print_scale_stats(f"{stem} — After Posing", scale_log["posed"])
            print_offset_stats(f"{stem} — Local (no MLP)", scale_log["offsets_local"])
            print_offset_stats(f"{stem} — Posed (local + MLP)", scale_log["offsets_posed"])
        else:
            rgb_uint8 = frame_result

        render_path = render_dir / f"{stem}.png"
        cv2.imwrite(str(render_path), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

        if source_rgb is not None:
            comparison = make_comparison(source_rgb, rgb_uint8)
            comp_path = comparison_dir / f"{stem}.png"
            cv2.imwrite(str(comp_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        if ply_dir is not None:
            gaussians = get_posed_gaussians(avatar, pose_data, betas, cam_t)
            save_gaussians_ply(
                ply_dir / f"{stem}.ply", gaussians,
                metadata["use_sh"], metadata["max_sh_degree"],
            )

        # Interpolation for this pose
        if args.interpolate:
            pose_frames = _run_interpolation_for_pose(
                args, avatar, metadata, pose_data, betas, focal, cam_t,
                renderer, render_device, bg_tensor,
                render_w, render_h, output_dir, stem, start_pose,
            )
            all_interp_frames.extend(pose_frames)

    print(f"Done. Renders: {render_dir}, Comparisons: {comparison_dir}")

    if args.interpolate and all_interp_frames:
        combined_path = output_dir / "combined.mp4"
        write_video(all_interp_frames, combined_path, args.fps)
        print(f"Saved combined video: {combined_path}")


def _run_interpolation_for_pose(
    args, avatar, metadata, pose_data, betas, focal, cam_t,
    renderer, render_device, bg_tensor,
    render_w, render_h, output_dir, stem, start_pose,
) -> list[np.ndarray]:
    """Render interpolation for a single pose in batch mode.

    Args:
        args: Parsed CLI arguments
        avatar: GaussianAvatar model
        metadata: Avatar metadata dict
        pose_data: Dict from load_pose()
        betas: Shape parameters tensor
        focal: Focal length in pixels
        cam_t: Camera translation
        renderer: Renderer instance
        render_device: Torch device
        bg_tensor: Background color tensor
        render_w: Render width in pixels
        render_h: Render height in pixels
        output_dir: Batch output directory
        stem: Pose file stem name
        start_pose: Starting pose vector (from --from-pose or canonical)

    Returns:
        List of rendered frames for combined video assembly
    """
    pose_interp_dir = output_dir / "interpolations" / stem
    frames_dir = pose_interp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = pose_interp_dir / "ply" if args.save_ply else None
    if ply_dir is not None:
        ply_dir.mkdir(parents=True, exist_ok=True)

    target = pose_data["pose"]
    interp_poses = interpolate_poses(
        start_pose, target, args.num_frames, args.easing, args.hold_frames,
    )

    total = len(interp_poses)
    rendered_frames = []
    for i, pose in enumerate(interp_poses):
        print(f"    Interp frame {i + 1}/{total}", end="\r")
        frame_pose_data = {**pose_data, "pose": pose}
        img = render_frame(
            avatar, frame_pose_data, betas, renderer, render_device, bg_tensor,
            render_w, render_h, focal, cam_t,
        )
        rendered_frames.append(img)
        frame_path = frames_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if ply_dir is not None:
            gaussians = get_posed_gaussians(avatar, frame_pose_data, betas, cam_t)
            save_gaussians_ply(
                ply_dir / f"frame_{i:04d}.ply", gaussians,
                metadata["use_sh"], metadata["max_sh_degree"],
            )
    print()

    video_path = pose_interp_dir / "interpolation.mp4"
    write_video(rendered_frames, video_path, args.fps)
    print(f"    Saved: {video_path}")

    return rendered_frames


if __name__ == "__main__":
    main()
