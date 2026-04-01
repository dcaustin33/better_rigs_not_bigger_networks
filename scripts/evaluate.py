#!/usr/bin/env python
"""Evaluation script for trained Gaussian Avatar models.

This script loads a trained checkpoint and evaluates it on the test set,
computing PSNR, SSIM, and LPIPS metrics. Optionally saves evaluation images.

By default, evaluation starts at the frame that begins the last 25% of the
video (matching the standard test split boundary). Use --start-frame to
override this with a custom starting frame index.

Configuration can be provided via a YAML file (--config) and/or CLI arguments.
CLI arguments override values from the YAML file.

Example usage:
    # Evaluate checkpoint on test set (last 25% of frames)
    uv run python scripts/evaluate.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual

    # Evaluate using a YAML config file
    uv run python scripts/evaluate.py \\
        --config configs/eval_default.yaml \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public

    # Dump default config to YAML and exit
    uv run python scripts/evaluate.py --dump-config configs/my_eval.yaml

    # Evaluate from a specific frame index
    uv run python scripts/evaluate.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual \\
        --start-frame 50

    # Save evaluation images
    uv run python scripts/evaluate.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public \\
        --save-images --num-images 20 \\
        --output-dir output/eval_results

    # Evaluate every 4th test frame (faster evaluation)
    uv run python scripts/evaluate.py \\
        --checkpoint output/experiment/checkpoints/final.pt \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual \\
        --skip 4
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from gaussian_avatar.configs.evaluation import BodyParamOptimConfig, EvaluationConfig
    from gaussian_avatar.models import GaussianAvatar
    from gaussian_avatar.models.per_frame_params import PerFrameParameters
    from gaussian_avatar.rendering import GaussianRenderer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained Gaussian Avatar model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pt file)",
    )

    config_group = parser.add_argument_group("Config")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides default EvaluationConfig)",
    )
    config_group.add_argument(
        "--dump-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Dump default EvaluationConfig to YAML file and exit",
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to people_snapshot_public dataset",
    )
    data_group.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Subject names to evaluate on (uses checkpoint config if not specified)",
    )
    data_group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        help="Evaluation image size (height width)",
    )
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers",
    )
    data_group.add_argument(
        "--dataset-format",
        type=str,
        default=None,
        choices=["auto", "public", "corrected", "zju_mocap"],
        help="Dataset format: 'auto' detects from directory structure, "
        "'public' for original HDF5/video format, 'corrected' for pre-extracted "
        "PNGs with per-frame SMPL pickles",
    )
    data_group.add_argument(
        "--undistort",
        action="store_true",
        default=None,
        help="Undistort images using camera distortion coefficients",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model files root directory (SMPL/SMPL-X/MHR)",
    )
    model_group.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["smpl", "smplx", "mhr"],
        help="Model type (smpl, smplx, or mhr)",
    )
    model_group.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["male", "female", "neutral"],
        help="SMPL/SMPL-X gender (ignored for MHR)",
    )
    model_group.add_argument(
        "--lod",
        type=int,
        default=None,
        choices=range(7),
        metavar="LOD",
        help="MHR level-of-detail (0-6). Only used with --model-type mhr.",
    )

    model_group.add_argument(
        "--background-color",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Background color RGB, each in [0, 1] (inherits from checkpoint config if not set)",
    )

    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Frame index to start evaluation at. Evaluates all frames from this "
        "index onwards. Default: start of last 25%% of video (matching test split)",
    )
    eval_group.add_argument(
        "--stop-frame",
        type=int,
        default=None,
        help="Frame index separating train/test (same as training config). "
        "When set, evaluation starts at this frame index. "
        "Overridden by --start-frame if both are provided.",
    )
    eval_group.add_argument(
        "--skip",
        type=int,
        default=None,
        help="Frame skip stride for evaluation. E.g. --skip 4 evaluates every "
        "4th test frame. When not specified, inherits from the training "
        "checkpoint config if available.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for evaluation outputs",
    )
    output_group.add_argument(
        "--save-images",
        action="store_true",
        default=None,
        help="Save evaluation images",
    )
    output_group.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of evaluation images to save",
    )
    output_group.add_argument(
        "--diagnostic",
        action="store_true",
        default=None,
        help="Save diagnostic images (error heatmaps, alpha-vs-mask, boundary "
        "analysis) and print per-region metric breakdowns",
    )
    output_group.add_argument(
        "--num-diagnostic",
        type=int,
        default=None,
        help="Number of diagnostic image sets to save",
    )

    optim_group = parser.add_argument_group("Body Parameter Optimization")
    optim_group.add_argument(
        "--body-optim-iters",
        type=int,
        default=None,
        metavar="N",
        help="Enable pre-evaluation body parameter optimization with N iterations. "
        "Freezes the avatar and optimizes per-frame body model parameters.",
    )
    optim_group.add_argument(
        "--body-optim-lr-pose",
        type=float,
        default=None,
        help="Learning rate for pose offsets during optimization",
    )
    optim_group.add_argument(
        "--body-optim-lr-betas",
        type=float,
        default=None,
        help="Learning rate for betas offsets during optimization",
    )
    optim_group.add_argument(
        "--body-optim-lr-trans",
        type=float,
        default=None,
        help="Learning rate for translation offsets during optimization",
    )

    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="Log evaluation metrics to Weights & Biases",
    )
    wandb_group.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (defaults to checkpoint filename)",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> "EvaluationConfig":
    """Build an EvaluationConfig from YAML and CLI overrides.

    Priority: CLI arguments > YAML config > EvaluationConfig defaults.

    Args:
        args: Parsed command line arguments

    Returns:
        Fully resolved EvaluationConfig
    """
    from gaussian_avatar.configs.evaluation import EvaluationConfig
    from gaussian_avatar.configs.yaml_utils import eval_config_from_yaml

    if args.config:
        print(f"Loading eval config from: {args.config}")
        config = eval_config_from_yaml(args.config)
    else:
        config = EvaluationConfig()

    # CLI overrides: apply explicit CLI args on top of YAML/defaults.
    # Only override when the CLI value is not None (i.e. explicitly provided).
    if args.checkpoint is not None:
        config.checkpoint = args.checkpoint
    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.subjects is not None:
        config.data.subjects = args.subjects
    if args.image_size is not None:
        config.image_size = tuple(args.image_size)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.dataset_format is not None:
        config.data.dataset_format = args.dataset_format
    if args.undistort is not None:
        config.data.undistort = args.undistort
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.model_type is not None:
        config.model_type = args.model_type
    if args.gender is not None:
        config.gender = args.gender
    if args.lod is not None:
        config.lod = args.lod
    if args.start_frame is not None:
        config.data.start_frame = args.start_frame
    if args.stop_frame is not None:
        config.data.stop_frame = args.stop_frame
    if args.skip is not None:
        config.data.skip = args.skip
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.save_images is not None:
        config.save_images = args.save_images
    if args.num_images is not None:
        config.num_images = args.num_images
    if args.diagnostic is not None:
        config.diagnostic = args.diagnostic
    if args.num_diagnostic is not None:
        config.num_diagnostic = args.num_diagnostic
    if args.background_color is not None:
        config.background_color = tuple(args.background_color)
    if args.body_optim_iters is not None:
        config.body_optim.enabled = True
        config.body_optim.num_iterations = args.body_optim_iters
    if args.body_optim_lr_pose is not None:
        config.body_optim.lr_pose = args.body_optim_lr_pose
    if args.body_optim_lr_betas is not None:
        config.body_optim.lr_betas = args.body_optim_lr_betas
    if args.body_optim_lr_trans is not None:
        config.body_optim.lr_trans = args.body_optim_lr_trans

    if args.wandb is not None:
        config.wandb = args.wandb
    if args.wandb_name is not None:
        config.wandb_name = args.wandb_name
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project

    return config


def format_metrics_table(
    metrics: dict[str, float],
    per_subject: dict[str, dict[str, float]] | None = None,
) -> str:
    """Format metrics as a readable table.

    Args:
        metrics: Overall metrics dictionary with psnr, ssim, lpips
        per_subject: Optional per-subject metrics breakdown

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Evaluation Results")
    lines.append("=" * 50)
    lines.append("")

    if per_subject:
        lines.append(f"{'Subject':<20} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
        lines.append("-" * 50)
        for subject, subj_metrics in sorted(per_subject.items()):
            lines.append(
                f"{subject:<20} "
                f"{subj_metrics['psnr']:>10.2f} "
                f"{subj_metrics['ssim']:>10.4f} "
                f"{subj_metrics['lpips']:>10.4f}"
            )
        lines.append("-" * 50)
        lines.append(
            f"{'Average':<20} "
            f"{metrics['psnr']:>10.2f} "
            f"{metrics['ssim']:>10.4f} "
            f"{metrics['lpips']:>10.4f}"
        )
    else:
        lines.append(f"{'Metric':<20} {'Value':>10}")
        lines.append("-" * 30)
        lines.append(f"{'PSNR (dB)':<20} {metrics['psnr']:>10.2f}")
        lines.append(f"{'SSIM':<20} {metrics['ssim']:>10.4f}")
        lines.append(f"{'LPIPS':<20} {metrics['lpips']:>10.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)


def save_metrics(
    metrics: dict[str, float],
    path: Path,
    per_subject: dict[str, dict[str, float]] | None = None,
) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Overall metrics dictionary
        path: Output file path
        per_subject: Optional per-subject metrics breakdown
    """
    output = {
        "overall": metrics,
    }
    if per_subject:
        output["per_subject"] = per_subject

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def compute_boundary_mask(
    mask: torch.Tensor,
    width: int = 5,
) -> torch.Tensor:
    """Compute a narrow band around the silhouette boundary.

    Args:
        mask: Binary mask (H, W) with values in {0, 1}
        width: Dilation radius in pixels for boundary band

    Returns:
        (H, W) float tensor that is 1 inside the boundary band, 0 elsewhere
    """
    mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
    kernel_size = 2 * width + 1
    dilated = F.max_pool2d(mask_4d, kernel_size, stride=1, padding=width)
    eroded = -F.max_pool2d(-mask_4d, kernel_size, stride=1, padding=width)
    boundary = (dilated - eroded).squeeze(0).squeeze(0)
    return boundary.clamp(0, 1)


def save_diagnostic_images(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    pred_alpha: torch.Tensor,
    gt_mask: torch.Tensor,
    output_dir: Path,
    prefix: str,
    lpips_map: torch.Tensor | None = None,
    ssim_error_map: torch.Tensor | None = None,
) -> None:
    """Save diagnostic visualisations for a single frame.

    Saves:
      - ``{prefix}_error_heatmap.png``:  per-pixel L1 error, jet-coloured
      - ``{prefix}_lpips_heatmap.png``:  per-pixel LPIPS distance, jet-coloured
        (only when *lpips_map* is provided)
      - ``{prefix}_ssim_heatmap.png``:   per-pixel SSIM error (1 - SSIM),
        jet-coloured (only when *ssim_error_map* is provided)
      - ``{prefix}_alpha_vs_mask.png``:  overlay showing alpha (green)
        vs GT mask (red) and their intersection (yellow)
      - ``{prefix}_boundary_error.png``: error only inside the boundary
        band (rest greyed out)
      - ``{prefix}_render.png``:  rendered image
      - ``{prefix}_gt.png``:     ground-truth image (masked)

    Args:
        pred_rgb: Rendered RGB (3, H, W) in [0, 1], premultiplied over black
        gt_rgb: GT RGB (3, H, W) in [0, 1], masked (black background)
        pred_alpha: Rendered alpha (H, W) in [0, 1]
        gt_mask: GT binary mask (H, W) in {0, 1}
        output_dir: Directory to write images
        prefix: Filename prefix (e.g. "male-3-casual_frame_0042")
        lpips_map: Optional per-pixel LPIPS spatial map (H, W)
        ssim_error_map: Optional per-pixel SSIM error map (H, W), 1 - SSIM
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_uint8(t: torch.Tensor) -> np.ndarray:
        return (t.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()

    def _save_jet_heatmap(
        error_map: torch.Tensor, path: Path, vmax: float | None = None
    ) -> None:
        """Normalize a scalar map and save as jet-coloured PNG.

        Args:
            error_map: Per-pixel scalar values (H, W)
            path: Output file path
            vmax: Fixed upper bound for normalization.  When set the
                colormap spans [0, vmax] so heatmaps are comparable
                across frames.  ``None`` falls back to per-image max.
        """
        arr = error_map.cpu().numpy()
        actual_max = float(arr.max())
        denom = vmax if vmax is not None else max(actual_max, 1e-5)
        normalised = np.clip(arr / denom, 0, 1)
        hm = (cm.jet(normalised)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(hm)
        draw = ImageDraw.Draw(img)
        label = f"max={actual_max:.4f}"
        if vmax is not None:
            label += f"  vmax={vmax}"
        draw.text((10, 10), label, fill=(255, 255, 255))
        img.save(path)

    l1_error = (pred_rgb - gt_rgb).abs().mean(dim=0)  # (H, W)
    l1_normalised = np.clip(
        l1_error.cpu().numpy() / max(l1_error.max().item(), 1e-5), 0, 1
    )
    _save_jet_heatmap(l1_error, output_dir / f"{prefix}_error_heatmap.png")

    if lpips_map is not None:
        _save_jet_heatmap(
            lpips_map, output_dir / f"{prefix}_lpips_heatmap.png", vmax=0.7
        )

    if ssim_error_map is not None:
        _save_jet_heatmap(
            ssim_error_map, output_dir / f"{prefix}_ssim_heatmap.png", vmax=0.7
        )

    # Red = GT mask only, Green = predicted alpha only, Yellow = overlap
    H, W = pred_alpha.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    alpha_np = (pred_alpha.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_np = (gt_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    overlay[:, :, 0] = mask_np       # R = GT mask
    overlay[:, :, 1] = alpha_np      # G = predicted alpha
    # B stays 0 — yellow where both are high
    Image.fromarray(overlay).save(output_dir / f"{prefix}_alpha_vs_mask.png")

    boundary = compute_boundary_mask(gt_mask, width=7)  # (H, W)
    # Show original render dimmed, with the boundary error in colour
    render_np = _to_uint8(pred_rgb).astype(np.float32) / 255.0
    dimmed = (render_np * 0.3)  # dim the full image
    boundary_np = boundary.cpu().numpy()
    heatmap_float = cm.jet(l1_normalised)[:, :, :3]
    for c in range(3):
        dimmed[:, :, c] = np.where(
            boundary_np > 0.5, heatmap_float[:, :, c], dimmed[:, :, c]
        )
    dimmed_uint8 = (dimmed * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(dimmed_uint8).save(
        output_dir / f"{prefix}_boundary_error.png"
    )

    Image.fromarray(_to_uint8(pred_rgb)).save(
        output_dir / f"{prefix}_render.png"
    )
    Image.fromarray(_to_uint8(gt_rgb)).save(
        output_dir / f"{prefix}_gt.png"
    )


def compute_masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute PSNR over only the masked (foreground) pixels.

    Args:
        pred: (3, H, W) in [0, 1]
        target: (3, H, W) in [0, 1]
        mask: (H, W) binary mask

    Returns:
        PSNR in dB computed over foreground pixels only
    """
    mask3 = mask.unsqueeze(0).expand_as(pred)
    fg_pixels = mask3.sum()
    if fg_pixels == 0:
        return float("inf")
    mse = ((pred - target) ** 2 * mask3).sum() / fg_pixels
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(1.0 / mse)).item()


def compute_gaussian_stats(
    avatar: "GaussianAvatar",
    global_scale_samples: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compute statistics of Gaussian scales and position offsets.

    Args:
        avatar: GaussianAvatar model
        global_scale_samples: Optional list of per-sample global scales (N, 3)
            to average for pose-dependent stats.

    Returns:
        Dictionary of stat name to float value.
    """
    gm = avatar.gaussian_model
    stats: dict[str, float] = {}

    with torch.no_grad():
        # Position offsets from face center
        offsets = gm.local_positions  # (N, 3)
        offset_dists = offsets.norm(dim=-1)  # (N,)

    stats["offset_dist/mean"] = offset_dists.mean().item()
    stats["offset_dist/max"] = offset_dists.max().item()
    stats["offset_dist/min"] = offset_dists.min().item()
    stats["offset_dist/std"] = offset_dists.std().item()

    for i, axis in enumerate(("x", "y", "z")):
        vals = offsets[:, i].abs()
        stats[f"offset_{axis}/mean"] = vals.mean().item()
        stats[f"offset_{axis}/max"] = vals.max().item()
        stats[f"offset_{axis}/std"] = vals.std().item()

    with torch.no_grad():
        # Local linear scales (pose-independent)
        local_scales = torch.exp(gm.local_scales)  # (N, 3)
        local_scale_all = local_scales.reshape(-1)

    stats["local_scale/mean"] = local_scale_all.mean().item()
    stats["local_scale/max"] = local_scale_all.max().item()
    stats["local_scale/min"] = local_scale_all.min().item()
    stats["local_scale/std"] = local_scale_all.std().item()

    for i, axis in enumerate(("x", "y", "z")):
        vals = local_scales[:, i]
        stats[f"local_scale_{axis}/mean"] = vals.mean().item()
        stats[f"local_scale_{axis}/max"] = vals.max().item()
        stats[f"local_scale_{axis}/min"] = vals.min().item()
        stats[f"local_scale_{axis}/std"] = vals.std().item()

    # Global scales (pose-dependent, averaged across eval samples)
    if global_scale_samples:
        with torch.no_grad():
            # Compute running mean to avoid OOM from stacking all samples
            avg_gs = global_scale_samples[0].clone()
            for i, s in enumerate(global_scale_samples[1:], 1):
                avg_gs += (s - avg_gs) / (i + 1)
            gs_all = avg_gs.reshape(-1)

        stats["global_scale/mean"] = gs_all.mean().item()
        stats["global_scale/max"] = gs_all.max().item()
        stats["global_scale/min"] = gs_all.min().item()
        stats["global_scale/std"] = gs_all.std().item()

        for i, axis in enumerate(("x", "y", "z")):
            vals = avg_gs[:, i]
            stats[f"global_scale_{axis}/mean"] = vals.mean().item()
            stats[f"global_scale_{axis}/max"] = vals.max().item()
            stats[f"global_scale_{axis}/min"] = vals.min().item()
            stats[f"global_scale_{axis}/std"] = vals.std().item()

    return stats


def format_gaussian_stats_table(stats: dict[str, float]) -> str:
    """Format Gaussian statistics as a readable table.

    Args:
        stats: Dictionary of stat name to float value

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("Gaussian Statistics")
    lines.append("=" * 60)

    lines.append(f"\n{'Offset from face center (L2 distance)':}")
    lines.append(f"  {'mean':>8}: {stats['offset_dist/mean']:.6f}")
    lines.append(f"  {'max':>8}: {stats['offset_dist/max']:.6f}")
    lines.append(f"  {'min':>8}: {stats['offset_dist/min']:.6f}")
    lines.append(f"  {'std':>8}: {stats['offset_dist/std']:.6f}")

    lines.append(f"\n{'Local scales (pose-independent)':}")
    lines.append(f"  {'':>8}  {'mean':>10} {'max':>10} {'min':>10} {'std':>10}")
    lines.append(f"  {'all':>8}: {stats['local_scale/mean']:>10.6f} {stats['local_scale/max']:>10.6f} {stats['local_scale/min']:>10.6f} {stats['local_scale/std']:>10.6f}")
    for axis in ("x", "y", "z"):
        lines.append(f"  {axis:>8}: {stats[f'local_scale_{axis}/mean']:>10.6f} {stats[f'local_scale_{axis}/max']:>10.6f} {stats[f'local_scale_{axis}/min']:>10.6f} {stats[f'local_scale_{axis}/std']:>10.6f}")

    if "global_scale/mean" in stats:
        lines.append(f"\n{'Global scales (avg across eval poses)':}")
        lines.append(f"  {'':>8}  {'mean':>10} {'max':>10} {'min':>10} {'std':>10}")
        lines.append(f"  {'all':>8}: {stats['global_scale/mean']:>10.6f} {stats['global_scale/max']:>10.6f} {stats['global_scale/min']:>10.6f} {stats['global_scale/std']:>10.6f}")
        for axis in ("x", "y", "z"):
            lines.append(f"  {axis:>8}: {stats[f'global_scale_{axis}/mean']:>10.6f} {stats[f'global_scale_{axis}/max']:>10.6f} {stats[f'global_scale_{axis}/min']:>10.6f} {stats[f'global_scale_{axis}/std']:>10.6f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def optimize_body_params_for_eval(
    optim_config: "BodyParamOptimConfig",
    avatar: "GaussianAvatar",
    renderer: "GaussianRenderer",
    test_loader: DataLoader,
    ckpt_config: dict,
    device: str,
    wandb_run: object | None = None,
) -> "PerFrameParameters":
    """Optimize per-frame body model parameters on the test set before evaluation.

    Freezes the avatar model and optimizes only per-frame offsets (pose,
    betas, translation) for N iterations to correct noisy dataset fits.
    Works with SMPL, SMPL-X, and MHR models.

    Args:
        optim_config: Body parameter optimization configuration.
        avatar: Trained GaussianAvatar model (will be frozen during optimization).
        renderer: Gaussian renderer.
        test_loader: DataLoader for the test set.
        ckpt_config: Checkpoint config dict for inheriting loss weights.
        device: Compute device string.
        wandb_run: Optional W&B run for logging.

    Returns:
        Trained PerFrameParameters with optimized body parameter offsets.
    """
    from gaussian_avatar.configs.per_frame_params import PerFrameParamsConfig
    from gaussian_avatar.losses.config import LossThresholds, LossWeights
    from gaussian_avatar.losses.total import TotalLoss
    from gaussian_avatar.models.per_frame_params import PerFrameParameters
    from gaussian_avatar.rendering import Camera

    print(f"\n{'=' * 50}")
    print(f"Pre-evaluation body model parameter optimization ({optim_config.num_iterations} iterations)")
    print(f"  optimize_pose={optim_config.optimize_pose}, "
          f"optimize_betas={optim_config.optimize_betas}, "
          f"optimize_trans={optim_config.optimize_trans}")
    print(f"  lr_pose={optim_config.lr_pose}, lr_betas={optim_config.lr_betas}, "
          f"lr_trans={optim_config.lr_trans}")
    print(f"{'=' * 50}")

    pfp_config = PerFrameParamsConfig(
        enabled=True,
        optimize_pose=optim_config.optimize_pose,
        optimize_betas=optim_config.optimize_betas,
        optimize_trans=optim_config.optimize_trans,
        lr_pose=optim_config.lr_pose,
        lr_betas=optim_config.lr_betas,
        lr_trans=optim_config.lr_trans,
    )

    per_frame_params = PerFrameParameters(pfp_config, test_loader.dataset)
    per_frame_params.to(device)

    avatar_param_states: list[tuple[torch.nn.Parameter, bool]] = []
    for param in avatar.parameters():
        avatar_param_states.append((param, param.requires_grad))
        param.requires_grad_(False)

    prev_enable_grad = getattr(avatar.mesh, "enable_grad", False)
    avatar.mesh.enable_grad = True
    avatar.train()

    # Resolve loss weights: explicit config > checkpoint > defaults
    if optim_config.loss_weights is not None:
        loss_weights = optim_config.loss_weights
    elif ckpt_config.get("loss_weights"):
        from gaussian_avatar.configs.yaml_utils import _dict_to_dataclass
        loss_weights = _dict_to_dataclass(LossWeights, ckpt_config["loss_weights"])
    else:
        loss_weights = LossWeights()

    if optim_config.loss_thresholds is not None:
        loss_thresholds = optim_config.loss_thresholds
    elif ckpt_config.get("loss_thresholds"):
        from gaussian_avatar.configs.yaml_utils import _dict_to_dataclass
        loss_thresholds = _dict_to_dataclass(LossThresholds, ckpt_config["loss_thresholds"])
    else:
        loss_thresholds = LossThresholds()

    loss_fn = TotalLoss(
        weights=loss_weights,
        thresholds=loss_thresholds,
    ).to(device)

    if loss_weights.lambda_aiap_position > 0 or loss_weights.lambda_aiap_covariance > 0:
        with torch.no_grad():
            canonical_positions = avatar.get_canonical_positions()
        loss_fn.initialize_aiap(canonical_positions)

    param_groups = per_frame_params.get_param_groups()
    optimizer = torch.optim.Adam(param_groups)

    data_iter = itertools.cycle(test_loader)
    pbar = tqdm(range(optim_config.num_iterations), desc="Body param optim", unit="iter")
    loss_history: list[float] = []
    ema_decay = 0.95
    ema_loss = None

    for i in pbar:
        batch = next(data_iter)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Apply per-frame offsets
        batch = per_frame_params.apply_to_batch(batch)

        pose = batch["pose"]
        betas = batch["betas"]
        trans = batch.get("trans")
        target_rgb = batch["image"]
        target_mask = batch["mask"]
        intrinsic = batch["intrinsic"]
        extrinsic = batch["extrinsic"]

        B = pose.shape[0]
        H, W = target_rgb.shape[2:]

        all_losses: list[dict[str, torch.Tensor]] = []

        for b in range(B):
            avatar_output = avatar(pose[b], betas[b])
            if trans is not None:
                avatar_output["means"] = avatar_output["means"] + trans[b]

            camera = Camera.from_dataset_sample(
                intrinsic=intrinsic[b],
                extrinsic=extrinsic[b],
                height=H,
                width=W,
            )

            render_output = renderer(
                means=avatar_output["means"],
                quats=avatar_output["quats"],
                scales=avatar_output["scales"],
                colors=avatar_output["colors"],
                opacities=avatar_output["opacities"],
                camera=camera,
                background=torch.zeros(3, device=device),
            )

            rendered_rgb = render_output["rgb"].unsqueeze(0)
            rendered_alpha = render_output["alpha"].unsqueeze(0).unsqueeze(0)

            target_rgb_b = target_rgb[b : b + 1]
            target_mask_b = target_mask[b : b + 1].unsqueeze(1)

            losses = loss_fn(
                rendered_rgb=rendered_rgb,
                rendered_alpha=rendered_alpha,
                target_rgb=target_rgb_b,
                target_mask=target_mask_b,
                local_positions=avatar_output["local_positions"],
                log_scales=avatar_output["log_scales"],
                delta_position=avatar_output["offsets"]["position"],
                delta_rotation=avatar_output["offsets"]["rotation_xyz"],
                delta_scale=avatar_output["offsets"]["scale"],
                delta_color=avatar_output["offsets"]["color"],
                canonical_positions=avatar_output["canonical_positions"],
                canonical_rotations=avatar_output["canonical_rotations"],
                canonical_scales=avatar_output["canonical_scales"],
                posed_positions=avatar_output["posed_positions"],
                posed_rotations=avatar_output["posed_rotations"],
                posed_scales=avatar_output["posed_scales"],
            )

            all_losses.append(losses)

        total_loss = sum(l["total"] for l in all_losses) / B
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_val = total_loss.item()
        loss_history.append(loss_val)
        ema_loss = loss_val if ema_loss is None else ema_decay * ema_loss + (1 - ema_decay) * loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}", ema=f"{ema_loss:.4f}")

        if wandb_run is not None:
            import wandb
            wandb.log({
                "body_optim/loss": loss_val,
                "body_optim/ema_loss": ema_loss,
            }, step=i)

    for param, requires_grad in avatar_param_states:
        param.requires_grad_(requires_grad)
    avatar.mesh.enable_grad = prev_enable_grad
    avatar.eval()

    if loss_history:
        n = len(loss_history)
        first_window = loss_history[:min(10, n)]
        last_window = loss_history[max(0, n - 10):]
        print(f"\nBody model parameter optimization complete ({n} iterations)")
        print(f"  loss first 10 avg: {sum(first_window) / len(first_window):.4f}")
        print(f"  loss last  10 avg: {sum(last_window) / len(last_window):.4f}")
        print(f"  loss min: {min(loss_history):.4f} (iter {loss_history.index(min(loss_history))})")
        reduction = (1 - sum(last_window) / len(last_window) / (sum(first_window) / len(first_window))) * 100
        print(f"  reduction: {reduction:.1f}%")
        if n >= 20:
            mid = n // 2
            second_half = loss_history[mid:]
            first_half = loss_history[:mid]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            if avg_second < avg_first - 1e-6:
                print(f"  trend: still decreasing (2nd half avg {avg_second:.4f} < 1st half avg {avg_first:.4f})")
            else:
                print(f"  trend: converged (2nd half avg {avg_second:.4f} ~= 1st half avg {avg_first:.4f})")
        print()
    else:
        print("\nBody model parameter optimization: no iterations run\n")

    return per_frame_params


def run_evaluation(config: "EvaluationConfig") -> dict[str, float]:
    """Run evaluation on test set.

    Args:
        config: Evaluation configuration (from YAML and/or CLI)

    Returns:
        Dictionary with evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if config.wandb:
        import wandb

        run_name = config.wandb_name or Path(config.checkpoint).stem
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "checkpoint": config.checkpoint,
                "subjects": config.data.subjects,
                "image_size": config.image_size,
                "start_frame": config.data.start_frame,
                "skip": config.data.skip,
                "model_type": config.model_type,
                "gender": config.gender,
                "lod": config.lod,
            },
            job_type="eval",
        )
        print(f"W&B run: {wandb_run.name} ({wandb_run.url})")

    # Import components (deferred to allow --help without heavy imports)
    from gaussian_avatar.data import (
        MHRDataset,
        PeopleSnapshotCorrectedDataset,
        PeopleSnapshotDataset,
        custom_collate,
    )
    from gaussian_avatar.data.mhr_dataset import MHRCorrectedDataset
    from gaussian_avatar.data.zju_mocap import ZJUMoCapDataset
    from gaussian_avatar.data.base import compute_split_indices
    from gaussian_avatar.data.people_snapshot import _detect_dataset_format
    from gaussian_avatar.models import GaussianAvatar
    from gaussian_avatar.models.mesh import create_mesh
    from gaussian_avatar.rendering import (
        Camera,
        GaussianRenderer,
        MockRenderer,
        GSPLAT_AVAILABLE,
    )
    from gaussian_avatar.training.checkpoint import load_checkpoint
    from gaussian_avatar.training.image_saver import ImageSaver
    from gaussian_avatar.training.metrics import (
        LPIPSMetric,
        compute_psnr,
        compute_ssim,
        compute_ssim_map,
    )

    if config.model_type == "mhr" and config.gender != "male":
        print(
            f"WARNING: gender={config.gender} is ignored for MHR model type. "
            "MHR uses identity coefficients instead of gender."
        )

    print(f"Loading {config.model_type.upper()} mesh from: {config.model_path}")
    mesh = create_mesh(
        model_type=config.model_type,
        model_path=config.model_path,
        gender=config.gender,
        lod=config.lod,
        device=device,
    )

    num_gaussians = config.num_gaussians
    if num_gaussians is None:
        num_gaussians = mesh.get_faces().shape[0]

    print("Creating GaussianAvatar...")
    avatar = GaussianAvatar(
        mesh=mesh,
        num_gaussians=num_gaussians,
        deformation_config=config.deformation,
        disable_offsets=config.disable_offsets,
        use_sh=config.use_sh,
        max_sh_degree=config.max_sh_degree,
        max_stretch=config.max_stretch,
        max_local_offset=config.max_local_offset,
    )

    checkpoint_path = Path(config.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_info = load_checkpoint(
        path=checkpoint_path,
        avatar=avatar,
        device=device,
    )
    avatar.to(device)
    iteration = checkpoint_info.get("iteration", 0)
    print(f"Loaded checkpoint from iteration {iteration}")

    # Inherit settings from checkpoint config when not provided via CLI
    ckpt_config = checkpoint_info.get("config") or {}
    ckpt_data = ckpt_config.get("data") or {}

    subjects = config.data.subjects
    if subjects is None:
        subjects = ckpt_data.get("subjects") or ckpt_config.get("subjects")
    if subjects is None:
        raise ValueError(
            "No subjects specified. Provide --subjects or use a checkpoint with config."
        )

    data_root = config.data.data_root
    if data_root is None:
        raise ValueError("No data root specified. Provide --data-root.")

    print(f"Loading dataset from: {data_root}")
    print(f"Subjects: {subjects}")
    subj_arg = subjects if len(subjects) > 1 else subjects[0]

    # Track dataset format for downstream logic (e.g. start_frame default)
    dataset_format = config.data.dataset_format

    if dataset_format == "zju_mocap":
        train_cameras = config.data.train_cameras or ["Camera_B1"]
        print(f"ZJU MoCap eval: train cameras={train_cameras}, evaluating on held-out cameras")

        # Resolve stop_frame: CLI/YAML > checkpoint config > None (all frames)
        zju_stop_frame = config.data.stop_frame
        if zju_stop_frame is None and ckpt_data.get("stop_frame"):
            zju_stop_frame = ckpt_data["stop_frame"]
            print(f"Inherited stop_frame={zju_stop_frame} from checkpoint config")
        if zju_stop_frame is not None:
            print(f"ZJU MoCap stop_frame: {zju_stop_frame} (evaluating frames [0, {zju_stop_frame}))")

        test_dataset = ZJUMoCapDataset(
            data_root=data_root,
            subjects=subj_arg,
            split="test",
            image_size=config.image_size,
            train_cameras=train_cameras,
            stop_frame=zju_stop_frame,
            model_type=config.model_type,
            preprocess_masks=config.data.preprocess_masks,
        )
    elif config.model_type == "mhr":
        # Detect corrected format: check for cam000/mhr/raw/ directory
        if dataset_format == "auto":
            check_subject = subjects[0] if isinstance(subjects, list) else subjects
            mhr_raw = Path(data_root) / check_subject / "cam000" / "mhr" / "raw"
            dataset_format = "corrected" if mhr_raw.is_dir() else "standard"
            print(f"Auto-detected MHR dataset format: {dataset_format}")

        if dataset_format == "corrected":
            mhr_kwargs = {}
            if config.data.stop_frame is not None:
                mhr_kwargs["stop_frame"] = config.data.stop_frame
            # NOTE: skip is NOT forwarded to the dataset — it would subsample
            # ALL frames (train+test) before we filter to test-only.  The
            # evaluate script applies skip itself after selecting test frames.
            if config.data.undistort:
                mhr_kwargs["undistort"] = True
            test_dataset = MHRCorrectedDataset(
                data_root=data_root,
                subjects=subj_arg,
                split="all",
                image_size=config.image_size,
                preprocess_masks=config.data.preprocess_masks,
                **mhr_kwargs,
            )
        else:
            test_dataset = MHRDataset(
                data_root=data_root,
                subjects=subj_arg,
                split="all",
                image_size=config.image_size,
                preprocess_masks=config.data.preprocess_masks,
            )
    else:
        if dataset_format == "auto":
            dataset_format = _detect_dataset_format(data_root, subjects)
            print(f"Auto-detected dataset format: {dataset_format}")

        undistort = config.data.undistort
        if dataset_format == "corrected":
            # Resolve fits model type: use checkpoint config if available,
            # otherwise fall back to the evaluation model_type.  Training
            # uses fits_model_type (or model_type) to pick smpls/ vs smplxs/.
            fits_model_type = config.model_type
            ckpt_model = ckpt_config.get("model") or {}
            if ckpt_model.get("fits_model_type"):
                fits_model_type = ckpt_model["fits_model_type"]
            elif ckpt_model.get("model_type"):
                fits_model_type = ckpt_model["model_type"]
            print(f"Corrected dataset fits model type: {fits_model_type}")

            test_dataset = PeopleSnapshotCorrectedDataset(
                data_root=data_root,
                subjects=subj_arg,
                split="all",
                image_size=config.image_size,
                undistort=undistort,
                model_type=fits_model_type,
                preprocess_masks=config.data.preprocess_masks,
            )
        else:
            test_dataset = PeopleSnapshotDataset(
                data_root=data_root,
                subjects=subj_arg,
                split="all",
                image_size=config.image_size,
                undistort=undistort,
                preprocess_masks=config.data.preprocess_masks,
            )

    # ZJU MoCap already has its split built in (test = held-out cameras),
    # so skip the start_frame filtering for it.
    if dataset_format == "zju_mocap":
        skip = config.data.skip
        if skip is not None and skip > 1:
            test_dataset._index_mapping = test_dataset._index_mapping[::skip]
            print(f"Frame skip: {skip} (evaluating every {skip}-th test frame)")
    else:
        # Determine start frame for evaluation
        start_frame = config.data.start_frame
        if start_frame is None and config.data.stop_frame is not None:
            # Use stop_frame as the evaluation start (matches training split)
            start_frame = config.data.stop_frame
        if start_frame is None and ckpt_data.get("stop_frame"):
            # Inherit stop_frame from training config
            start_frame = ckpt_data["stop_frame"]
            print(f"Inherited stop_frame={start_frame} from checkpoint config")
        if start_frame is None:
            # Default: start of last 25% of video (matching test split boundary)
            subject = test_dataset.subjects[0]
            subject_data = test_dataset.get_subject_data(subject)
            start_frame_arg = 0 if dataset_format == "corrected" else 1
            _, test_indices = compute_split_indices(
                subject_data.num_frames, start_frame=start_frame_arg
            )
            start_frame = int(test_indices[0])

        test_dataset._index_mapping = [
            (s, f) for s, f in test_dataset._index_mapping if f >= start_frame
        ]

        # Apply frame skip to subsample test frames (only if explicitly set;
        # training skip should NOT carry over to evaluation by default)
        skip = config.data.skip
        if skip is not None and skip > 1:
            test_dataset._index_mapping = test_dataset._index_mapping[::skip]
            print(f"Frame skip: {skip} (evaluating every {skip}-th test frame)")

    # Resolve background color (CLI > YAML > checkpoint > default black)
    bg_color = config.background_color
    if bg_color is None and ckpt_config.get("background_color"):
        bg_color = tuple(ckpt_config["background_color"])
        print(f"Inherited background_color={bg_color} from checkpoint config")
    bg_color = bg_color or (0.0, 0.0, 0.0)

    if dataset_format == "zju_mocap":
        print(f"Evaluating {len(test_dataset)} samples (held-out cameras)")
    else:
        print(f"Evaluating from frame {start_frame} ({len(test_dataset)} samples)")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=custom_collate,
    )

    sh_degree = config.max_sh_degree if config.use_sh else None
    if GSPLAT_AVAILABLE:
        print("Using GaussianRenderer (gsplat)")
        renderer = GaussianRenderer(
            background_color=bg_color,
            sh_degree=sh_degree,
        ).to(device)
    else:
        print("WARNING: gsplat not available, using MockRenderer")
        renderer = MockRenderer(allow_forward=True)

    optimized_per_frame_params = None
    if config.body_optim.enabled:
        optimized_per_frame_params = optimize_body_params_for_eval(
            optim_config=config.body_optim,
            avatar=avatar,
            renderer=renderer,
            test_loader=test_loader,
            ckpt_config=ckpt_config,
            device=device,
            wandb_run=wandb_run,
        )

    image_saver = None
    if config.save_images:
        images_dir = output_dir / "images"
        image_saver = ImageSaver(output_dir=images_dir, save_gt=True)
        print(f"Saving evaluation images to: {images_dir}")

    diagnostic = config.diagnostic
    diagnostic_dir = output_dir / "diagnostic" if diagnostic else None
    num_diagnostic = config.num_diagnostic
    diagnostics_saved = 0

    print("Running evaluation...")
    avatar.eval()
    lpips_net = getattr(config, "lpips_net", "vgg")
    print(f"LPIPS network: {lpips_net}")
    lpips_metric = LPIPSMetric(device=device, net=lpips_net)

    psnrs: list[float] = []
    ssims: list[float] = []
    lpips_vals: list[float] = []
    fg_psnrs: list[float] = []
    boundary_psnrs: list[float] = []
    global_scale_samples: list[torch.Tensor] = []
    images_saved = 0

    per_subject_metrics: dict[str, dict[str, list[float]]] = {}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
        for batch in pbar:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if optimized_per_frame_params is not None:
                batch = optimized_per_frame_params.apply_to_batch(batch)

            pose = batch["pose"]
            betas = batch["betas"]
            trans = batch.get("trans")
            target_rgb = batch["image"]
            target_mask = batch["mask"]
            intrinsic = batch["intrinsic"]
            extrinsic = batch["extrinsic"]
            frame_idx = batch.get("frame_idx", list(range(pose.shape[0])))
            subject = batch.get("subject", [None] * pose.shape[0])

            B = pose.shape[0]
            H, W = target_rgb.shape[2:]

            for b in range(B):
                avatar_output = avatar(pose[b], betas[b])
                if trans is not None:
                    avatar_output["means"] = avatar_output["means"] + trans[b]

                camera = Camera.from_dataset_sample(
                    intrinsic=intrinsic[b],
                    extrinsic=extrinsic[b],
                    height=H,
                    width=W,
                )

                render_output = renderer(
                    means=avatar_output["means"],
                    quats=avatar_output["quats"],
                    scales=avatar_output["scales"],
                    colors=avatar_output["colors"],
                    opacities=avatar_output["opacities"],
                    camera=camera,
                )

                pred_rgb = render_output["rgb"]
                pred_alpha = render_output["alpha"]
                gt_rgb_raw = target_rgb[b]
                gt_mask = target_mask[b]

                # Composite GT over the same background as the renderer
                # so metrics are not penalised by background mismatch.
                # pred_rgb is already composited over the renderer's
                # background, so do NOT multiply by alpha again.
                # Clamp to [0,1] — gsplat accumulation can overshoot
                # by tiny amounts (e.g. 1.001) due to floating-point.
                pred_rgb = pred_rgb.clamp(0.0, 1.0)

                bg_tensor = torch.tensor(bg_color, device=device, dtype=gt_rgb_raw.dtype)
                mask3 = gt_mask.unsqueeze(0)
                gt_rgb = gt_rgb_raw * mask3 + bg_tensor.view(3, 1, 1) * (1 - mask3)

                psnr = compute_psnr(pred_rgb, gt_rgb)
                ssim = compute_ssim(pred_rgb, gt_rgb)
                lpips_val = lpips_metric(pred_rgb, gt_rgb)

                psnrs.append(psnr)
                ssims.append(ssim)
                lpips_vals.append(lpips_val)
                global_scale_samples.append(avatar_output["scales"])

                # Diagnostic: foreground-only and boundary PSNR
                if diagnostic:
                    fg_psnrs.append(compute_masked_psnr(pred_rgb, gt_rgb, gt_mask))
                    bnd = compute_boundary_mask(gt_mask, width=7)
                    boundary_psnrs.append(
                        compute_masked_psnr(pred_rgb, gt_rgb, bnd)
                    )

                subj = subject[b] if isinstance(subject, list) else subject
                if subj:
                    if subj not in per_subject_metrics:
                        per_subject_metrics[subj] = {
                            "psnr": [],
                            "ssim": [],
                            "lpips": [],
                        }
                    per_subject_metrics[subj]["psnr"].append(psnr)
                    per_subject_metrics[subj]["ssim"].append(ssim)
                    per_subject_metrics[subj]["lpips"].append(lpips_val)

                if image_saver is not None and images_saved < config.num_images:
                    frame = frame_idx[b] if isinstance(frame_idx, list) else frame_idx
                    image_saver.save_evaluation_image(
                        rendered_rgb=pred_rgb,
                        target_rgb=gt_rgb,
                        frame_idx=frame,
                        subject=subj,
                    )

                    posed_verts = avatar.mesh.forward(pose[b], betas[b])
                    if trans is not None:
                        posed_verts = posed_verts + trans[b]
                    overlay_prefix = f"frame_{frame:04d}"
                    if subj:
                        overlay_prefix = f"{subj}_{overlay_prefix}"
                    image_saver.save_mesh_overlay(
                        target_rgb=gt_rgb,
                        vertices=posed_verts,
                        faces=avatar.mesh.get_faces(),
                        intrinsic=intrinsic[b],
                        extrinsic=extrinsic[b],
                        path=images_dir / f"{overlay_prefix}_gt_mesh_overlay.png",
                    )

                    images_saved += 1

                if diagnostic and diagnostics_saved < num_diagnostic:
                    frame = frame_idx[b] if isinstance(frame_idx, list) else frame_idx
                    prefix = f"frame_{frame:04d}"
                    if subj:
                        prefix = f"{subj}_{prefix}"
                    lpips_map = lpips_metric.spatial_map(pred_rgb, gt_rgb)
                    ssim_error = compute_ssim_map(pred_rgb, gt_rgb)
                    save_diagnostic_images(
                        pred_rgb=pred_rgb,
                        gt_rgb=gt_rgb,
                        pred_alpha=pred_alpha,
                        gt_mask=gt_mask,
                        output_dir=diagnostic_dir,
                        prefix=prefix,
                        lpips_map=lpips_map,
                        ssim_error_map=ssim_error,
                    )

                    diagnostics_saved += 1

            pbar.set_postfix(
                psnr=f"{sum(psnrs) / len(psnrs):.2f}",
                ssim=f"{sum(ssims) / len(ssims):.4f}",
                lpips=f"{sum(lpips_vals) / len(lpips_vals):.4f}",
            )

    overall_metrics = {
        "psnr": sum(psnrs) / len(psnrs) if psnrs else 0.0,
        "ssim": sum(ssims) / len(ssims) if ssims else 0.0,
        "lpips": sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0.0,
    }

    gaussian_stats = compute_gaussian_stats(
        avatar,
        global_scale_samples=global_scale_samples,
    )
    overall_metrics["gaussian_stats"] = gaussian_stats

    per_subject_avg = None
    if per_subject_metrics:
        per_subject_avg = {}
        for subj, metrics in per_subject_metrics.items():
            per_subject_avg[subj] = {
                "psnr": sum(metrics["psnr"]) / len(metrics["psnr"]),
                "ssim": sum(metrics["ssim"]) / len(metrics["ssim"]),
                "lpips": sum(metrics["lpips"]) / len(metrics["lpips"]),
            }

    print()
    print(format_metrics_table(overall_metrics, per_subject_avg))
    print(format_gaussian_stats_table(gaussian_stats))

    if diagnostic and fg_psnrs:
        overall_metrics["fg_psnr"] = sum(fg_psnrs) / len(fg_psnrs)
        overall_metrics["boundary_psnr"] = (
            sum(boundary_psnrs) / len(boundary_psnrs)
        )
        print()
        print("Diagnostic Breakdown")
        print("-" * 50)
        print(f"  Full-image PSNR:      {overall_metrics['psnr']:.2f} dB")
        print(f"  Foreground-only PSNR: {overall_metrics['fg_psnr']:.2f} dB")
        print(f"  Boundary-band PSNR:   {overall_metrics['boundary_psnr']:.2f} dB")
        print(f"  (boundary = 7px band around GT silhouette)")
        print()
        delta = overall_metrics["fg_psnr"] - overall_metrics["boundary_psnr"]
        print(
            f"  FG-vs-boundary gap:   {delta:+.2f} dB "
            f"({'boundary is the bottleneck' if delta > 0 else 'interior is the bottleneck'})"
        )
        print(f"\n  Diagnostic images saved to: {diagnostic_dir}")

    metrics_path = output_dir / "metrics.json"
    save_metrics(overall_metrics, metrics_path, per_subject_avg)
    print(f"\nMetrics saved to: {metrics_path}")

    if config.save_images:
        print(f"Images saved to: {output_dir / 'images'}")

    if wandb_run is not None:
        import wandb

        wandb.log(
            {
                "eval/psnr": overall_metrics["psnr"],
                "eval/ssim": overall_metrics["ssim"],
                "eval/lpips": overall_metrics["lpips"],
                **{f"eval/gaussians/{k}": v for k, v in gaussian_stats.items()},
            }
        )
        if per_subject_avg:
            for subj, subj_metrics in per_subject_avg.items():
                wandb.log(
                    {
                        f"eval/{subj}/psnr": subj_metrics["psnr"],
                        f"eval/{subj}/ssim": subj_metrics["ssim"],
                        f"eval/{subj}/lpips": subj_metrics["lpips"],
                    }
                )
        wandb.finish()

    return overall_metrics


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # Handle --dump-config early (no heavy imports needed)
    if args.dump_config:
        from gaussian_avatar.configs.evaluation import EvaluationConfig
        from gaussian_avatar.configs.yaml_utils import eval_config_to_yaml

        eval_config_to_yaml(EvaluationConfig(), args.dump_config)
        print(f"Wrote default eval config to {args.dump_config}")
        return

    config = build_config(args)

    if config.checkpoint is None:
        print("error: --checkpoint is required (via CLI or YAML config)")
        raise SystemExit(2)

    run_evaluation(config)


if __name__ == "__main__":
    main()
