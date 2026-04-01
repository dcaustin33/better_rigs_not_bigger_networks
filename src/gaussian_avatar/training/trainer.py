"""Main training loop for Gaussian Avatar.

This module provides the Trainer class that orchestrates the complete
training process including forward pass, loss computation, optimization,
densification, and logging.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaussian_avatar.configs.training import TrainingConfig
from gaussian_avatar.losses import TotalLoss
from gaussian_avatar.losses.ssl import (
    compute_ssl_silhouette_loss,
    jitter_camera,
    perturb_pose,
    rasterize_mesh_silhouette,
)
from gaussian_avatar.models.mesh_cache import MeshCache
from gaussian_avatar.models.per_frame_params import PerFrameParameters
from gaussian_avatar.rendering import Camera
from gaussian_avatar.training.checkpoint import (
    load_checkpoint as _load_checkpoint,
    save_checkpoint as _save_checkpoint,
)
from gaussian_avatar.training.densification import DensificationManager
from gaussian_avatar.training.image_saver import ImageSaver
from gaussian_avatar.training.metrics import LPIPSMetric, compute_psnr, compute_ssim, compute_ssim_map
from gaussian_avatar.training.optimizer import create_optimizer, create_scheduler

if TYPE_CHECKING:
    from gaussian_avatar.models import GaussianAvatar
    from gaussian_avatar.rendering import GaussianRenderer, MockRenderer

logger = logging.getLogger(__name__)


class Trainer:
    """Main training loop for Gaussian Avatar.

    Orchestrates the training process including forward pass,
    loss computation, optimization, densification, and logging.

    Args:
        avatar: GaussianAvatar model
        renderer: GaussianRenderer for rasterization (or MockRenderer for testing)
        loss_fn: TotalLoss instance
        train_loader: Training DataLoader
        test_loader: Test DataLoader (for evaluation)
        config: TrainingConfig with hyperparameters
        checkpoint_dir: Directory for saving checkpoints
        use_wandb: Whether to log to Weights & Biases
    """

    def __init__(
        self,
        avatar: "GaussianAvatar",
        renderer: "GaussianRenderer | MockRenderer",
        loss_fn: TotalLoss,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: TrainingConfig,
        checkpoint_dir: str | Path,
        use_wandb: bool = True,
    ) -> None:
        self.avatar = avatar
        self.renderer = renderer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.use_wandb = use_wandb

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.per_frame_params: PerFrameParameters | None = None
        if config.per_frame_params.enabled:
            self.per_frame_params = PerFrameParameters(
                config.per_frame_params, train_loader.dataset
            )
            self.per_frame_params.to(avatar.gaussian_model.local_positions.device)
            # Enable gradient flow through mesh forward pass
            if config.per_frame_params.optimize_pose or config.per_frame_params.optimize_betas:
                avatar.mesh.enable_grad = True
                logger.info("Enabled gradient flow through mesh forward pass")

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.densification = DensificationManager(
            avatar.gaussian_model,
            config.densification,
        )

        if self.config.save_images_interval > 0 or self.config.save_all_last_n > 0:
            images_dir = (
                Path(self.config.save_images_dir)
                if self.config.save_images_dir
                else self.checkpoint_dir.parent / "images"
            )
            self.image_saver = ImageSaver(
                output_dir=images_dir,
                save_gt=self.config.save_gt_images,
                background=True,
            )
        else:
            self.image_saver = None

        self.iteration = 0
        self._ssl_call_count = 0
        self._ssl_renders_dir: Path | None = None
        if config.ssl.save_renders_dir and config.ssl.save_renders_interval > 0:
            self._ssl_renders_dir = (
                Path(config.output_dir) / config.experiment_name / config.ssl.save_renders_dir
            )
            self._ssl_renders_dir.mkdir(parents=True, exist_ok=True)
        self._lpips_metric: LPIPSMetric | None = None
        self._mesh_cache: MeshCache | None = None

        if config.torch_compile:
            self._apply_torch_compile()

        if config.use_sh:
            self._current_sh_degree = config.get_sh_degree_at(0)
            self.renderer.sh_degree = self._current_sh_degree
        else:
            self._current_sh_degree = None

        if self.use_wandb:
            try:
                import wandb

                self.wandb = wandb
            except ImportError:
                self.wandb = None
                self.use_wandb = False
        else:
            self.wandb = None

    @property
    def device(self) -> torch.device:
        """Return the device of the avatar model."""
        return self.avatar.gaussian_model.local_positions.device

    def _setup_optimizer(self) -> torch.optim.Adam:
        """Create optimizer with per-parameter learning rates."""
        return create_optimizer(
            self.avatar, self.config, per_frame_params=self.per_frame_params
        )

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Create learning rate scheduler."""
        return create_scheduler(self.optimizer, self.config)

    def _apply_torch_compile(self) -> None:
        """Compile the deformation MLP with torch.compile for faster training."""
        if self.avatar.deformation_mlp is not None:
            logger.info("Compiling deformation MLP with torch.compile")
            self.avatar.deformation_mlp = torch.compile(
                self.avatar.deformation_mlp, dynamic=True
            )

    def _can_use_mesh_cache(self) -> bool:
        """Check whether the mesh forward pass cache can be used.

        The cache is valid only when pose and betas are fixed across iterations
        (i.e., per-frame pose/betas optimization is disabled). Translation
        optimization is fine because trans is added to Gaussian means after
        the avatar forward pass.

        Returns:
            True if caching is safe, False if per-frame optimization would
            invalidate cached mesh outputs.
        """
        if self.per_frame_params is None:
            return True
        cfg = self.per_frame_params.config
        return not cfg.optimize_pose and not cfg.optimize_betas

    def _build_aiap_neighbors(self) -> None:
        """Build k-NN neighbor cache for AIAP loss from canonical positions."""
        with torch.no_grad():
            canonical_positions = self.avatar.get_canonical_positions()
        self.loss_fn.initialize_aiap(canonical_positions)

    def _in_densification_window(self, iteration: int) -> bool:
        """Check if iteration is within densification schedule window."""
        cfg = self.config.densification
        return cfg.start_iter <= iteration <= cfg.stop_iter

    def _should_densify(self, iteration: int) -> bool:
        """Check if densification should run at this iteration."""
        if not self._in_densification_window(iteration):
            return False
        cfg = self.config.densification
        return (iteration - cfg.start_iter) % cfg.interval == 0

    def _should_reset_opacity(self, iteration: int) -> bool:
        """Check if opacity reset should run at this iteration."""
        if not self._in_densification_window(iteration):
            return False
        cfg = self.config.densification
        # Guard against firing at start_iter where (start - start) % interval == 0.
        # At start_iter, densification runs first and creates new Gaussians;
        # resetting opacity immediately would clobber their initial values.
        if iteration == cfg.start_iter:
            return False
        return (iteration - cfg.start_iter) % cfg.opacity_reset_interval == 0

    def _should_run_at_interval(self, iteration: int, interval: int) -> bool:
        """Check if an action should run at this iteration based on interval."""
        return interval > 0 and iteration % interval == 0

    def _should_update_neighbors(self, iteration: int) -> bool:
        """Check if AIAP neighbors should be updated at this iteration."""
        return self._should_run_at_interval(iteration, self.config.neighbor_update_interval)

    def _should_save_images(self, iteration: int) -> bool:
        """Check if images should be saved at this iteration."""
        if self._should_run_at_interval(iteration, self.config.save_images_interval):
            return True
        # Save every iteration in the last N iterations
        n = self.config.save_all_last_n
        if n > 0 and hasattr(self, "_num_iterations"):
            return iteration >= self._num_iterations - n
        return False

    def _should_run_ssl(self, iteration: int) -> bool:
        """Check if SSL loss should be computed at this iteration."""
        cfg = self.config.ssl
        if not cfg.enabled or cfg.interval <= 0:
            return False
        # Skip SSL entirely when all SSL loss weights are zero
        weights = self.config.loss_weights
        if (
            weights.lambda_ssl_aiap_position == 0
            and weights.lambda_ssl_aiap_covariance == 0
            and weights.lambda_ssl_silhouette == 0
        ):
            return False
        if iteration < cfg.start_iter or iteration > cfg.stop_iter:
            return False
        return iteration % cfg.interval == 0

    def _compute_ssl_loss(
        self,
        pose: Tensor,
        betas: Tensor,
        intrinsic: Tensor,
        extrinsic: Tensor,
        H: int,
        W: int,
        alpha: float | None,
        trans: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute SSL loss for one sample with perturbed pose and jittered camera.

        Args:
            pose: (P,) pose parameters (detached from per-frame optimization)
            betas: (S,) shape coefficients
            intrinsic: (3, 3) camera intrinsic
            extrinsic: (4, 4) camera extrinsic
            H: Image height
            W: Image width
            alpha: Frequency annealing alpha
            trans: Optional (3,) translation

        Returns:
            Dictionary with ssl_aiap_position, ssl_aiap_covariance,
            ssl_silhouette, and ssl_total losses
        """
        cfg = self.config.ssl
        weights = self.config.loss_weights

        perturbed_pose = perturb_pose(pose.detach(), cfg.pose_noise_std)
        jittered_ext = jitter_camera(
            extrinsic.detach(), cfg.camera_rotation_std, cfg.camera_translation_std
        )

        self._ssl_call_count += 1

        avatar_out = self.avatar(perturbed_pose, betas.detach(), alpha=alpha)
        if trans is not None:
            avatar_out["means"] = avatar_out["means"] + trans.detach()

        camera = Camera.from_dataset_sample(intrinsic, jittered_ext, H, W)
        render_out = self.renderer(
            means=avatar_out["means"],
            quats=avatar_out["quats"],
            scales=avatar_out["scales"],
            colors=avatar_out["colors"],
            opacities=avatar_out["opacities"],
            camera=camera,
            background=torch.zeros(3, device=self.device),
        )

        # Mesh silhouette for silhouette loss (no grad needed)
        with torch.no_grad():
            verts = self.avatar.mesh.forward(perturbed_pose, betas.detach())
            if trans is not None:
                verts = verts + trans.detach()
            faces = self.avatar.mesh.get_faces()
            mesh_mask = rasterize_mesh_silhouette(
                verts, faces, intrinsic, jittered_ext, H, W
            )

        # Save SSL renders + mesh overlay periodically
        if (
            self._ssl_renders_dir is not None
            and cfg.save_renders_interval > 0
            and self._ssl_call_count % cfg.save_renders_interval == 0
        ):
            with torch.no_grad():
                from PIL import Image

                from gaussian_avatar.training.image_saver import (
                    render_wireframe_overlay,
                )

                prefix = f"ssl_iter{self.iteration:06d}"

                rgb_np = (
                    (render_out["rgb"].clamp(0, 1) * 255)
                    .byte()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )
                Image.fromarray(rgb_np).save(
                    self._ssl_renders_dir / f"{prefix}_render.png"
                )

                overlay_img = render_wireframe_overlay(
                    image=render_out["rgb"],
                    vertices=verts,
                    faces=faces,
                    intrinsic=intrinsic,
                    extrinsic=jittered_ext,
                )
                overlay_img.save(
                    self._ssl_renders_dir / f"{prefix}_mesh_overlay.png"
                )

        aiap_pos = self.loss_fn.aiap_loss.position_loss(
            avatar_out["canonical_positions"], avatar_out["posed_positions"]
        )
        aiap_cov = self.loss_fn.aiap_loss.covariance_loss(
            avatar_out["canonical_rotations"],
            avatar_out["canonical_scales"],
            avatar_out["posed_rotations"],
            avatar_out["posed_scales"],
        )

        sil = compute_ssl_silhouette_loss(
            render_out["alpha"], mesh_mask, cfg.margin_pixels, cfg.temperature
        )

        ssl_total = (
            weights.lambda_ssl_aiap_position * aiap_pos
            + weights.lambda_ssl_aiap_covariance * aiap_cov
            + weights.lambda_ssl_silhouette * sil
        )

        return {
            "ssl_aiap_position": aiap_pos,
            "ssl_aiap_covariance": aiap_cov,
            "ssl_silhouette": sil,
            "ssl_total": ssl_total,
        }

    def _get_target_resolution(self, iteration: int) -> tuple[int, int]:
        """Get the target training resolution for the current iteration.

        Delegates to the config's resolution schedule. Returns image_size
        when no schedule is configured.

        Args:
            iteration: Current training iteration

        Returns:
            (height, width) target resolution
        """
        return self.config.get_resolution_at(iteration)

    def _resize_batch(
        self,
        batch: dict[str, Tensor],
        target_h: int,
        target_w: int,
    ) -> dict[str, Tensor]:
        """Resize batch images and masks to the target resolution.

        When the target resolution differs from the batch's image resolution,
        this method bilinearly interpolates images, nearest-neighbor interpolates
        masks, and proportionally scales camera intrinsics.

        Args:
            batch: Training batch with 'image', 'mask', and 'intrinsic' keys
            target_h: Target image height
            target_w: Target image width

        Returns:
            Batch with resized tensors (modified in-place and returned)
        """
        images = batch["image"]
        _, _, h_orig, w_orig = images.shape

        if h_orig == target_h and w_orig == target_w:
            return batch

        # Resize images: (B, 3, H, W) -> (B, 3, target_h, target_w)
        batch["image"] = F.interpolate(
            images,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Resize masks: (B, H, W) -> (B, target_h, target_w)
        masks = batch["mask"]
        batch["mask"] = F.interpolate(
            masks.unsqueeze(1),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(1)

        # Scale intrinsics proportionally
        scale_w = target_w / w_orig
        scale_h = target_h / h_orig
        intrinsics = batch["intrinsic"].clone()
        intrinsics[:, 0, 0] *= scale_w  # fx
        intrinsics[:, 1, 1] *= scale_h  # fy
        intrinsics[:, 0, 2] *= scale_w  # cx
        intrinsics[:, 1, 2] *= scale_h  # cy
        batch["intrinsic"] = intrinsics

        return batch

    def _log_losses(self, losses: dict[str, float], iteration: int) -> None:
        """Log training losses to wandb."""
        if not self.use_wandb or self.wandb is None:
            return

        log_dict = {f"train/{k}": v for k, v in losses.items()}

        # Also log key training metrics under metrics/ for easy comparison
        metrics_keys = ("total", "reconstruction", "mask", "regularization", "aiap")
        ssl_metrics_keys = (
            "ssl_total",
            "ssl_aiap_position",
            "ssl_aiap_covariance",
            "ssl_silhouette",
        )
        for k in (*metrics_keys, *ssl_metrics_keys):
            if k in losses:
                log_dict[f"metrics/train_{k}"] = losses[k]

        self.wandb.log(log_dict, step=iteration)

    def _log_densification(self, stats: dict, iteration: int) -> None:
        """Log densification statistics to wandb.

        Args:
            stats: Dictionary with 'split', 'cloned', 'pruned' counts
            iteration: Current training iteration
        """
        if not self.use_wandb or self.wandb is None:
            return

        num_gaussians = self.avatar.gaussian_model.num_gaussians

        self.wandb.log(
            {
                "densification/split": stats["split"],
                "densification/cloned": stats["cloned"],
                "densification/pruned": stats["pruned"],
                "densification/total_gaussians": num_gaussians,
            },
            step=iteration,
        )

    def _save_densification_heatmaps(
        self, stats: dict, iteration: int
    ) -> None:
        """Save per-face split/clone heatmaps on the canonical mesh.

        Args:
            stats: Densification stats dict containing split_per_face
                and clone_per_face tensors.
            iteration: Current training iteration.
        """
        if self.image_saver is None:
            return

        split_per_face = stats.get("split_per_face")
        clone_per_face = stats.get("clone_per_face")
        if split_per_face is None or clone_per_face is None:
            return

        with torch.no_grad():
            canonical_verts = self.avatar.mesh.get_canonical_vertices()
            faces = self.avatar.mesh.get_faces()

        self.image_saver.save_densification_heatmap(
            vertices=canonical_verts,
            faces=faces,
            split_per_face=split_per_face,
            clone_per_face=clone_per_face,
            iteration=iteration,
        )

    def _log_gradient_stats(self, iteration: int) -> None:
        """Log gradient accumulation statistics to wandb.

        Logs mean, std, max, and min of per-Gaussian average gradient
        magnitudes from the densification manager's accumulators.

        Args:
            iteration: Current training iteration
        """
        if not self.use_wandb or self.wandb is None:
            return

        stats = self.densification.get_gradient_stats()
        self.wandb.log(
            {f"gradients/{k}": v for k, v in stats.items()},
            step=iteration,
        )

    def _log_learning_rates(self, iteration: int) -> None:
        """Log per-parameter-group learning rates to wandb."""
        if not self.use_wandb or self.wandb is None:
            return

        lr_dict = {}
        last_lrs = self.scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            lr_dict[f"lr/{name}"] = last_lrs[i] if i < len(last_lrs) else group["lr"]

        self.wandb.log(lr_dict, step=iteration)

    def _compute_gaussian_stats(
        self,
        global_scales: Tensor | None = None,
        position_offsets: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute statistics of Gaussian scales and position offsets.

        Computes mean, max, min, std of:
        - Local linear scales (exp of log-scale params), per axis and overall
        - Position offset distances from parent triangle centroids
        - Posed offset distances (local + MLP offsets, equals global-space
          Gaussian-to-centroid distance since the local frame is orthonormal)
        - Global scales (if provided), per axis and overall

        Args:
            global_scales: Optional posed global scales (N, 3) for logging
                pose-dependent scale stats.
            position_offsets: Optional MLP position offsets (N, 3) from last
                forward pass, for computing posed offset stats.

        Returns:
            Dictionary of stat name to float value.
        """
        gm = self.avatar.gaussian_model
        stats: dict[str, float] = {}

        with torch.no_grad():
            # Position offsets from face center (local params only, no MLP)
            offsets = gm.local_positions  # (N, 3)
            offset_dists = offsets.norm(dim=-1)  # (N,)

            stats["offset_dist/mean"] = offset_dists.mean().item()
            stats["offset_dist/max"] = offset_dists.max().item()
            stats["offset_dist/min"] = offset_dists.min().item()
            stats["offset_dist/std"] = offset_dists.std().item()

            # Per-axis offset stats
            for i, axis in enumerate(("x", "y", "z")):
                vals = offsets[:, i].abs()
                stats[f"offset_{axis}/mean"] = vals.mean().item()
                stats[f"offset_{axis}/max"] = vals.max().item()
                stats[f"offset_{axis}/std"] = vals.std().item()

            # Posed offsets (local + MLP, from last forward pass)
            # Since the triangle local frame is orthonormal, the norm of
            # (local_pos + mlp_offset) equals the global-space distance
            # between the Gaussian and its parent triangle centroid.
            if position_offsets is not None:
                posed_offsets = offsets + position_offsets.detach()
                posed_dists = posed_offsets.norm(dim=-1)

                stats["posed_offset_dist/mean"] = posed_dists.mean().item()
                stats["posed_offset_dist/max"] = posed_dists.max().item()
                stats["posed_offset_dist/min"] = posed_dists.min().item()
                stats["posed_offset_dist/std"] = posed_dists.std().item()

                for i, axis in enumerate(("x", "y", "z")):
                    vals = posed_offsets[:, i].abs()
                    stats[f"posed_offset_{axis}/mean"] = vals.mean().item()
                    stats[f"posed_offset_{axis}/max"] = vals.max().item()
                    stats[f"posed_offset_{axis}/std"] = vals.std().item()

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

            # Global scales (pose-dependent, from last forward pass)
            if global_scales is not None:
                gs = global_scales.detach()
                gs_all = gs.reshape(-1)

                stats["global_scale/mean"] = gs_all.mean().item()
                stats["global_scale/max"] = gs_all.max().item()
                stats["global_scale/min"] = gs_all.min().item()
                stats["global_scale/std"] = gs_all.std().item()

                for i, axis in enumerate(("x", "y", "z")):
                    vals = gs[:, i]
                    stats[f"global_scale_{axis}/mean"] = vals.mean().item()
                    stats[f"global_scale_{axis}/max"] = vals.max().item()
                    stats[f"global_scale_{axis}/min"] = vals.min().item()
                    stats[f"global_scale_{axis}/std"] = vals.std().item()

        return stats

    def _log_gaussian_stats(self, iteration: int) -> None:
        """Log Gaussian scale and offset statistics to wandb.

        Args:
            iteration: Current training iteration
        """
        if not self.use_wandb or self.wandb is None:
            return

        global_scales = getattr(self, "_last_global_scales", None)
        stats = self._compute_gaussian_stats(global_scales=global_scales)

        # MLP position offset stats (pose-dependent δμ from last forward pass)
        avatar_out = getattr(self, "_last_avatar_output", None)
        if avatar_out is not None and "offsets" in avatar_out:
            with torch.no_grad():
                delta_pos = avatar_out["offsets"]["position"].detach()
                delta_norms = delta_pos.norm(dim=-1)
                stats["mlp_offset/mean"] = delta_norms.mean().item()
                stats["mlp_offset/max"] = delta_norms.max().item()
                stats["mlp_offset/std"] = delta_norms.std().item()
                for i, axis in enumerate(("x", "y", "z")):
                    vals = delta_pos[:, i].abs()
                    stats[f"mlp_offset_{axis}/mean"] = vals.mean().item()
                    stats[f"mlp_offset_{axis}/max"] = vals.max().item()

        self.wandb.log(
            {f"gaussians/{k}": v for k, v in stats.items()},
            step=iteration,
        )

    def _log_frequency_annealing(self, iteration: int) -> None:
        """Log frequency annealing alpha to wandb.

        Args:
            iteration: Current training iteration
        """
        if not self.use_wandb or self.wandb is None:
            return
        alpha = self.config.deformation.get_frequency_alpha(iteration)
        if alpha is not None:
            self.wandb.log({"train/frequency_alpha": alpha}, step=iteration)

    def _log_metrics(self, metrics: dict[str, float], iteration: int) -> None:
        """Log evaluation metrics to wandb."""
        if not self.use_wandb or self.wandb is None:
            return

        log_dict = {f"eval/{k}": v for k, v in metrics.items()}

        # Also log key eval metrics under metrics/ for easy comparison
        eval_keys = ("psnr", "ssim", "lpips")
        for k in eval_keys:
            if k in metrics:
                log_dict[f"metrics/eval_{k}"] = metrics[k]

        self.wandb.log(log_dict, step=iteration)

    def _save_training_images(
        self,
        render_output: dict[str, Tensor],
        target_rgb: Tensor,
        target_mask: Tensor | None,
        iteration: int,
        force_gt: bool = False,
    ) -> None:
        """Save training images if image saver is configured."""
        if self.image_saver is None:
            return

        save_gt = self.config.save_gt_images or force_gt

        # Use flat directory for last-N saves
        flat = force_gt

        bg = self.renderer.background.view(3, 1, 1)

        # train_step renders with black bg (premultiplied output).
        # Composite over the renderer's configured background for display.
        rgb = render_output["rgb"]
        alpha = render_output.get("alpha")
        if alpha is not None:
            rgb = rgb + bg * (1.0 - alpha.unsqueeze(0))

        # Composite GT over same background to replace garbage values
        # in transparent regions (e.g. green from corrected dataset PNGs).
        gt = target_rgb
        if save_gt and gt is not None and target_mask is not None:
            mask3 = target_mask.unsqueeze(0)  # (1, H, W)
            gt = gt * mask3 + bg * (1.0 - mask3)

        self.image_saver.save_training_image(
            rendered_rgb=rgb,
            rendered_alpha=alpha,
            target_rgb=gt if save_gt else None,
            iteration=iteration,
            flat=flat,
        )

        # Save LPIPS and SSIM heatmaps during the last N iterations
        if force_gt and gt is not None:
            rgb_clamped = rgb.clamp(0.0, 1.0)
            last_n_dir = self.image_saver.output_dir / "last_n"
            last_n_dir.mkdir(exist_ok=True)
            prefix = f"iter_{iteration:06d}_"
            with torch.no_grad():
                if self._lpips_metric is None:
                    self._lpips_metric = LPIPSMetric(device=str(self.device))
                lpips_map = self._lpips_metric.spatial_map(rgb_clamped, gt)
                self.image_saver.save_heatmap(
                    lpips_map,
                    last_n_dir / f"{prefix}lpips_heatmap.png",
                    vmax=ImageSaver.LPIPS_VMAX,
                )
                ssim_error = compute_ssim_map(rgb_clamped, gt)
                self.image_saver.save_heatmap(
                    ssim_error,
                    last_n_dir / f"{prefix}ssim_heatmap.png",
                    vmax=ImageSaver.SSIM_VMAX,
                )

        if save_gt and self._last_pose is not None:
            with torch.no_grad():
                posed_verts = self.avatar.mesh.forward(
                    self._last_pose, self._last_betas
                )
                if self._last_trans is not None:
                    posed_verts = posed_verts + self._last_trans

            if flat:
                last_n_dir = self.image_saver.output_dir / "last_n"
                last_n_dir.mkdir(exist_ok=True)
                prefix = f"iter_{iteration:06d}_"
                overlay_path = last_n_dir / f"{prefix}gt_mesh_overlay.png"
            else:
                iter_dir = self.image_saver.output_dir / f"iter_{iteration:06d}"
                iter_dir.mkdir(exist_ok=True)
                overlay_path = iter_dir / "gt_mesh_overlay.png"
            self.image_saver.save_mesh_overlay(
                target_rgb=gt,
                vertices=posed_verts,
                faces=self.avatar.mesh.get_faces(),
                intrinsic=self._last_intrinsic,
                extrinsic=self._last_extrinsic,
                path=overlay_path,
            )

        # Save MLP position offset heatmaps
        avatar_out = getattr(self, "_last_avatar_output", None)
        if avatar_out is not None and "offsets" in avatar_out:
            position_offsets = avatar_out["offsets"]["position"]
            H, W = render_output["rgb"].shape[1:]
            with torch.no_grad():
                self.image_saver.save_offset_mesh_heatmap(
                    vertices=self.avatar.mesh.get_canonical_vertices(),
                    faces=self.avatar.mesh.get_faces(),
                    triangle_idx=self.avatar.gaussian_model.triangle_indices,
                    position_offsets=position_offsets,
                    iteration=iteration,
                )
                self.image_saver.save_projected_offset_heatmap(
                    means=avatar_out["means"],
                    position_offsets=position_offsets,
                    intrinsic=self._last_intrinsic,
                    extrinsic=self._last_extrinsic,
                    height=H,
                    width=W,
                    iteration=iteration,
                )

    def _update_aiap_neighbors(self) -> None:
        """Re-initialize AIAP neighbors with current Gaussian positions."""
        self._build_aiap_neighbors()

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Execute single training iteration.

        Processes the batch by iterating over each sample (since rendering
        is per-image), computing losses, and accumulating gradients.
        When ``gradient_accumulation_steps > 1``, the optimizer only zeros
        gradients and steps every N calls, controlled by ``self.iteration``.

        Args:
            batch: Dictionary containing:
                - 'pose': (B, P) pose parameters (P depends on model type)
                - 'betas': (B, S) shape coefficients (S depends on model type)
                - 'image': (B, 3, H, W) target RGB images
                - 'mask': (B, H, W) target masks
                - 'intrinsic': (B, 3, 3) camera intrinsics
                - 'extrinsic': (B, 4, 4) camera extrinsics

        Returns:
            Dictionary of loss values (floats) for logging
        """
        accum_steps = self.config.gradient_accumulation_steps
        is_accumulation_boundary = (self.iteration % accum_steps == 0)

        if is_accumulation_boundary:
            self.optimizer.zero_grad(set_to_none=True)

        target_h, target_w = self._get_target_resolution(self.iteration)
        batch = self._resize_batch(batch, target_h, target_w)

        if self.per_frame_params is not None:
            batch = self.per_frame_params.apply_to_batch(batch)

        pose = batch["pose"]
        betas = batch["betas"]
        trans = batch.get("trans")
        target_rgb = batch["image"]
        target_mask = batch["mask"]
        intrinsic = batch["intrinsic"]
        extrinsic = batch["extrinsic"]

        B = pose.shape[0]
        H, W = target_rgb.shape[2:]

        # Compute frequency annealing alpha for positional encoding
        alpha = self.config.deformation.get_frequency_alpha(self.iteration)

        all_losses: list[dict[str, Tensor]] = []
        all_render_outputs: list[dict[str, Tensor]] = []
        all_avatar_outputs: list[dict[str, Tensor]] = []

        for b in range(B):
            cache_entry = None
            if self._mesh_cache is not None:
                subject_b = batch["subject"][b]
                frame_idx_b = batch["frame_idx"][b]
                if isinstance(frame_idx_b, Tensor):
                    frame_idx_b = frame_idx_b.item()
                cache_entry = self._mesh_cache.get(subject_b, frame_idx_b)

            if cache_entry is not None:
                avatar_output = self.avatar.forward_with_cache(
                    posed_transforms=cache_entry.posed_transforms,
                    stretches=cache_entry.stretches,
                    pose_6d=cache_entry.pose_6d,
                    betas=betas[b],
                    alpha=alpha,
                )
            else:
                avatar_output = self.avatar(pose[b], betas[b], alpha=alpha)
            all_avatar_outputs.append(avatar_output)
            if trans is not None:
                avatar_output["means"] = avatar_output["means"] + trans[b]

            camera = Camera.from_dataset_sample(
                intrinsic=intrinsic[b],
                extrinsic=extrinsic[b],
                height=H,
                width=W,
            )

            # Render with black background so output is premultiplied-alpha.
            # TotalLoss handles compositing over the configured background.
            # Using the renderer's own background here would double-composite.
            render_output = self.renderer(
                means=avatar_output["means"],
                quats=avatar_output["quats"],
                scales=avatar_output["scales"],
                colors=avatar_output["colors"],
                opacities=avatar_output["opacities"],
                camera=camera,
                background=torch.zeros(3, device=self.device),
                absgrad=self.config.densification.use_absgrad,
            )

            all_render_outputs.append(render_output)

            rendered_rgb = render_output["rgb"].unsqueeze(0)  # (1, 3, H, W)
            rendered_alpha = render_output["alpha"].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            target_rgb_b = target_rgb[b : b + 1]  # (1, 3, H, W)
            target_mask_b = target_mask[b : b + 1].unsqueeze(1)  # (1, 1, H, W)

            losses = self.loss_fn(
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
                raw_opacities=self.avatar.gaussian_model.raw_opacities,
            )

            all_losses.append(losses)

        total_loss = sum(l["total"] for l in all_losses) / B / accum_steps

        # SSL step: perturbed pose + jittered camera
        ssl_avg_losses: dict[str, float] = {}
        if self._should_run_ssl(self.iteration):
            ssl_total = torch.tensor(0.0, device=self.device)
            ssl_accum: dict[str, float] = {}
            for b in range(B):
                ssl_losses = self._compute_ssl_loss(
                    pose[b], betas[b], intrinsic[b], extrinsic[b],
                    H, W, alpha,
                    trans=trans[b] if trans is not None else None,
                )
                ssl_total = ssl_total + ssl_losses["ssl_total"]
                for k, v in ssl_losses.items():
                    ssl_accum[k] = ssl_accum.get(k, 0.0) + v.item()
            total_loss = total_loss + ssl_total / B / accum_steps
            ssl_avg_losses = {k: v / B for k, v in ssl_accum.items()}

        total_loss.backward()

        # Accumulate viewspace gradients after backward so .grad is populated
        for ro in all_render_outputs:
            self.densification.accumulate_gradients(
                ro["viewspace_points"],
                ro["visibility_filter"],
                means2d=ro.get("_means2d"),
                use_absgrad=self.config.densification.use_absgrad,
            )

        is_step_boundary = ((self.iteration + 1) % accum_steps == 0)

        if is_step_boundary:
            if self.config.grad_clip > 0:
                all_params = list(self.avatar.parameters())
                if self.per_frame_params is not None:
                    all_params.extend(self.per_frame_params.parameters())
                nn.utils.clip_grad_norm_(
                    all_params,
                    self.config.grad_clip,
                )

            self.optimizer.step()

        self._last_render_output = all_render_outputs[-1] if all_render_outputs else None
        self._last_target_rgb = target_rgb[-1] if B > 0 else None
        self._last_target_mask = target_mask[-1] if B > 0 else None
        self._last_global_scales = all_avatar_outputs[-1]["scales"].detach() if all_avatar_outputs else None
        self._last_avatar_output = all_avatar_outputs[-1] if all_avatar_outputs else None
        self._last_pose = pose[-1]
        self._last_betas = betas[-1]
        self._last_trans = trans[-1] if trans is not None else None
        self._last_intrinsic = intrinsic[-1]
        self._last_extrinsic = extrinsic[-1]

        avg_losses: dict[str, float] = {}
        for key in all_losses[0].keys():
            avg_losses[key] = sum(l[key].item() for l in all_losses) / B
        avg_losses.update(ssl_avg_losses)

        return avg_losses

    def train(self, num_iterations: int) -> None:
        """Run training loop.

        Args:
            num_iterations: Total number of training iterations to run
        """
        self._num_iterations = num_iterations
        data_iter = iter(self.train_loader)

        self._build_aiap_neighbors()

        if self._can_use_mesh_cache():
            self._mesh_cache = MeshCache()
            self._mesh_cache.populate(
                self.avatar, self.train_loader.dataset, self.device
            )

        current_resolution = self._get_target_resolution(self.iteration)
        logger.info(
            "Starting training at resolution %dx%d",
            current_resolution[1],
            current_resolution[0],
        )

        pbar = tqdm(
            range(self.iteration, num_iterations),
            initial=self.iteration,
            total=num_iterations,
            desc="Training",
            dynamic_ncols=True,
        )

        for iteration in pbar:
            self.iteration = iteration

            new_resolution = self._get_target_resolution(iteration)
            if new_resolution != current_resolution:
                logger.info(
                    "Iteration %d: changing resolution from %dx%d to %dx%d",
                    iteration,
                    current_resolution[1],
                    current_resolution[0],
                    new_resolution[1],
                    new_resolution[0],
                )
                if self.use_wandb and self.wandb is not None:
                    self.wandb.log(
                        {
                            "train/resolution_h": new_resolution[0],
                            "train/resolution_w": new_resolution[1],
                        },
                        step=iteration,
                    )
                current_resolution = new_resolution

            if self._current_sh_degree is not None:
                new_sh_degree = self.config.get_sh_degree_at(iteration)
                if new_sh_degree != self._current_sh_degree:
                    logger.info(
                        "Iteration %d: changing SH degree from %d to %d",
                        iteration,
                        self._current_sh_degree,
                        new_sh_degree,
                    )
                    if self.use_wandb and self.wandb is not None:
                        self.wandb.log(
                            {"train/sh_degree": new_sh_degree},
                            step=iteration,
                        )
                    self._current_sh_degree = new_sh_degree
                    self.renderer.sh_degree = new_sh_degree

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = self._move_batch_to_device(batch)
            losses = self.train_step(batch)

            num_gs = self.avatar.gaussian_model.num_gaussians
            pbar.set_postfix(
                loss=f"{losses['total']:.4f}",
                psnr=f"{losses.get('psnr', 0):.1f}" if "psnr" in losses else None,
                gs=f"{num_gs:,}",
            )

            if iteration % self.config.log_interval == 0:
                self._log_losses(losses, iteration)
                self._log_gradient_stats(iteration)
                self._log_learning_rates(iteration)
                self._log_gaussian_stats(iteration)
                self._log_frequency_annealing(iteration)

            if self._should_densify(iteration):
                densify_stats = self.densification.densify(self.optimizer)
                self._log_densification(densify_stats, iteration)
                self._save_densification_heatmaps(densify_stats, iteration)
                self._update_aiap_neighbors()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self._should_reset_opacity(iteration):
                self.densification.reset_opacity()

            accum_steps = self.config.gradient_accumulation_steps
            if (iteration + 1) % accum_steps == 0:
                self.scheduler.step()

            if self._should_save_images(iteration):
                # Force GT saving during the last N iterations
                force_gt = (
                    self.config.save_all_last_n > 0
                    and iteration >= num_iterations - self.config.save_all_last_n
                )
                self._save_training_images(
                    self._last_render_output,
                    self._last_target_rgb,
                    self._last_target_mask,
                    iteration,
                    force_gt=force_gt,
                )

            if self._should_update_neighbors(iteration) and iteration > 0:
                self._update_aiap_neighbors()

            if iteration > 0 and iteration % self.config.eval_interval == 0:
                if self.image_saver is not None:
                    self.image_saver.flush()
                metrics = self.evaluate()
                self._log_metrics(metrics, iteration)

            if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
                if self.image_saver is not None:
                    self.image_saver.flush()
                self.save_checkpoint()

        pbar.close()
        self.iteration = num_iterations

        # Always save final checkpoint and run evaluation after training
        logger.info("Training complete. Saving final checkpoint and running evaluation.")
        if self.image_saver is not None:
            self.image_saver.flush()
        self.save_checkpoint()
        metrics = self.evaluate()
        self._log_metrics(metrics, self.iteration)
        if self.image_saver is not None:
            self.image_saver.shutdown()

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to the trainer's device.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Batch with tensors moved to device
        """
        moved = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on test set.

        Computes PSNR, SSIM, and LPIPS metrics averaged over all test samples.
        When ``config.eval_max_samples`` is set, a random subset of that size
        is evaluated instead of the full test set.
        Optionally saves evaluation images.

        Returns:
            Dictionary with mean PSNR, SSIM, LPIPS values
        """
        self.avatar.eval()

        if self._lpips_metric is None:
            self._lpips_metric = LPIPSMetric(device=str(self.device))
        lpips_metric = self._lpips_metric

        psnrs: list[float] = []
        ssims: list[float] = []
        lpips_vals: list[float] = []
        per_image_metrics: list[dict] = []
        global_scale_samples: list[Tensor] = []
        images_saved = 0

        eval_image_saver = None
        if self.config.num_eval_images_to_save > 0:
            eval_dir = self.checkpoint_dir.parent / "eval_images" / f"iter_{self.iteration:06d}"
            eval_image_saver = ImageSaver(output_dir=eval_dir, save_gt=True)

        # Use a subset of the test set when eval_max_samples is configured
        eval_loader = self.test_loader
        max_samples = self.config.eval_max_samples
        if max_samples is not None and hasattr(self.test_loader, "dataset"):
            dataset = self.test_loader.dataset
            n = len(dataset)
            if max_samples < n:
                indices = torch.randperm(n)[:max_samples].tolist()
                subset = torch.utils.data.Subset(dataset, indices)
                eval_loader = DataLoader(
                    subset,
                    batch_size=self.test_loader.batch_size,
                    shuffle=False,
                    num_workers=min(2, self.test_loader.num_workers),
                    pin_memory=self.test_loader.pin_memory,
                    collate_fn=self.test_loader.collate_fn,
                )
                logger.info(
                    "Partial eval: sampling %d / %d test images",
                    max_samples, n,
                )

        for batch in eval_loader:
            batch = self._move_batch_to_device(batch)

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
                avatar_output = self.avatar(pose[b], betas[b])
                if trans is not None:
                    avatar_output["means"] = avatar_output["means"] + trans[b]

                camera = Camera.from_dataset_sample(
                    intrinsic=intrinsic[b],
                    extrinsic=extrinsic[b],
                    height=H,
                    width=W,
                )

                render_output = self.renderer(
                    means=avatar_output["means"],
                    quats=avatar_output["quats"],
                    scales=avatar_output["scales"],
                    colors=avatar_output["colors"],
                    opacities=avatar_output["opacities"],
                    camera=camera,
                )

                pred_rgb = render_output["rgb"]
                pred_alpha = render_output["alpha"]
                gt_rgb = target_rgb[b]
                gt_mask = target_mask[b]

                # Composite GT over the same background as the renderer
                # so metrics are not penalised by background mismatch.
                # pred_rgb is already composited over the renderer's
                # background, so do NOT multiply by alpha again.
                bg = self.renderer.background.view(3, 1, 1)
                mask3 = gt_mask.unsqueeze(0)
                gt_rgb = gt_rgb * mask3 + bg * (1 - mask3)

                frame = frame_idx[b] if hasattr(frame_idx, '__getitem__') else frame_idx
                subj = subject[b] if hasattr(subject, '__getitem__') else subject

                # Clamp to [0,1] — gsplat accumulation can overshoot
                # by tiny amounts (e.g. 1.001) due to floating-point.
                pred_rgb = pred_rgb.clamp(0.0, 1.0)

                frame_psnr = compute_psnr(pred_rgb, gt_rgb)
                frame_ssim = compute_ssim(pred_rgb, gt_rgb)
                frame_lpips = lpips_metric(pred_rgb, gt_rgb)

                psnrs.append(frame_psnr)
                ssims.append(frame_ssim)
                lpips_vals.append(frame_lpips)
                global_scale_samples.append(avatar_output["scales"])

                per_image_metrics.append({
                    "frame_idx": int(frame),
                    "subject": subj if subj else None,
                    "psnr": frame_psnr,
                    "ssim": frame_ssim,
                    "lpips": frame_lpips,
                })

                if (
                    eval_image_saver is not None
                    and images_saved < self.config.num_eval_images_to_save
                ):
                    eval_image_saver.save_evaluation_image(
                        rendered_rgb=pred_rgb,
                        target_rgb=gt_rgb,
                        frame_idx=frame,
                        subject=subj,
                    )

                    # LPIPS spatial heatmap
                    lpips_map = lpips_metric.spatial_map(pred_rgb, gt_rgb)
                    eval_image_saver.save_lpips_heatmap(
                        lpips_map=lpips_map,
                        frame_idx=frame,
                        subject=subj,
                    )

                    # SSIM spatial error heatmap
                    ssim_error = compute_ssim_map(pred_rgb, gt_rgb)
                    eval_image_saver.save_ssim_heatmap(
                        ssim_error_map=ssim_error,
                        frame_idx=frame,
                        subject=subj,
                    )

                    posed_verts = self.avatar.mesh.forward(pose[b], betas[b])
                    if trans is not None:
                        posed_verts = posed_verts + trans[b]
                    prefix = f"frame_{frame:04d}"
                    if subj:
                        prefix = f"{subj}_{prefix}"
                    eval_image_saver.save_mesh_overlay(
                        target_rgb=gt_rgb,
                        vertices=posed_verts,
                        faces=self.avatar.mesh.get_faces(),
                        intrinsic=intrinsic[b],
                        extrinsic=extrinsic[b],
                        path=eval_dir / f"{prefix}_gt_mesh_overlay.png",
                    )

                    eval_image_saver.save_projected_density(
                        means=avatar_output["means"],
                        intrinsic=intrinsic[b],
                        extrinsic=extrinsic[b],
                        height=H,
                        width=W,
                        frame_idx=frame,
                        subject=subj,
                    )

                    images_saved += 1

        # Save canonical mesh density heatmap (once per evaluation)
        if eval_image_saver is not None:
            eval_image_saver.save_canonical_density(
                vertices=self.avatar.mesh.get_canonical_vertices(),
                faces=self.avatar.mesh.get_faces(),
                triangle_idx=self.avatar.gaussian_model.triangle_indices,
                num_faces=self.avatar.mesh.get_faces().shape[0],
            )

        if eval_image_saver is not None and per_image_metrics:
            metrics_json = {
                "iteration": self.iteration,
                "per_image": per_image_metrics,
                "mean": {
                    "psnr": sum(m["psnr"] for m in per_image_metrics) / len(per_image_metrics),
                    "ssim": sum(m["ssim"] for m in per_image_metrics) / len(per_image_metrics),
                    "lpips": sum(m["lpips"] for m in per_image_metrics) / len(per_image_metrics),
                },
            }
            metrics_path = eval_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_json, f, indent=2)

        self.avatar.train()

        # Compute Gaussian stats using mean of global scales across eval poses
        avg_global_scales = (
            torch.stack(global_scale_samples).mean(dim=0)
            if global_scale_samples
            else None
        )
        gaussian_stats = self._compute_gaussian_stats(global_scales=avg_global_scales)

        metrics = {
            "psnr": sum(psnrs) / len(psnrs) if psnrs else 0.0,
            "ssim": sum(ssims) / len(ssims) if ssims else 0.0,
            "lpips": sum(lpips_vals) / len(lpips_vals) if lpips_vals else 0.0,
        }
        metrics.update({f"gaussians/{k}": v for k, v in gaussian_stats.items()})
        return metrics

    def save_checkpoint(self, path: str | Path | None = None) -> None:
        """Save model and optimizer state.

        Args:
            path: Checkpoint file path. If None, uses default naming
                  based on current iteration in checkpoint_dir.
        """
        if path is None:
            path = self.checkpoint_dir / f"iter_{self.iteration:06d}.pt"

        model_type = getattr(self.avatar.mesh, "model_type", None)
        if not isinstance(model_type, str):
            model_type = None
        lod = getattr(self.avatar.mesh, "lod", None)
        if not isinstance(lod, int):
            lod = None

        training_betas = self._extract_training_betas()

        _save_checkpoint(
            path=path,
            avatar=self.avatar,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            iteration=self.iteration,
            config=self.config,
            densification=self.densification,
            per_frame_params=self.per_frame_params,
            model_type=model_type,
            lod=lod,
            training_betas=training_betas,
        )

    def _extract_training_betas(self) -> dict[str, list[float]] | None:
        """Extract per-subject shape coefficients from the training dataset.

        Returns:
            Dict mapping subject name to betas as a list of floats,
            or None if the dataset doesn't support get_subject_data.
        """
        dataset = self.train_loader.dataset
        subjects = getattr(dataset, "_subjects", None)
        if subjects is None or not hasattr(dataset, "get_subject_data"):
            return None

        betas_dict: dict[str, list[float]] = {}
        for subject in subjects:
            subject_data = dataset.get_subject_data(subject)
            betas_dict[subject] = subject_data.betas.tolist()
        return betas_dict

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model and optimizer state.

        Validates that the checkpoint's model_type matches the current
        avatar's mesh model_type.

        Args:
            path: Checkpoint file path
        """
        expected_type = getattr(self.avatar.mesh, "model_type", None)
        if not isinstance(expected_type, str):
            expected_type = None

        checkpoint_info = _load_checkpoint(
            path=path,
            avatar=self.avatar,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            densification=self.densification,
            per_frame_params=self.per_frame_params,
            expected_model_type=expected_type,
        )
        self.iteration = checkpoint_info["iteration"]

        # Verify scheduler and iteration counter are in sync
        if self.scheduler.last_epoch != self.iteration:
            logger.warning(
                "Scheduler last_epoch (%d) != iteration (%d), "
                "resynchronizing scheduler",
                self.scheduler.last_epoch,
                self.iteration,
            )
            self.scheduler.last_epoch = self.iteration
