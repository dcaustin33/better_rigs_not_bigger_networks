"""Configuration dataclass for the evaluation pipeline.

This module provides:
- EvalDataConfig: Data loading settings for evaluation (subjects, frame range,
  skip stride, dataset format).
- EvaluationConfig: Complete evaluation configuration including data, model,
  output, and W&B settings. Mirrors the CLI arguments in scripts/evaluate.py
  so that evaluations can be fully specified via YAML.
"""

from dataclasses import dataclass, field

from gaussian_avatar.configs.deformation import DeformationConfig
from gaussian_avatar.losses.config import LossThresholds, LossWeights

@dataclass
class BodyParamOptimConfig:
    """Configuration for pre-evaluation body model parameter optimization.

    When enabled, an optimization phase runs before evaluation that freezes
    the avatar model and optimizes per-frame body parameters (pose, betas,
    translation) on the test set. This corrects noisy/imprecise fits
    to improve alignment at evaluation time. Works with SMPL, SMPL-X,
    and MHR models.

    Args:
        enabled: Whether to run body parameter optimization before evaluation.
        num_iterations: Number of optimization iterations (batch iterations,
            cycling through test data).
        optimize_pose: Optimize per-frame pose offsets.
        optimize_betas: Optimize per-subject betas offsets.
        optimize_trans: Optimize per-frame translation offsets.
        lr_pose: Learning rate for pose offsets.
        lr_betas: Learning rate for betas offsets.
        lr_trans: Learning rate for translation offsets.
        loss_weights: Loss weights for the optimization. When ``None``,
            inherits from the training checkpoint config.
        loss_thresholds: Loss thresholds for the optimization. When ``None``,
            inherits from the training checkpoint config.
    """

    enabled: bool = False
    num_iterations: int = 100
    optimize_pose: bool = True
    optimize_betas: bool = False
    optimize_trans: bool = True
    lr_pose: float = 1e-4
    lr_betas: float = 1e-5
    lr_trans: float = 1e-3
    loss_weights: LossWeights | None = None
    loss_thresholds: LossThresholds | None = None

@dataclass
class EvalDataConfig:
    """Configuration for evaluation data loading.

    Groups all dataset-related settings so they can live under a ``data:``
    key in the YAML config, mirroring the training config structure.

    Args:
        data_root: Path to the dataset root directory.
        subjects: Subject names to evaluate on. When ``None``, uses the
            subjects stored in the checkpoint config.
        dataset_format: Dataset format: ``"auto"``, ``"public"``, or
            ``"corrected"``.
        start_frame: Frame index to start evaluation at. Evaluates all
            frames from this index onwards. ``None`` means start of last
            25% of video (matching test split).
        stop_frame: Frame index separating train/test. When set,
            evaluation starts at this frame. Overridden by ``start_frame``
            if both are provided.
        skip: Frame skip stride for evaluation. E.g. ``skip=4`` evaluates
            every 4th test frame for faster evaluation. ``None`` means
            evaluate every frame (or inherit from checkpoint config).
        undistort: Whether to undistort images using camera distortion
            coefficients before evaluation.
        preprocess_masks: Apply erosion and Gaussian blur to masks before
            use. Matches the training default (``True``).
        num_workers: Number of DataLoader worker processes.
    """

    data_root: str | None = None
    subjects: list[str] | None = None
    dataset_format: str = "auto"
    start_frame: int | None = None
    stop_frame: int | None = None
    skip: int | None = None
    undistort: bool = False
    preprocess_masks: bool = True
    num_workers: int = 4
    train_cameras: list[str] | None = None

@dataclass
class EvaluationConfig:
    """Complete evaluation configuration.

    Groups all settings for running model evaluation so they can be specified
    in a YAML file under an ``evaluation:`` top-level key. CLI arguments
    override values loaded from YAML.

    Args:
        checkpoint: Path to the trained model checkpoint (.pt file).
        data: Data loading configuration (subjects, frame range, skip).
        image_size: Evaluation image size as ``(height, width)``.
        batch_size: Batch size for evaluation.
        model_path: Path to SMPL/SMPL-X model files root directory.
        model_type: Model type (``"smpl"``, ``"smplx"``, or ``"mhr"``).
        gender: SMPL/SMPL-X gender (``"male"``, ``"female"``, ``"neutral"``).
            Ignored for MHR.
        lod: MHR level-of-detail (0-6). Only used when ``model_type="mhr"``.
        num_gaussians: Number of Gaussians. When ``None``, defaults to
            the number of mesh faces.
        disable_offsets: When true, skip the deformation MLP entirely.
        max_stretch: Maximum allowed per-axis stretch factor when transforming
            Gaussians from canonical to posed space. Stretches are clamped to
            [1/max_stretch, max_stretch]. Default 2.0.
        deformation: Deformation MLP architecture config. Must match the
            architecture used during training for checkpoint loading.
        body_optim: Pre-evaluation body parameter optimization config.
        output_dir: Directory for evaluation outputs.
        save_images: Whether to save evaluation images.
        num_images: Number of evaluation images to save.
        diagnostic: Save diagnostic images (error heatmaps, alpha-vs-mask,
            boundary analysis).
        num_diagnostic: Number of diagnostic image sets to save.
        wandb: Log evaluation metrics to Weights & Biases.
        wandb_name: W&B run name. Defaults to checkpoint filename.
        wandb_project: W&B project name.
    """

    checkpoint: str | None = None

    data: EvalDataConfig = field(default_factory=EvalDataConfig)
    image_size: tuple[int, int] = (512, 512)
    batch_size: int = 1

    model_path: str = "mesh_models"
    model_type: str = "smpl"
    gender: str = "male"
    lod: int = 1
    num_gaussians: int | None = None
    disable_offsets: bool = False
    max_stretch: float = 10.0
    max_local_offset: float | None = None
    use_sh: bool = False
    max_sh_degree: int = 3

    deformation: DeformationConfig = field(default_factory=DeformationConfig)

    body_optim: BodyParamOptimConfig = field(default_factory=BodyParamOptimConfig)

    lpips_net: str = "vgg"

    background_color: tuple[float, float, float] | None = None

    output_dir: str = "output/eval"
    save_images: bool = False
    num_images: int = 10
    diagnostic: bool = False
    num_diagnostic: int = 10

    wandb: bool = False
    wandb_name: str | None = None
    wandb_project: str = "gaussian-avatar-eval"
