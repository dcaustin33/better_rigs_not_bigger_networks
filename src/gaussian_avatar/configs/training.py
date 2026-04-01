"""Configuration dataclasses for training pipeline.

This module provides:
- DataConfig: Dataset and data loading configuration
- DensificationConfig: Densification schedule and thresholds
- ResolutionScheduleEntry: A single (iteration, resolution) entry for progressive training
- TrainingConfig: Complete training configuration including LR, scheduling, and nested configs
"""

from dataclasses import dataclass, field

from gaussian_avatar.configs.deformation import DeformationConfig
from gaussian_avatar.configs.per_frame_params import PerFrameParamsConfig
from gaussian_avatar.losses.config import LossWeights, LossThresholds

@dataclass
class SSLConfig:
    """Self-supervised loss configuration.

    Controls pose perturbation and camera jitter for SSL regularization
    that improves generalization to novel poses.

    Args:
        enabled: Whether SSL loss is active
        interval: Run SSL every N iterations (0 = disabled)
        start_iter: First iteration to run SSL
        stop_iter: Last iteration to run SSL
        pose_noise_std: Std of Gaussian noise added to pose (radians)
        camera_rotation_std: Std of camera rotation jitter (radians)
        camera_translation_std: Std of camera translation jitter (meters)
        margin_pixels: Slack zone outside mesh boundary for silhouette loss
        temperature: Sharpness of the soft silhouette falloff
        save_renders_interval: Save SSL renders every N SSL calls (0 = disabled)
        save_renders_dir: Directory for saved renders (relative to output dir; None = disabled)
    """

    enabled: bool = False
    interval: int = 10
    start_iter: int = 1000
    stop_iter: int = 30000
    pose_noise_std: float = 0.15
    camera_rotation_std: float = 0.02
    camera_translation_std: float = 0.05
    margin_pixels: float = 3.0
    temperature: float = 2.0
    save_renders_interval: int = 0
    save_renders_dir: str | None = None

@dataclass
class ModelConfig:
    """Configuration for the body model (SMPL, SMPL-X, or MHR).

    Groups model-related settings so they can live in the YAML config
    alongside training hyperparameters, removing the need for CLI-only
    ``--model-type`` / ``--lod`` flags.

    Args:
        model_type: Body model type: ``"smpl"``, ``"smplx"``, or ``"mhr"``.
        model_path: Root path to model files directory.
        gender: Gender for SMPL/SMPL-X (ignored for MHR).
        lod: MHR level-of-detail 0-6 (ignored for SMPL/SMPL-X).
        fits_model_type: Which fits directory the dataset reads from
            (``"smpl"`` → ``smpls/``, ``"smplx"`` → ``smplxs/``).
            When ``None`` (default), uses ``model_type``. Useful for
            loading SMPL pose fits with an SMPL-X mesh model.
    """

    model_type: str = "smpl"
    model_path: str = "mesh_models"
    gender: str = "male"
    lod: int = 1
    fits_model_type: str | None = None

@dataclass
class DataConfig:
    """Configuration for dataset loading and data pipeline.

    Groups all dataset-related settings so they can live in the YAML config
    alongside training hyperparameters.

    Args:
        data_root: Path to the dataset root directory (e.g.
            ``datasets/people_snapshot_public``). Required for training;
            can also be provided via ``--data-root`` CLI flag.
        subjects: List of subject names to load, or ``["all"]`` to load
            every subject discovered in data_root.
        dataset_format: Which loader to use: ``"auto"`` detects from
            directory structure, ``"public"`` for original HDF5/video
            format, ``"corrected"`` for pre-extracted PNGs with per-frame
            SMPL pickles.
        stop_frame: Optional frame index that separates training from
            testing.  Frames ``[0, stop_frame)`` are used for training
            and ``[stop_frame, N)`` for testing.  When ``None`` the
            default 75/25 split is used.  Currently only supported by
            the corrected dataset format.
        frame_indices: Optional explicit list of frame indices to use,
            completely overriding split logic.  All indices must be >= 1
            for the public format (frame 0 has no mask) and >= 0 for
            the corrected format.
        undistort: Apply lens distortion correction to images and masks
            using per-subject camera calibration parameters.
        skip: Optional frame skip stride.  When set, only every skip-th
            frame is used (e.g. ``skip=4`` keeps every 4th frame).
            Currently only supported by the corrected dataset format.
        num_workers: Number of DataLoader worker processes.
        train_cameras: Camera names for training with ZJU MoCap dataset
            (e.g. ``["Camera_B1"]``).  Test split uses all cameras NOT
            in this list.  Only used when ``dataset_format="zju_mocap"``.
        preprocess_masks: Apply erosion and Gaussian blur to masks before
            resizing.  When ``True`` (default), a 5×5 erosion followed by
            a 3×3 Gaussian blur cleans up noisy mask edges.  Set to
            ``False`` to skip erosion/blur and only resize + normalize.
    """

    data_root: str | None = None
    subjects: list[str] = field(default_factory=lambda: ["male-3-casual"])
    dataset_format: str = "auto"
    stop_frame: int | None = None
    frame_indices: list[int] | None = None
    undistort: bool = False
    skip: int | None = None
    num_workers: int = 4
    train_cameras: list[str] | None = None
    preprocess_masks: bool = True

@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Args:
        project: W&B project name.
        entity: W&B entity (username or team). None uses the default entity.
        name: Run display name in W&B. None auto-generates a name.
        enabled: Whether to enable W&B logging.
    """

    project: str = "gaussian-avatar"
    entity: str | None = None
    name: str | None = None
    enabled: bool = True

@dataclass
class ResolutionScheduleEntry:
    """A single entry in the resolution schedule.

    Defines when to switch to a new training resolution during progressive
    resolution training.

    Args:
        iteration: Training iteration at which this resolution becomes active
        resolution: Target image size (height, width) from this iteration onward
    """

    iteration: int
    resolution: tuple[int, int]

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                f"iteration must be non-negative, got {self.iteration}"
            )
        if len(self.resolution) != 2:
            raise ValueError(
                f"resolution must be (height, width), got {self.resolution}"
            )
        if self.resolution[0] <= 0 or self.resolution[1] <= 0:
            raise ValueError(
                f"resolution dimensions must be positive, got {self.resolution}"
            )

@dataclass
class SHScheduleEntry:
    """A single entry in the spherical harmonics degree schedule.

    Defines when to increase the active SH degree during progressive
    SH training. Degree 0 is DC-only (view-independent), higher degrees
    add view-dependent appearance bands.

    Args:
        iteration: Training iteration at which this SH degree becomes active
        sh_degree: SH degree to activate (0-3)
    """

    iteration: int
    sh_degree: int

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                f"iteration must be non-negative, got {self.iteration}"
            )
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError(
                f"sh_degree must be in [0, 3], got {self.sh_degree}"
            )

@dataclass
class DensificationConfig:
    """Configuration for densification operations.

    Controls the schedule and thresholds for adaptive Gaussian density
    management through split, clone, and prune operations.

    Args:
        start_iter: First iteration to run densification
        interval: Iterations between densification steps
        stop_iter: Last iteration to run densification
        grad_threshold: Minimum average gradient to trigger split/clone
        min_opacity: Minimum opacity to avoid pruning
        split_scale_threshold: Minimum scale (linear) to trigger split vs clone.
            Gaussians with high gradient AND scale > threshold are split.
            Gaussians with high gradient AND scale <= threshold are cloned.
        clone_position_noise: Std dev of position noise added to clones
        opacity_reset_interval: Iterations between opacity resets
        opacity_reset_value: Opacity cap for reset (opacities above this are clamped down)
        max_gaussians: Maximum number of Gaussians (optional limit)
        force_split_scale: If set, Gaussians whose max canonical linear
            scale (exp(local_scales).max(dim=-1)) exceeds this value are
            split unconditionally, regardless of gradient. None disables.
        protect_all_triangles: When True (default), pruning ensures every
            triangle keeps at least one Gaussian to maintain full mesh
            coverage. When False, triangles may become empty if all their
            Gaussians are pruned.
        use_absgrad: When True, use absolute-value gradients (absgrad)
            from gsplat for densification instead of regular signed
            gradients. Absgrad accumulates |dL/d(means2d)| per Gaussian,
            preventing gradient cancellation for Gaussians straddling
            edges. Requires gsplat with absgrad support. Default False
            (standard gradients).
    """

    start_iter: int = 500
    interval: int = 200
    stop_iter: int = 15000
    grad_threshold: float = 1e-7
    min_opacity: float = 0.005
    split_scale_threshold: float = 0.005
    clone_position_noise: float = 0.01
    opacity_reset_interval: int = 3000
    opacity_reset_value: float = 0.05
    max_gaussians: int | None = None
    force_split_scale: float | None = None
    protect_all_triangles: bool = True
    use_absgrad: bool = False

@dataclass
class TrainingConfig:
    """Complete training configuration.

    Combines all hyperparameters for the training loop including learning rates,
    scheduling, logging intervals, and nested configurations for densification,
    losses, and deformation.

    Args:
        num_iterations: Total number of training iterations
        batch_size: Training batch size
        gradient_accumulation_steps: Number of forward/backward passes to
            accumulate gradients over before each optimizer step. Effective
            batch size is batch_size * gradient_accumulation_steps. Must be >= 1.
        num_gaussians: Total number of Gaussians (randomly assigned to triangles)
        disable_offsets: When True, skip the deformation MLP entirely so
            training uses only the raw Gaussian properties without any
            pose-dependent corrections
        max_stretch: Maximum allowed per-axis stretch factor when transforming
            Gaussians from canonical to posed space. Stretches are clamped to
            [1/max_stretch, max_stretch]. Default 2.0.
        image_size: Base training image size (height, width) and the resolution
            at which the dataset loads images. When resolution_schedule is
            provided, this is automatically set to the maximum resolution in
            the schedule.
        resolution_schedule: List of ResolutionScheduleEntry defining when
            to switch training resolution. Each entry specifies an iteration
            and a (height, width) resolution that becomes active at that
            iteration. Entries must be sorted by iteration in ascending order.
            When empty, image_size is used for the entire training.
            When provided, image_size is auto-derived as the component-wise
            maximum of all scheduled resolutions.
        lr_position: Learning rate for Gaussian local positions
        lr_rotation: Learning rate for Gaussian local rotations
        lr_scale: Learning rate for Gaussian local scales
        lr_color: Learning rate for Gaussian base colors
        lr_opacity: Learning rate for Gaussian opacity
        lr_mlp: Learning rate for deformation MLP parameters
        lr_decay_start: Iteration to start LR decay
        lr_decay_rate: Final LR multiplier (e.g., 0.1 = decay to 10% of initial)
        lr_decay_steps: Number of iterations over which to decay
        grad_clip: Maximum gradient norm (0 = disabled)
        log_interval: Iterations between loss logging
        eval_interval: Iterations between evaluations
        checkpoint_interval: Iterations between checkpoint saves
        neighbor_update_interval: Iterations between AIAP neighbor updates
        save_images_interval: Iterations between saving training images (0 = disabled)
        save_images_dir: Directory for training images (default: output_dir/images)
        save_gt_images: Whether to save ground truth alongside renders
        save_all_last_n: Save every training image (with GT) for the last N iterations (0 = disabled)
        eval_max_samples: Maximum number of test images to evaluate during
            training-time evaluation.  When ``None`` (default), all test images
            are used.  Set to e.g. 200 for a faster partial evaluation.
        num_eval_images_to_save: Number of images to save during evaluation
        data: DataConfig for dataset loading (paths, subjects, split, etc.)
        densification: DensificationConfig for adaptive density control
        loss_weights: LossWeights for loss function weights
        loss_thresholds: LossThresholds for regularization thresholds
        deformation: DeformationConfig for MLP architecture
        per_frame_params: PerFrameParamsConfig for per-frame body model parameter optimization
    """

    num_iterations: int = 30000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_gaussians: int = 100000
    disable_offsets: bool = False
    torch_compile: bool = False
    image_size: tuple[int, int] = (512, 512)
    resolution_schedule: list[ResolutionScheduleEntry] = field(default_factory=list)

    # Triangle stretch clamping
    max_stretch: float = 2.0

    # Tanh bounding for local position offsets.
    # When set, effective local positions are max_local_offset * tanh(raw),
    # providing a hard upper bound on how far Gaussians can drift from their
    # parent triangle centroid. Units are meters (0.05 = 5 cm).
    # None disables bounding (backward compatible with existing checkpoints).
    max_local_offset: float | None = None

    use_sh: bool = False
    max_sh_degree: int = 3
    sh_schedule: list[SHScheduleEntry] = field(default_factory=list)

    lr_position: float = 0.00016
    lr_rotation: float = 0.001
    lr_scale: float = 0.005
    lr_color: float = 0.0025
    lr_sh: float = 0.0025
    lr_opacity: float = 0.05

    lr_mlp: float = 0.0005

    lr_decay_start: int = 10000
    lr_decay_rate: float = 0.1
    lr_decay_steps: int = 10000

    grad_clip: float = 1.0

    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000

    neighbor_update_interval: int = 200

    save_images_interval: int = 0
    save_images_dir: str | None = None
    save_gt_images: bool = True
    save_all_last_n: int = 0

    eval_max_samples: int | None = None

    num_eval_images_to_save: int = 5

    # Background color for compositing during training loss computation.
    # When set, this static RGB color (each in [0, 1]) is used instead of
    # a random background each step. None means random (default behavior).
    background_color: tuple[float, float, float] | None = None

    # Output paths (not hyperparameters, but convenient to keep in one config)
    output_dir: str = "output"
    experiment_name: str = "default"

    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    data: DataConfig = field(default_factory=DataConfig)
    densification: DensificationConfig = field(default_factory=DensificationConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    loss_thresholds: LossThresholds = field(default_factory=LossThresholds)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    per_frame_params: PerFrameParamsConfig = field(default_factory=PerFrameParamsConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)

    def __post_init__(self) -> None:
        """Validate configuration and auto-derive image_size from schedule."""
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )

        # Validate SH schedule
        if self.use_sh:
            if not self.sh_schedule:
                self.sh_schedule = [SHScheduleEntry(iteration=0, sh_degree=0)]
            for i in range(1, len(self.sh_schedule)):
                prev = self.sh_schedule[i - 1].iteration
                curr = self.sh_schedule[i].iteration
                if curr <= prev:
                    raise ValueError(
                        f"sh_schedule must be sorted by iteration in "
                        f"ascending order with unique iterations, but entry {i} "
                        f"(iter={curr}) <= entry {i - 1} (iter={prev})"
                    )
            max_in_schedule = max(e.sh_degree for e in self.sh_schedule)
            if max_in_schedule > self.max_sh_degree:
                raise ValueError(
                    f"sh_schedule contains degree {max_in_schedule} which "
                    f"exceeds max_sh_degree={self.max_sh_degree}"
                )

        if not self.resolution_schedule:
            return

        # Check entries are sorted by iteration
        for i in range(1, len(self.resolution_schedule)):
            prev = self.resolution_schedule[i - 1].iteration
            curr = self.resolution_schedule[i].iteration
            if curr <= prev:
                raise ValueError(
                    f"resolution_schedule must be sorted by iteration in "
                    f"ascending order with unique iterations, but entry {i} "
                    f"(iter={curr}) <= entry {i - 1} (iter={prev})"
                )

        # Auto-derive image_size as the component-wise max of all scheduled
        # resolutions. The dataset loads at this size; the trainer downsamples
        # when a smaller resolution is active.
        max_h = max(e.resolution[0] for e in self.resolution_schedule)
        max_w = max(e.resolution[1] for e in self.resolution_schedule)
        self.image_size = (max_h, max_w)

    def get_resolution_at(self, iteration: int) -> tuple[int, int]:
        """Get the target training resolution for a given iteration.

        Returns the resolution from the last schedule entry whose iteration
        is <= the given iteration. If no schedule is defined or the iteration
        is before the first entry, returns image_size.

        Args:
            iteration: Current training iteration

        Returns:
            (height, width) target resolution
        """
        if not self.resolution_schedule:
            return self.image_size

        result = self.image_size
        for entry in self.resolution_schedule:
            if entry.iteration <= iteration:
                result = entry.resolution
            else:
                break

        return result

    def get_sh_degree_at(self, iteration: int) -> int:
        """Get the active SH degree for a given iteration.

        Returns the degree from the last schedule entry whose iteration
        is <= the given iteration. If no schedule is defined or the iteration
        is before the first entry, returns 0.

        Args:
            iteration: Current training iteration

        Returns:
            Active SH degree (0-3)
        """
        if not self.sh_schedule:
            return 0

        result = 0
        for entry in self.sh_schedule:
            if entry.iteration <= iteration:
                result = entry.sh_degree
            else:
                break

        return result
