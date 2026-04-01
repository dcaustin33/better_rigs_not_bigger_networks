#!/usr/bin/env python
"""Training script for Gaussian Avatar.

This script provides the CLI entry point for training Gaussian Avatar models
on the PeopleSnapshot dataset. It handles argument parsing, component setup,
and training orchestration.

Example usage:
    # Train on single subject
    uv run python scripts/train.py \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual \\
        --output-dir output \\
        --experiment-name male3_exp1

    # Resume training
    uv run python scripts/train.py \\
        --data-root datasets/people_snapshot_public \\
        --subjects male-3-casual \\
        --resume output/male3_exp1/checkpoints/iter_10000.pt
"""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train Gaussian Avatar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to dataset directory. Overrides data.data_root in YAML config.",
    )
    data_group.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Subject names to train on. Overrides data.subjects in YAML config.",
    )
    data_group.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=None,
        help="Training image size (height width)",
    )
    data_group.add_argument(
        "--frame-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific frame indices to train on (overrides split). "
        "All indices must be >= 1. Useful for single-image overfitting.",
    )
    data_group.add_argument(
        "--dataset-format",
        type=str,
        default=None,
        choices=["auto", "public", "corrected", "zju_mocap"],
        help="Dataset format. Overrides data.dataset_format in YAML config.",
    )
    data_group.add_argument(
        "--stop-frame",
        type=int,
        default=None,
        help="Frame index separating train/test. Frames [0, stop_frame) are "
        "train, [stop_frame, N) are test. Overrides data.stop_frame in YAML.",
    )
    data_group.add_argument(
        "--skip",
        type=int,
        default=None,
        help="Frame skip stride. E.g. --skip 4 keeps every 4th frame. "
        "Overrides data.skip in YAML config.",
    )
    data_group.add_argument(
        "--resolution-schedule",
        type=int,
        nargs="+",
        default=None,
        help="Progressive resolution schedule as flat list of "
        "(iteration height width) triples. Example: "
        "--resolution-schedule 0 256 256 1000 512 512 "
        "means 256x256 for iters 0-999, then 512x512. "
        "When set, --image-size is auto-derived from the max resolution.",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model files root directory (SMPL/SMPL-X/MHR). "
        "Overrides model.model_path in YAML config.",
    )
    model_group.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["male", "female", "neutral"],
        help="SMPL/SMPL-X gender (ignored for MHR). "
        "Overrides model.gender in YAML config.",
    )
    model_group.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["smpl", "smplx", "mhr"],
        help="Model type (smpl, smplx, or mhr). "
        "Overrides model.model_type in YAML config.",
    )
    model_group.add_argument(
        "--lod",
        type=int,
        default=None,
        choices=range(7),
        metavar="LOD",
        help="MHR level-of-detail (0-6). Only used with --model-type mhr. "
        "Overrides model.lod in YAML config.",
    )
    model_group.add_argument(
        "--num-gaussians",
        type=int,
        default=None,
        help="Total number of Gaussians (randomly assigned to triangles) (default: 100000)",
    )

    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Total number of training iterations (default: 30000)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: 1)",
    )
    train_group.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers. Overrides data.num_workers in YAML.",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for checkpoints and logs. Overrides output_dir in YAML.",
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for organizing outputs. Overrides experiment_name in YAML.",
    )

    image_group = parser.add_argument_group("Image Saving")
    image_group.add_argument(
        "--save-images-interval",
        type=int,
        default=None,
        help="Save training images every N iterations (0 to disable, default: 0)",
    )
    image_group.add_argument(
        "--save-eval-images",
        action="store_true",
        help="Save images during evaluation",
    )
    image_group.add_argument(
        "--num-eval-images",
        type=int,
        default=5,
        help="Number of evaluation images to save",
    )

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name. Overrides wandb.project in YAML.",
    )
    log_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (username or team). Overrides wandb.entity in YAML.",
    )
    log_group.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run display name. Overrides wandb.name in YAML.",
    )
    log_group.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    optim_group = parser.add_argument_group("Per-frame optimization")
    optim_group.add_argument(
        "--optimize-pose",
        action="store_true",
        help="Enable per-frame pose optimization (skips mesh caching)",
    )

    config_group = parser.add_argument_group("Config")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides default TrainingConfig)",
    )
    config_group.add_argument(
        "--dump-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Dump default TrainingConfig to YAML file and exit",
    )

    resume_group = parser.add_argument_group("Resume")
    resume_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Handle --dump-config early (no heavy imports needed)
    if args.dump_config:
        from gaussian_avatar.configs import TrainingConfig
        from gaussian_avatar.configs.yaml_utils import config_to_yaml

        config_to_yaml(TrainingConfig(), args.dump_config)
        print(f"Wrote default config to {args.dump_config}")
        return

    # data-root check deferred until after config loading (may come from YAML)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Training on CPU is not recommended. CUDA is required for gsplat.")

    # Import components (deferred to allow --help without heavy imports)
    from gaussian_avatar.configs import ResolutionScheduleEntry, TrainingConfig
    from gaussian_avatar.configs.yaml_utils import (
        config_from_yaml,
        config_to_flat_dict,
    )
    from gaussian_avatar.data import create_dataloaders
    from gaussian_avatar.data.mhr_dataset import (
        create_mhr_corrected_dataloaders,
        create_mhr_dataloaders,
    )
    from gaussian_avatar.data.zju_mocap import create_zju_mocap_dataloaders
    from gaussian_avatar.losses import TotalLoss
    from gaussian_avatar.models import GaussianAvatar
    from gaussian_avatar.models.mesh import create_mesh
    from gaussian_avatar.rendering import GaussianRenderer, MockRenderer, GSPLAT_AVAILABLE
    from gaussian_avatar.training import Trainer

    if args.config:
        print(f"Loading config from: {args.config}")
        config = config_from_yaml(args.config)
    else:
        config = TrainingConfig()

    # CLI overrides: apply explicit CLI args on top of YAML/defaults
    if args.num_iterations is not None:
        config.num_iterations = args.num_iterations
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.image_size is not None:
        config.image_size = tuple(args.image_size)
    if args.num_gaussians is not None:
        config.num_gaussians = args.num_gaussians
    if args.save_images_interval is not None:
        config.save_images_interval = args.save_images_interval
    if args.save_eval_images:
        config.num_eval_images_to_save = args.num_eval_images
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name

    if args.data_root is not None:
        config.data.data_root = args.data_root
    if args.subjects is not None:
        config.data.subjects = args.subjects
    if args.dataset_format is not None:
        config.data.dataset_format = args.dataset_format
    if args.stop_frame is not None:
        config.data.stop_frame = args.stop_frame
    if args.frame_indices is not None:
        config.data.frame_indices = args.frame_indices
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.skip is not None:
        config.data.skip = args.skip

    if args.model_type is not None:
        config.model.model_type = args.model_type
    if args.model_path is not None:
        config.model.model_path = args.model_path
    if args.gender is not None:
        config.model.gender = args.gender
    if args.lod is not None:
        config.model.lod = args.lod

    if args.wandb_project is not None:
        config.wandb.project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb.entity = args.wandb_entity
    if args.wandb_name is not None:
        config.wandb.name = args.wandb_name
    if args.no_wandb:
        config.wandb.enabled = False

    if args.optimize_pose:
        config.per_frame_params.enabled = True
        config.per_frame_params.optimize_pose = True

    # Parse CLI resolution schedule (overrides YAML if provided)
    if args.resolution_schedule is not None:
        flat = args.resolution_schedule
        if len(flat) % 3 != 0:
            raise ValueError(
                "--resolution-schedule must be a flat list of (iteration height width) "
                f"triples, but got {len(flat)} values which is not divisible by 3"
            )
        resolution_schedule: list[ResolutionScheduleEntry] = []
        for i in range(0, len(flat), 3):
            resolution_schedule.append(
                ResolutionScheduleEntry(
                    iteration=flat[i],
                    resolution=(flat[i + 1], flat[i + 2]),
                )
            )
        config.resolution_schedule = resolution_schedule
        # Re-run validation to auto-derive image_size
        config.__post_init__()
        print(f"Resolution schedule: {[(e.iteration, e.resolution) for e in resolution_schedule]}")

    output_dir = Path(config.output_dir) / config.experiment_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    use_wandb = config.wandb.enabled
    if use_wandb:
        try:
            import wandb

            wandb_config = config_to_flat_dict(config)

            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name or config.experiment_name,
                config=wandb_config,
            )
            print(f"Initialized wandb project: {config.wandb.project}")
        except ImportError:
            print("WARNING: wandb not installed, logging disabled")
            use_wandb = False

    model_type = config.model.model_type
    model_path = config.model.model_path
    gender = config.model.gender
    lod = config.model.lod

    if model_type == "mhr" and gender != "male":
        print(
            f"WARNING: gender={gender} is ignored for MHR model type. "
            "MHR uses identity coefficients instead of gender."
        )

    print(f"Loading {model_type.upper()} mesh from: {model_path}")
    mesh = create_mesh(
        model_type=model_type,
        model_path=model_path,
        gender=gender,
        lod=lod,
        device=device,
    )

    print(f"Creating GaussianAvatar with {config.num_gaussians} Gaussians")
    if config.disable_offsets:
        print("Offsets DISABLED: training without deformation MLP")
    avatar = GaussianAvatar(
        mesh=mesh,
        num_gaussians=config.num_gaussians,
        deformation_config=config.deformation,
        disable_offsets=config.disable_offsets,
        use_sh=config.use_sh,
        max_sh_degree=config.max_sh_degree,
        max_stretch=config.max_stretch,
        max_local_offset=config.max_local_offset,
    ).to(device)
    num_gaussians = avatar.gaussian_model.num_gaussians
    print(f"Total Gaussians: {num_gaussians}")

    bg_color = tuple(config.background_color) if config.background_color else (0.0, 0.0, 0.0)
    initial_sh_degree = config.get_sh_degree_at(0) if config.use_sh else None
    if GSPLAT_AVAILABLE:
        print("Using GaussianRenderer (gsplat)")
        renderer = GaussianRenderer(
            background_color=bg_color,
            sh_degree=initial_sh_degree,
        ).to(device)
    else:
        print("WARNING: gsplat not available, using MockRenderer (training will not work)")
        renderer = MockRenderer(allow_forward=True)

    loss_fn = TotalLoss(
        weights=config.loss_weights,
        thresholds=config.loss_thresholds,
        background_color=config.background_color,
    )

    if config.data.data_root is None:
        argparse.ArgumentParser().error(
            "--data-root is required (via CLI or data.data_root in YAML config)"
        )

    subjects = config.data.subjects
    # Normalize single-item list "all" to string for dataset constructor
    if len(subjects) == 1 and subjects[0] == "all":
        subjects = "all"

    print(f"Loading dataset from: {config.data.data_root}")
    print(f"Subjects: {subjects}")
    if not config.data.preprocess_masks:
        print("Mask preprocessing (erosion + blur) DISABLED")

    if config.data.dataset_format == "zju_mocap":
        train_cameras = config.data.train_cameras or ["Camera_B1"]
        print(f"ZJU MoCap: train cameras={train_cameras}")
        zju_kwargs = {}
        if config.data.stop_frame is not None:
            zju_kwargs["stop_frame"] = config.data.stop_frame
            print(f"Stop frame: {config.data.stop_frame}")
        if config.data.skip is not None:
            zju_kwargs["skip"] = config.data.skip
            print(f"Frame skip: {config.data.skip}")
        train_loader, test_loader = create_zju_mocap_dataloaders(
            data_root=config.data.data_root,
            subjects=subjects,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.data.num_workers,
            train_cameras=train_cameras,
            model_type=model_type,
            preprocess_masks=config.data.preprocess_masks,
            **zju_kwargs,
        )
    elif model_type == "mhr":
        # Detect corrected format: check for cam000/mhr/raw/ in first subject
        dataset_format = config.data.dataset_format
        if dataset_format == "auto":
            check_subject = subjects[0] if isinstance(subjects, list) else subjects
            if check_subject == "all":
                # Check any subject directory for corrected structure
                data_root_path = Path(config.data.data_root)
                for d in data_root_path.iterdir():
                    if d.is_dir() and (d / "cam000" / "mhr" / "raw").is_dir():
                        dataset_format = "corrected"
                        break
                else:
                    dataset_format = "standard"
            else:
                mhr_raw = Path(config.data.data_root) / check_subject / "cam000" / "mhr" / "raw"
                dataset_format = "corrected" if mhr_raw.is_dir() else "standard"
            print(f"Auto-detected MHR dataset format: {dataset_format}")

        if dataset_format == "corrected":
            mhr_kwargs = {}
            if config.data.stop_frame is not None:
                mhr_kwargs["stop_frame"] = config.data.stop_frame
                print(f"Stop frame: {config.data.stop_frame}")
            if config.data.skip is not None:
                mhr_kwargs["skip"] = config.data.skip
                print(f"Frame skip: {config.data.skip}")
            if config.data.undistort:
                mhr_kwargs["undistort"] = True
                print("Undistortion ENABLED for all images")
            train_loader, test_loader = create_mhr_corrected_dataloaders(
                data_root=config.data.data_root,
                subjects=subjects,
                image_size=config.image_size,
                batch_size=config.batch_size,
                num_workers=config.data.num_workers,
                preprocess_masks=config.data.preprocess_masks,
                **mhr_kwargs,
            )
        else:
            train_loader, test_loader = create_mhr_dataloaders(
                data_root=config.data.data_root,
                subjects=subjects,
                image_size=config.image_size,
                batch_size=config.batch_size,
                num_workers=config.data.num_workers,
                preprocess_masks=config.data.preprocess_masks,
            )
    else:
        dataloader_kwargs = {}
        if config.data.frame_indices is not None:
            dataloader_kwargs["frame_indices"] = config.data.frame_indices
        if config.data.stop_frame is not None:
            dataloader_kwargs["stop_frame"] = config.data.stop_frame
            print(f"Stop frame: {config.data.stop_frame} (train=[0,{config.data.stop_frame}), test=[{config.data.stop_frame},N))")
        if config.data.skip is not None:
            dataloader_kwargs["skip"] = config.data.skip
            print(f"Frame skip: {config.data.skip} (using every {config.data.skip}-th frame)")
        if config.data.undistort:
            dataloader_kwargs["undistort"] = True
            print("Undistortion ENABLED for all images")
        dataloader_kwargs["preprocess_masks"] = config.data.preprocess_masks
        # Pass model_type so corrected dataset loads from the right fits directory
        # (smpls/ for SMPL, smplxs/ for SMPL-X). fits_model_type overrides this
        # to allow e.g. SMPL-X mesh with SMPL pose fits.
        dataloader_kwargs["model_type"] = (
            config.model.fits_model_type or config.model.model_type
        )
        train_loader, test_loader = create_dataloaders(
            data_root=config.data.data_root,
            subjects=subjects,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.data.num_workers,
            dataset_format=config.data.dataset_format,
            **dataloader_kwargs,
        )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("Initializing trainer...")
    trainer = Trainer(
        avatar=avatar,
        renderer=renderer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"Resumed at iteration: {trainer.iteration}")

    print(f"Starting training for {config.num_iterations} iterations...")
    trainer.train(config.num_iterations)

    final_checkpoint = checkpoint_dir / "final.pt"
    trainer.save_checkpoint(final_checkpoint)
    print(f"Saved final checkpoint: {final_checkpoint}")

    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass

    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
