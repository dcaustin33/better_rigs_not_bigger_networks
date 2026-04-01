"""Checkpoint save/load functions for training state persistence.

This module provides functions for saving and loading training checkpoints,
including model weights, optimizer state, scheduler state, and configuration.
"""

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    avatar: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    iteration: int,
    config: object,
    metrics: dict | None = None,
    densification: object | None = None,
    per_frame_params: nn.Module | None = None,
    model_type: str | None = None,
    lod: int | None = None,
    training_betas: dict[str, list[float]] | None = None,
) -> None:
    """Save training checkpoint.

    Creates a checkpoint file containing all state needed to resume training
    or load a trained model for inference.

    Args:
        path: Checkpoint file path (.pt file)
        avatar: GaussianAvatar model (or any nn.Module)
        optimizer: Optimizer with state to save
        scheduler: LR scheduler with state to save
        iteration: Current training iteration
        config: Configuration dataclass (will be converted to dict via asdict)
        metrics: Optional dictionary of evaluation metrics to include
        densification: Optional DensificationManager whose gradient
            accumulators should be persisted for mid-interval resume
        per_frame_params: Optional PerFrameParameters module to persist
        model_type: Body model type (e.g. "smpl", "smplx", "mhr")
        lod: MHR level-of-detail (0-6), only relevant for MHR
        training_betas: Per-subject shape coefficients used during training.
            Maps subject name to list of floats (e.g. 10 for SMPL, 45 for MHR).

    Raises:
        TypeError: If config is not a dataclass
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "avatar_state_dict": avatar.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "iteration": iteration,
        "config": asdict(config) if is_dataclass(config) else config,
    }

    # Store SH mode flag for validation on load
    if hasattr(avatar, "gaussian_model") and hasattr(avatar.gaussian_model, "use_sh"):
        checkpoint["use_sh"] = avatar.gaussian_model.use_sh

    if model_type is not None:
        checkpoint["model_type"] = model_type

    if lod is not None:
        checkpoint["lod"] = lod

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if densification is not None and hasattr(densification, "state_dict"):
        checkpoint["densification_state"] = densification.state_dict()

    if per_frame_params is not None:
        checkpoint["per_frame_params_state_dict"] = per_frame_params.state_dict()

    if training_betas is not None:
        checkpoint["training_betas"] = training_betas

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    avatar: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: _LRScheduler | None = None,
    device: str | torch.device = "cpu",
    densification: object | None = None,
    per_frame_params: nn.Module | None = None,
    expected_model_type: str | None = None,
) -> dict:
    """Load training checkpoint.

    Restores model weights and optionally optimizer/scheduler/densification
    state from a saved checkpoint.

    Args:
        path: Checkpoint file path
        avatar: GaussianAvatar model to load weights into
        optimizer: Optional optimizer to restore state (None for inference-only)
        scheduler: Optional scheduler to restore state (None for inference-only)
        device: Device to load tensors to ("cpu" or "cuda")
        densification: Optional DensificationManager to restore gradient
            accumulators into (None for inference-only)
        per_frame_params: Optional PerFrameParameters module to restore
            state into (None if not using per-frame optimization)
        expected_model_type: If provided, validate the checkpoint's
            model_type matches. Raises ValueError on mismatch.

    Returns:
        Dictionary containing:
            - "iteration": Training iteration from checkpoint
            - "config": Configuration dict from checkpoint
            - "metrics": Metrics dict (if saved, else None)
            - "model_type": Body model type from checkpoint (None if absent)
            - "lod": MHR LOD level from checkpoint (None if absent)
            - "training_betas": Per-subject shape coefficients (None if absent)

    Raises:
        FileNotFoundError: If checkpoint path does not exist
        ValueError: If expected_model_type is set and doesn't match checkpoint
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Validate model_type if caller specified an expectation
    ckpt_model_type = checkpoint.get("model_type")
    if expected_model_type is not None and ckpt_model_type is not None:
        if ckpt_model_type != expected_model_type:
            raise ValueError(
                f"Checkpoint model_type '{ckpt_model_type}' does not match "
                f"expected model_type '{expected_model_type}'"
            )

    # Validate SH mode matches
    ckpt_use_sh = checkpoint.get("use_sh")
    if ckpt_use_sh is not None and hasattr(avatar, "gaussian_model"):
        model_use_sh = getattr(avatar.gaussian_model, "use_sh", False)
        if ckpt_use_sh != model_use_sh:
            raise ValueError(
                f"Checkpoint use_sh={ckpt_use_sh} does not match "
                f"model use_sh={model_use_sh}"
            )

    # Resize Gaussian model if checkpoint has a different count (e.g. after
    # densification pruned/split Gaussians during training).
    state_dict = checkpoint["avatar_state_dict"]
    _pos_key = "gaussian_model._local_positions"
    resized = False
    old_params = None
    if _pos_key in state_dict and hasattr(avatar, "gaussian_model"):
        ckpt_num = state_dict[_pos_key].shape[0]
        model_num = avatar.gaussian_model.num_gaussians
        if ckpt_num != model_num:
            # Capture old parameter references before resize replaces them
            old_params = {name: param for name, param in avatar.named_parameters()}
            avatar.gaussian_model.resize(ckpt_num)
            resized = True

    avatar.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        if resized and old_params is not None:
            # After resize, the optimizer's param_groups still reference old
            # (now orphaned) parameter objects. Remap them to the new parameters
            # before loading optimizer state, so load_state_dict correctly keys
            # state entries by the new parameter objects.
            new_params = {name: param for name, param in avatar.named_parameters()}
            old_to_new = {}
            for name, old_param in old_params.items():
                new_param = new_params.get(name)
                if new_param is not None and new_param is not old_param:
                    old_to_new[id(old_param)] = new_param

            for group in optimizer.param_groups:
                for i, p in enumerate(group["params"]):
                    if id(p) in old_to_new:
                        group["params"][i] = old_to_new[id(p)]

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if (
        densification is not None
        and hasattr(densification, "load_state_dict")
        and "densification_state" in checkpoint
    ):
        densification.load_state_dict(checkpoint["densification_state"])

    if per_frame_params is not None and "per_frame_params_state_dict" in checkpoint:
        per_frame_params.load_state_dict(checkpoint["per_frame_params_state_dict"])

    return {
        "iteration": checkpoint.get("iteration", 0),
        "config": checkpoint.get("config"),
        "metrics": checkpoint.get("metrics"),
        "model_type": checkpoint.get("model_type"),
        "lod": checkpoint.get("lod"),
        "training_betas": checkpoint.get("training_betas"),
    }
