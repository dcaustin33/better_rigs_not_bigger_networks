"""Optimizer and learning rate scheduler factories for Gaussian Avatar training.

This module provides factory functions for creating:
- Adam optimizer with per-parameter learning rates for different Gaussian properties
- Exponential decay LR scheduler with configurable start/stop iterations
"""

from typing import Any, Protocol

import torch
from torch.optim.lr_scheduler import LambdaLR


class HasGaussianModel(Protocol):
    """Protocol for objects that have a gaussian_model attribute."""

    @property
    def gaussian_model(self): ...

    @property
    def deformation_mlp(self): ...


class HasLRConfig(Protocol):
    """Protocol for config objects with learning rate parameters."""

    lr_position: float
    lr_rotation: float
    lr_scale: float
    lr_color: float
    lr_opacity: float
    lr_mlp: float


class HasDecayConfig(Protocol):
    """Protocol for config objects with LR decay parameters."""

    lr_decay_start: int
    lr_decay_rate: float
    lr_decay_steps: int


def create_optimizer(
    avatar: HasGaussianModel,
    config: HasLRConfig,
    per_frame_params: Any = None,
) -> torch.optim.Adam:
    """Create Adam optimizer with per-parameter learning rates.

    Sets up separate parameter groups for each Gaussian property type,
    allowing fine-grained control over learning rates. The parameter
    groups are:
    - position: local_positions (small LR for stable geometry)
    - rotation: local_rotations (moderate LR)
    - scale: local_scales (moderate LR)
    - color: base_colors (moderate LR)
    - opacity: raw_opacities (high LR for faster convergence)
    - mlp: deformation MLP parameters (separate from Gaussian params)
    - pose/betas/trans: per-frame body model offsets (if provided)

    Args:
        avatar: GaussianAvatar model with gaussian_model and deformation_mlp
        config: Configuration with lr_position, lr_rotation, lr_scale,
            lr_color, lr_opacity, lr_mlp attributes
        per_frame_params: Optional PerFrameParameters module whose param
            groups will be appended to the optimizer

    Returns:
        Configured Adam optimizer
    """
    param_groups = [
        {
            "name": "position",
            "params": [avatar.gaussian_model._local_positions],
            "lr": config.lr_position,
        },
        {
            "name": "rotation",
            "params": [avatar.gaussian_model.local_rotations],
            "lr": config.lr_rotation,
        },
        {
            "name": "scale",
            "params": [avatar.gaussian_model.local_scales],
            "lr": config.lr_scale,
        },
        {
            "name": "sh" if getattr(avatar.gaussian_model, "use_sh", False) else "color",
            "params": [avatar.gaussian_model.sh_coefficients]
            if getattr(avatar.gaussian_model, "use_sh", False)
            else [avatar.gaussian_model.base_colors],
            "lr": getattr(config, "lr_sh", config.lr_color)
            if getattr(avatar.gaussian_model, "use_sh", False)
            else config.lr_color,
        },
        {
            "name": "opacity",
            "params": [avatar.gaussian_model.raw_opacities],
            "lr": config.lr_opacity,
        },
    ]

    if avatar.deformation_mlp is not None:
        param_groups.append(
            {
                "name": "mlp",
                "params": list(avatar.deformation_mlp.parameters()),
                "lr": config.lr_mlp,
            },
        )

    if per_frame_params is not None:
        param_groups.extend(per_frame_params.get_param_groups())

    return torch.optim.Adam(param_groups, eps=1e-15)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: HasDecayConfig,
) -> LambdaLR:
    """Create exponential decay learning rate scheduler.

    The scheduler maintains constant LR until lr_decay_start, then
    decays exponentially over lr_decay_steps iterations to reach
    final_lr = initial_lr * lr_decay_rate.

    The decay formula is:
        lr(t) = initial_lr * decay_rate^progress
    where progress = (t - decay_start) / decay_steps, clamped to [0, 1].

    Args:
        optimizer: Optimizer to schedule
        config: Configuration with lr_decay_start, lr_decay_rate,
            lr_decay_steps attributes

    Returns:
        Configured LambdaLR scheduler
    """

    def lr_lambda(iteration: int) -> float:
        if iteration < config.lr_decay_start:
            return 1.0

        progress = (iteration - config.lr_decay_start) / max(config.lr_decay_steps, 1)
        progress = min(progress, 1.0)

        return config.lr_decay_rate**progress

    return LambdaLR(optimizer, lr_lambda)
