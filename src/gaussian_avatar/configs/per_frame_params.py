"""Configuration for per-frame body model parameter optimization.

This module provides PerFrameParamsConfig, which controls whether and how
per-frame pose, shape, and translation parameters are jointly optimized
alongside Gaussian and MLP parameters during training. Works with any
body model (SMPL, SMPL-X, MHR).
"""

from dataclasses import dataclass


@dataclass
class PerFrameParamsConfig:
    """Configuration for per-frame body model parameter optimization.

    When enabled, learnable offsets (initialized to zero) are added on top
    of the dataset's body model parameters. This allows the model to correct
    noisy or imprecise fits from the dataset.

    Args:
        enabled: Whether to enable per-frame parameter optimization.
        optimize_pose: Learn per-frame pose offsets (dimension depends on
            model type: 72 for SMPL, 66 for SMPL-X, 204 for MHR).
        optimize_betas: Learn per-subject shape offsets (shared across frames).
        optimize_trans: Learn per-frame translation offsets (3-dim).
        lr_pose: Learning rate for pose offsets.
        lr_betas: Learning rate for shape offsets.
        lr_trans: Learning rate for translation offsets.
    """

    enabled: bool = False
    optimize_pose: bool = True
    optimize_betas: bool = True
    optimize_trans: bool = True
    lr_pose: float = 1e-5
    lr_betas: float = 1e-5
    lr_trans: float = 1e-4
