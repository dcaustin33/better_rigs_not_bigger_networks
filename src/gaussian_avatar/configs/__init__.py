"""Configuration dataclasses for the Gaussian Avatar pipeline.

This module provides configuration structures for:
- DeformationConfig: MLP architecture and encoding settings
- AvatarConfig: Complete avatar model configuration
- DensificationConfig: Densification schedule and thresholds
- TrainingConfig: Complete training configuration
- EvaluationConfig: Complete evaluation configuration
"""

from gaussian_avatar.configs.deformation import AvatarConfig, DeformationConfig
from gaussian_avatar.configs.evaluation import EvalDataConfig, EvaluationConfig
from gaussian_avatar.configs.per_frame_params import PerFrameParamsConfig
from gaussian_avatar.configs.training import (
    DataConfig,
    DensificationConfig,
    ModelConfig,
    ResolutionScheduleEntry,
    SHScheduleEntry,
    TrainingConfig,
    WandbConfig,
)
from gaussian_avatar.configs.yaml_utils import (
    config_from_yaml,
    config_to_flat_dict,
    config_to_yaml,
    eval_config_from_yaml,
    eval_config_to_yaml,
)

__all__ = [
    "DataConfig",
    "DeformationConfig",
    "AvatarConfig",
    "DensificationConfig",
    "ModelConfig",
    "EvalDataConfig",
    "EvaluationConfig",
    "PerFrameParamsConfig",
    "ResolutionScheduleEntry",
    "SHScheduleEntry",
    "TrainingConfig",
    "WandbConfig",
    "config_from_yaml",
    "config_to_yaml",
    "config_to_flat_dict",
    "eval_config_from_yaml",
    "eval_config_to_yaml",
]
