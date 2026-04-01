"""YAML configuration file utilities.

This module provides functions to load, save, and flatten TrainingConfig
and EvaluationConfig dataclasses from/to YAML files. Supports partial
configs where missing fields use dataclass defaults.

Functions:
    config_from_yaml: Load a YAML file into a TrainingConfig
    config_to_yaml: Dump a TrainingConfig to a YAML file
    config_to_flat_dict: Flatten a TrainingConfig into a dot-separated dict for wandb
    eval_config_from_yaml: Load a YAML file into an EvaluationConfig
    eval_config_to_yaml: Dump an EvaluationConfig to a YAML file
"""

import dataclasses
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from gaussian_avatar.configs.deformation import DeformationConfig
from gaussian_avatar.configs.evaluation import (
    EvalDataConfig,
    EvaluationConfig,
    BodyParamOptimConfig,
)
from gaussian_avatar.configs.per_frame_params import PerFrameParamsConfig
from gaussian_avatar.configs.training import (
    DataConfig,
    DensificationConfig,
    ModelConfig,
    ResolutionScheduleEntry,
    SHScheduleEntry,
    SSLConfig,
    TrainingConfig,
    WandbConfig,
)
from gaussian_avatar.losses.config import LossThresholds, LossWeights


def _dict_to_dataclass(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass instance.

    Handles nested dataclass fields, list-to-tuple conversion for
    tuple[int, int] fields, and list[ResolutionScheduleEntry] fields.
    Missing keys use the dataclass defaults.

    Args:
        dc_type: The dataclass type to instantiate
        data: Dictionary of field values (may be partial)

    Returns:
        An instance of dc_type populated from data
    """
    # Map of field name -> dataclass type for nested fields.
    # When multiple parent types share the same field name (e.g. "data"),
    # the type is resolved from the parent's type hints below.
    nested_dataclass_types: dict[str, type] = {
        "model": ModelConfig,
        "data": DataConfig,
        "wandb": WandbConfig,
        "densification": DensificationConfig,
        "loss_weights": LossWeights,
        "loss_thresholds": LossThresholds,
        "deformation": DeformationConfig,
        "per_frame_params": PerFrameParamsConfig,
        "ssl": SSLConfig,
        "body_optim": BodyParamOptimConfig,
    }
    # Override "data" type based on parent dataclass
    if dc_type is EvaluationConfig:
        nested_dataclass_types["data"] = EvalDataConfig

    kwargs: dict[str, Any] = {}
    hints = get_type_hints(dc_type)

    for field_info in dataclasses.fields(dc_type):
        name = field_info.name
        if name not in data:
            continue

        value = data[name]
        hint = hints.get(name)

        # Nested dataclass field
        if name in nested_dataclass_types and isinstance(value, dict):
            kwargs[name] = _dict_to_dataclass(nested_dataclass_types[name], value)
            continue

        # list[ResolutionScheduleEntry] - convert list of dicts
        if name == "resolution_schedule" and isinstance(value, list):
            entries = []
            for item in value:
                if isinstance(item, dict):
                    res = item.get("resolution")
                    if isinstance(res, list):
                        item = {**item, "resolution": tuple(res)}
                    entries.append(ResolutionScheduleEntry(**item))
                else:
                    entries.append(item)
            kwargs[name] = entries
            continue

        # list[SHScheduleEntry] - convert list of dicts
        if name == "sh_schedule" and isinstance(value, list):
            entries = []
            for item in value:
                if isinstance(item, dict):
                    entries.append(SHScheduleEntry(**item))
                else:
                    entries.append(item)
            kwargs[name] = entries
            continue

        # tuple fields (e.g. image_size, background_color) - YAML lists become tuples.
        # Handle both plain tuple[...] and Optional tuple[...] | None.
        if hint is not None and isinstance(value, list):
            origin = get_origin(hint)
            if origin is tuple:
                kwargs[name] = tuple(value)
                continue
            # Union types (e.g. tuple[float, float, float] | None)
            args = get_args(hint)
            if args and any(get_origin(a) is tuple for a in args):
                kwargs[name] = tuple(value)
                continue

        # int | None union types - pass through
        kwargs[name] = value

    return dc_type(**kwargs)


def config_from_yaml(path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file.

    The YAML file should have a top-level ``training:`` key containing fields
    matching TrainingConfig. Missing fields use dataclass defaults, so partial
    configs are supported.

    Args:
        path: Path to the YAML configuration file

    Returns:
        A TrainingConfig instance populated from the file

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML file has no 'training' key
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"YAML file is empty: {path}")

    training_data = raw.get("training", None)
    if training_data is None:
        raise ValueError(
            f"YAML config must have a top-level 'training:' key, "
            f"but {path} has keys: {list(raw.keys())}"
        )

    # Backward compat: migrate top-level 'undistort' into 'data' section
    if "undistort" in training_data and "data" not in training_data:
        training_data.setdefault("data", {})["undistort"] = training_data.pop("undistort")
    elif "undistort" in training_data:
        training_data["data"].setdefault("undistort", training_data.pop("undistort"))

    return _dict_to_dataclass(TrainingConfig, training_data)


def config_to_yaml(config: TrainingConfig, path: str | Path) -> None:
    """Dump a TrainingConfig to a YAML file.

    Converts the dataclass to a dict using ``dataclasses.asdict()`` and writes
    it under a ``training:`` top-level key. Useful for generating default
    config templates via ``--dump-config``.

    Args:
        config: The TrainingConfig to serialize
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = dataclasses.asdict(config)
    wrapped = {"training": data}

    # Register a representer so tuples are dumped as YAML sequences (lists)
    # instead of Python-specific !!python/tuple tags.
    dumper = yaml.Dumper
    dumper.add_representer(
        tuple, lambda d, t: d.represent_sequence("tag:yaml.org,2002:seq", t)
    )

    with open(path, "w") as f:
        yaml.dump(wrapped, f, Dumper=dumper, default_flow_style=False, sort_keys=False)


def config_to_flat_dict(config: TrainingConfig) -> dict[str, Any]:
    """Flatten a TrainingConfig into a dot-separated dictionary.

    Recursively walks nested dataclass fields and produces keys like
    ``densification.start_iter`` and ``loss_weights.lambda_l1``. Useful
    for logging the full configuration to wandb.

    Args:
        config: The TrainingConfig to flatten

    Returns:
        Dictionary with dot-separated keys and scalar values
    """

    def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {}
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for field_info in dataclasses.fields(obj):
                name = field_info.name
                value = getattr(obj, name)
                key = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
                if dataclasses.is_dataclass(value) and not isinstance(value, type):
                    result.update(_flatten(value, key))
                elif isinstance(value, list):
                    # For resolution_schedule, store as list of dicts
                    result[key] = [
                        dataclasses.asdict(v)
                        if dataclasses.is_dataclass(v)
                        else v
                        for v in value
                    ]
                elif isinstance(value, tuple):
                    result[key] = list(value)
                else:
                    result[key] = value
        return result

    return _flatten(config)


def eval_config_from_yaml(path: str | Path) -> EvaluationConfig:
    """Load an EvaluationConfig from a YAML file.

    The YAML file should have a top-level ``evaluation:`` key containing
    fields matching EvaluationConfig. Missing fields use dataclass defaults,
    so partial configs are supported.

    Args:
        path: Path to the YAML configuration file

    Returns:
        An EvaluationConfig instance populated from the file

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML file has no 'evaluation' key
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"YAML file is empty: {path}")

    eval_data = raw.get("evaluation", None)
    if eval_data is None:
        raise ValueError(
            f"YAML config must have a top-level 'evaluation:' key, "
            f"but {path} has keys: {list(raw.keys())}"
        )

    return _dict_to_dataclass(EvaluationConfig, eval_data)


def eval_config_to_yaml(config: EvaluationConfig, path: str | Path) -> None:
    """Dump an EvaluationConfig to a YAML file.

    Converts the dataclass to a dict using ``dataclasses.asdict()`` and writes
    it under an ``evaluation:`` top-level key. Useful for generating default
    config templates via ``--dump-config``.

    Args:
        config: The EvaluationConfig to serialize
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = dataclasses.asdict(config)
    wrapped = {"evaluation": data}

    # Register a representer so tuples are dumped as YAML sequences (lists)
    # instead of Python-specific !!python/tuple tags.
    dumper = yaml.Dumper
    dumper.add_representer(
        tuple, lambda d, t: d.represent_sequence("tag:yaml.org,2002:seq", t)
    )

    with open(path, "w") as f:
        yaml.dump(wrapped, f, Dumper=dumper, default_flow_style=False, sort_keys=False)
