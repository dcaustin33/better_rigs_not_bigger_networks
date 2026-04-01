"""MHR parameter loading utilities.

This module provides functions for loading MHR (Momentum Human Rig)
parameters from .npz files, including model parameters (poses),
identity coefficients (shape), and optional expression coefficients.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from gaussian_avatar.constants import MHR_EXPRESSION_DIM, MHR_POSE_DIM, MHR_SHAPE_DIM

def load_mhr_params(params_path: Union[str, Path]) -> Dict[str, Optional[np.ndarray]]:
    """Load MHR parameters from an .npz file.

    Parses MHR parameter files containing per-frame model parameters,
    shared identity coefficients, and optional expression coefficients.

    Args:
        params_path: Path to .npz file containing MHR parameters.

    Returns:
        Dictionary containing:
            - "model_parameters": (T, 204) float32 array of per-frame model parameters
            - "identity_coeffs": (45,) float32 array of identity coefficients
            - "expression_coeffs": (T, 72) float32 array of expression coefficients,
              or None if not present in the file

    Raises:
        FileNotFoundError: If params_path does not exist
        KeyError: If required fields (model_parameters, identity_coeffs) are missing
        ValueError: If array shapes don't match expected MHR dimensions
    """
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"MHR params file not found: {params_path}")

    data = np.load(params_path)

    if "model_parameters" not in data:
        raise KeyError(
            f"Required field 'model_parameters' not found in '{params_path}'. "
            f"Available keys: {list(data.keys())}"
        )
    if "identity_coeffs" not in data:
        raise KeyError(
            f"Required field 'identity_coeffs' not found in '{params_path}'. "
            f"Available keys: {list(data.keys())}"
        )

    model_parameters = np.array(data["model_parameters"], dtype=np.float32)
    identity_coeffs = np.array(data["identity_coeffs"], dtype=np.float32)

    if model_parameters.ndim != 2 or model_parameters.shape[1] != MHR_POSE_DIM:
        raise ValueError(
            f"model_parameters must have shape (T, {MHR_POSE_DIM}), "
            f"got {model_parameters.shape}"
        )
    if identity_coeffs.shape != (MHR_SHAPE_DIM,):
        raise ValueError(
            f"identity_coeffs must have shape ({MHR_SHAPE_DIM},), "
            f"got {identity_coeffs.shape}"
        )

    expression_coeffs = None
    if "expression_coeffs" in data:
        expression_coeffs = np.array(data["expression_coeffs"], dtype=np.float32)
        if (
            expression_coeffs.ndim != 2
            or expression_coeffs.shape[1] != MHR_EXPRESSION_DIM
        ):
            raise ValueError(
                f"expression_coeffs must have shape (T, {MHR_EXPRESSION_DIM}), "
                f"got {expression_coeffs.shape}"
            )
        num_frames = model_parameters.shape[0]
        if expression_coeffs.shape[0] != num_frames:
            raise ValueError(
                f"expression_coeffs frame count ({expression_coeffs.shape[0]}) "
                f"does not match model_parameters frame count ({num_frames})"
            )

    return {
        "model_parameters": model_parameters,
        "identity_coeffs": identity_coeffs,
        "expression_coeffs": expression_coeffs,
    }
