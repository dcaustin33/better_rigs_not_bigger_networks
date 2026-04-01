"""Loss functions for Gaussian Avatar training.

This module provides all loss functions needed for training Gaussian avatars,
including reconstruction losses (L1, SSIM, LPIPS), mask losses, regularization
losses, and AIAP isometry losses.
"""

from gaussian_avatar.losses.config import LossWeights, LossThresholds
from gaussian_avatar.losses.reconstruction import (
    L1Loss,
    L2Loss,
    SSIMLoss,
    LPIPSLoss,
    ReconstructionLoss,
)
from gaussian_avatar.losses.mask import MaskLoss
from gaussian_avatar.losses.regularization import (
    PositionRegularizationLoss,
    ScaleRegularizationLoss,
    ScaleIsotropyLoss,
    OpacityBinaryLoss,
    OffsetRegularizationLoss,
    RegularizationLoss,
)
from gaussian_avatar.losses.aiap import (
    NeighborCache,
    AIAPPositionLoss,
    AIAPCovarianceLoss,
    AIAPLoss,
)
from gaussian_avatar.losses.ssl import (
    perturb_pose,
    jitter_camera,
    rasterize_mesh_silhouette,
    compute_ssl_silhouette_loss,
)
from gaussian_avatar.losses.total import TotalLoss
from gaussian_avatar.losses.utils import (
    compute_boundary_weight_map,
    composite_premultiplied,
    composite_with_random_background,
    ensure_batch_dim,
    get_device_from_tensors,
)

__all__ = [
    # Config
    "LossWeights",
    "LossThresholds",
    # Reconstruction
    "L1Loss",
    "L2Loss",
    "SSIMLoss",
    "LPIPSLoss",
    "ReconstructionLoss",
    # Mask
    "MaskLoss",
    # Regularization
    "PositionRegularizationLoss",
    "ScaleRegularizationLoss",
    "ScaleIsotropyLoss",
    "OpacityBinaryLoss",
    "OffsetRegularizationLoss",
    "RegularizationLoss",
    # AIAP
    "NeighborCache",
    "AIAPPositionLoss",
    "AIAPCovarianceLoss",
    "AIAPLoss",
    # SSL
    "perturb_pose",
    "jitter_camera",
    "rasterize_mesh_silhouette",
    "compute_ssl_silhouette_loss",
    # Total
    "TotalLoss",
    # Utils
    "compute_boundary_weight_map",
    "composite_premultiplied",
    "composite_with_random_background",
    "ensure_batch_dim",
    "get_device_from_tensors",
]
