"""Configuration dataclasses for loss functions."""

from dataclasses import dataclass


@dataclass
class LossWeights:
    """Configuration for loss function weights.

    Args:
        lambda_l1: Weight for L1 reconstruction loss
        lambda_l2: Weight for L2 reconstruction loss
        lambda_ssim: Weight for SSIM structural similarity loss
        lambda_lpips: Weight for LPIPS perceptual loss
        lpips_net: LPIPS backbone network ("alex", "vgg", "squeeze")
        mask_reconstruction_loss: When True, L1/L2 losses are computed only
            inside the foreground mask. When False, they are computed over
            the whole composited image.
        lambda_mask: Weight for mask/silhouette loss
        lambda_position_reg: Weight for position regularization
        lambda_scale_reg: Weight for scale regularization
        lambda_offset_reg: Weight for MLP offset regularization
        lambda_offset_color_reg: Weight for MLP color offset regularization
        lambda_scale_isotropy: Weight for tangent-plane scale isotropy loss
        lambda_aiap_position: Weight for AIAP position isometry loss
        lambda_aiap_covariance: Weight for AIAP covariance isometry loss
    """

    # Reconstruction
    lambda_l1: float = 1.0
    lambda_l2: float = 0.0
    lambda_ssim: float = 0.2
    lambda_lpips: float = 0.1
    lpips_net: str = "alex"
    mask_reconstruction_loss: bool = True

    # Mask
    lambda_mask: float = 0.1

    # Regularization
    lambda_position_reg: float = 0.01
    lambda_scale_reg: float = 0.01
    lambda_offset_reg: float = 0.001
    lambda_offset_color_reg: float = 0.001
    lambda_scale_isotropy: float = 0.0

    # Boundary weighting
    boundary_weight: float = 0.0
    boundary_kernel_size: int = 3

    # Opacity binary regularization
    lambda_opacity_binary: float = 0.0

    # AIAP
    lambda_aiap_position: float = 0.01
    lambda_aiap_covariance: float = 0.01

    # SSL (self-supervised loss on perturbed poses)
    lambda_ssl_aiap_position: float = 0.01
    lambda_ssl_aiap_covariance: float = 0.01
    lambda_ssl_silhouette: float = 0.05


@dataclass
class LossThresholds:
    """Configuration for loss function thresholds.

    Args:
        epsilon_position: Maximum local position magnitude before penalty (meters)
        epsilon_scale: Maximum linear scale before penalty
        k_neighbors: Number of neighbors for AIAP loss computation
    """

    epsilon_position: float = 0.05
    epsilon_scale: float = 0.15
    k_neighbors: int = 5
