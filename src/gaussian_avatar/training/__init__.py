"""Training module for Gaussian Avatar.

This module provides:
- Trainer: Main training loop orchestration
- create_optimizer: Factory for Adam optimizer with per-parameter learning rates
- create_scheduler: Factory for exponential decay LR scheduler
- compute_psnr: Peak Signal-to-Noise Ratio metric
- compute_ssim: Structural Similarity Index metric
- compute_ssim_map: Per-pixel SSIM error map for visualization
- LPIPSMetric: Learned Perceptual Image Patch Similarity metric
- save_checkpoint: Save training state to file
- load_checkpoint: Load training state from file
- ImageSaver: Utility for saving training and evaluation images
- DensificationManager: Adaptive Gaussian density control
"""

from gaussian_avatar.training.checkpoint import load_checkpoint, save_checkpoint
from gaussian_avatar.training.densification import DensificationManager
from gaussian_avatar.training.image_saver import ImageSaver
from gaussian_avatar.training.metrics import LPIPSMetric, compute_psnr, compute_ssim, compute_ssim_map
from gaussian_avatar.training.optimizer import create_optimizer, create_scheduler
from gaussian_avatar.training.trainer import Trainer

__all__ = [
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "compute_psnr",
    "compute_ssim",
    "compute_ssim_map",
    "LPIPSMetric",
    "save_checkpoint",
    "load_checkpoint",
    "ImageSaver",
    "DensificationManager",
]
