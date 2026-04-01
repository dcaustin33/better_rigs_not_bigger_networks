"""Base classes for dataset implementations.

This module defines the abstract base class and data structures that all
dataset implementations must inherit from and use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_split_indices(
    num_frames: int,
    test_ratio: float = 0.25,
    start_frame: int = 0,
    stop_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute train/test frame indices.

    Splits valid frames chronologically: first (1 - test_ratio) for training,
    remainder for testing.

    Args:
        num_frames: Total number of frames (poses)
        test_ratio: Fraction of frames to use for testing (default 0.25).
            Ignored when stop_frame is set.
        start_frame: First valid frame index (default 0). Use 1 for the
            original PeopleSnapshot dataset where frame 0 has no mask.
        stop_frame: Optional frame index that separates train from test.
            When set, frames [start_frame, stop_frame) are train and
            [stop_frame, num_frames) are test.

    Returns:
        Tuple of (train_indices, test_indices) arrays

    Raises:
        ValueError: If stop_frame is out of range
    """
    if stop_frame is not None:
        if stop_frame < start_frame or stop_frame >= num_frames:
            raise ValueError(
                f"stop_frame must be in [{start_frame}, {num_frames - 1}], "
                f"got {stop_frame}"
            )
        train_indices = np.arange(start_frame, stop_frame)
        test_indices = np.arange(stop_frame, num_frames)
        return train_indices, test_indices

    valid_frames = np.arange(start_frame, num_frames)
    split_point = int(len(valid_frames) * (1 - test_ratio))

    train_indices = valid_frames[:split_point]
    test_indices = valid_frames[split_point:]

    return train_indices, test_indices


@dataclass
class SubjectData:
    """Static per-subject data container.

    Holds all subject-specific data that remains constant across frames,
    including SMPL shape parameters, personal vertex offsets, and camera
    calibration information.

    Args:
        name: Subject identifier (e.g., "male-3-casual")
        betas: SMPL shape coefficients, shape (10,)
        v_personal: Per-vertex position offsets from canonical SMPL, shape (6890, 3)
        intrinsic: 3x3 camera intrinsic matrix
        extrinsic: 4x4 world-to-camera transform matrix
        distortion: Lens distortion coefficients (k1, k2, p1, p2, k3), shape (5,)
        image_size: Original image dimensions as (height, width)
        num_frames: Total number of frames for this subject
        train_indices: Frame indices belonging to training split
        test_indices: Frame indices belonging to test split
    """

    name: str
    betas: np.ndarray
    v_personal: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    distortion: np.ndarray
    image_size: Tuple[int, int]
    num_frames: int
    train_indices: np.ndarray
    test_indices: np.ndarray


class BaseDataset(ABC, Dataset):
    """Abstract base class for all dataset implementations.

    Provides the interface that all dataset implementations must follow.
    Inherits from both ABC for abstract method enforcement and PyTorch's
    Dataset for DataLoader compatibility.

    Subclasses must implement:
        - __len__: Return total number of samples
        - __getitem__: Return a sample dictionary for a given index
        - get_subject_data: Return static data for a given subject
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary for the given index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing at minimum:
                - "image": (3, H, W) float32 tensor in [0, 1]
                - "mask": (H, W) float32 tensor in {0, 1}
                - "pose": (72,) float32 tensor of SMPL pose parameters
                - "betas": (10,) float32 tensor of SMPL shape coefficients
                - "trans": (3,) float32 tensor of root translation
                - "intrinsic": (3, 3) float32 camera intrinsic matrix
                - "extrinsic": (4, 4) float32 world-to-camera transform
                - "frame_idx": int frame index within subject
                - "subject": str subject identifier
        """
        pass

    @abstractmethod
    def get_subject_data(self, subject: str) -> SubjectData:
        """Get all static data for a subject.

        Args:
            subject: Subject identifier (e.g., "male-3-casual")

        Returns:
            SubjectData containing betas, v_personal, camera info, etc.
        """
        pass
