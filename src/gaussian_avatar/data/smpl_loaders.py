"""SMPL parameter loading utilities.

This module provides functions for loading SMPL parameters from PeopleSnapshot
dataset files including poses, shapes, translations, and consensus data.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union

import h5py
import numpy as np


class BaseHDF5Reader(ABC):
    """Abstract base class for memory-mapped HDF5 readers.

    Provides common functionality for reading data from HDF5 files using
    SWMR (single-writer-multiple-reader) mode for multi-process safety.

    File handles are opened lazily on first access so that forked DataLoader
    worker processes each get their own handle (HDF5 is not fork-safe).

    Subclasses must implement:
        - _get_dataset_key: Return the HDF5 dataset key to read
        - _process_item: Process and return data for a single index

    Args:
        file_path: Path to HDF5 file
        dataset_key: Key of the dataset within the HDF5 file

    Raises:
        FileNotFoundError: If file_path does not exist
    """

    def __init__(self, file_path: Union[str, Path], dataset_key: str) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._file_path = file_path
        self._dataset_key = dataset_key
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None

        # Eagerly validate: open, check key, read length, then close.
        # Workers will reopen lazily via _ensure_open().
        try:
            f = h5py.File(file_path, "r", swmr=True)
        except ValueError as e:
            if "SWMR" in str(e) or "swmr" in str(e):
                raise ValueError(
                    f"Failed to open '{file_path}' in SWMR mode. "
                    f"The file may not have been created with SWMR support. "
                    f"Try recreating the HDF5 file with libver='latest' or "
                    f"use h5py.File(..., libver='latest') when creating it. "
                    f"Original error: {e}"
                ) from e
            raise

        if dataset_key not in f:
            available_keys = list(f.keys())
            f.close()
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in '{file_path}'. "
                f"Available keys: {available_keys}"
            )

        self._length = f[dataset_key].shape[0]
        f.close()

    def _ensure_open(self) -> None:
        """Open the HDF5 file if not already open in this process."""
        if self._file is None:
            self._file = h5py.File(self._file_path, "r", swmr=True)
            self._dataset = self._file[self._dataset_key]

    def __len__(self) -> int:
        """Return the number of items available."""
        return self._length

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get a single item by index.

        Args:
            idx: Item index (0-indexed)

        Returns:
            Processed data array

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds (0-{len(self) - 1})")

        self._ensure_open()
        return self._process_item(idx)

    @abstractmethod
    def _process_item(self, idx: int) -> np.ndarray:
        """Process and return data for a single index.

        Args:
            idx: Valid item index

        Returns:
            Processed data array
        """
        pass

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset = None

    def __enter__(self) -> "BaseHDF5Reader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def load_poses(poses_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load SMPL pose parameters from an HDF5 file.

    Parses reconstructed_poses.hdf5 files from the PeopleSnapshot dataset,
    which contain per-frame SMPL pose parameters, shape coefficients,
    and root translations.

    Args:
        poses_path: Path to reconstructed_poses.hdf5 file

    Returns:
        Dictionary containing:
            - "poses": (T, 72) float32 array of SMPL pose parameters per frame
            - "betas": (10,) float32 array of shape coefficients (shared across frames)
            - "trans": (T, 3) float32 array of root translation per frame

    Raises:
        FileNotFoundError: If poses_path does not exist
    """
    poses_path = Path(poses_path)
    if not poses_path.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_path}")

    with h5py.File(poses_path, "r") as f:
        poses = np.array(f["pose"], dtype=np.float32)
        betas = np.array(f["betas"], dtype=np.float32)
        trans = np.array(f["trans"], dtype=np.float32)

    return {
        "poses": poses,
        "betas": betas,
        "trans": trans,
    }


def load_consensus(consensus_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load SMPL consensus data from a pickle file.

    Parses consensus.pkl files from the PeopleSnapshot dataset, which contain
    shape coefficients and per-vertex position offsets that capture
    subject-specific details like clothing and hair.

    Args:
        consensus_path: Path to consensus.pkl file

    Returns:
        Dictionary containing:
            - "betas": (10,) float64 array of shape coefficients
            - "v_personal": (6890, 3) float64 array of per-vertex offsets

    Raises:
        FileNotFoundError: If consensus_path does not exist
    """
    consensus_path = Path(consensus_path)
    if not consensus_path.exists():
        raise FileNotFoundError(f"Consensus file not found: {consensus_path}")

    with open(consensus_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return {
        "betas": data["betas"],
        "v_personal": data["v_personal"],
    }


def load_corrected_smpl_frame(pkl_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load per-frame SMPL parameters from a corrected PeopleSnapshot pickle file.

    Parses per-frame SMPL pickle files from the corrected dataset, which store
    global_orient (3,) and body_pose (69,) separately. These are concatenated
    into a single pose vector (72,) for compatibility with the training pipeline.

    Args:
        pkl_path: Path to per-frame SMPL pickle file (e.g., smpls/000001.pkl)

    Returns:
        Dictionary containing:
            - "pose": (72,) float32 array (global_orient + body_pose)
            - "betas": (10,) float32 array of shape coefficients
            - "trans": (3,) float32 array of root translation
            - "v_personal": (6890, 3) float32 array of per-vertex offsets

    Raises:
        FileNotFoundError: If pkl_path does not exist
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"SMPL frame file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    global_orient = np.array(data["global_orient"], dtype=np.float32)
    body_pose = np.array(data["body_pose"], dtype=np.float32)
    pose = np.concatenate([global_orient, body_pose], axis=0)

    return {
        "pose": pose,
        "betas": np.array(data["betas"], dtype=np.float32),
        "trans": np.array(data["transl"], dtype=np.float32),
        "v_personal": np.array(data["v_personal"], dtype=np.float32),
    }


class MaskReader(BaseHDF5Reader):
    """Memory-mapped reader for mask HDF5 files.

    Provides efficient access to foreground masks stored in HDF5 format,
    using SWMR (single-writer-multiple-reader) mode for multi-process safety.

    The mask file contains T-1 masks for T pose frames, where the first pose
    frame (index 0) has no corresponding mask.

    Args:
        mask_path: Path to masks.hdf5 file

    Raises:
        FileNotFoundError: If mask_path does not exist
    """

    def __init__(self, mask_path: Union[str, Path]) -> None:
        super().__init__(mask_path, "masks")

    def _process_item(self, idx: int) -> np.ndarray:
        """Get a single mask by index.

        Args:
            idx: Mask index (0-indexed)

        Returns:
            (H, W) int8 array with values in {0, 1}
        """
        return np.array(self._dataset[idx], dtype=np.int8)


def pose_to_mask_index(pose_idx: int) -> int:
    """Convert a pose frame index to the corresponding mask index.

    In PeopleSnapshot, masks have T-1 frames while poses have T frames.
    The first pose frame (index 0) has no corresponding mask.

    Args:
        pose_idx: Pose frame index (must be >= 1)

    Returns:
        Corresponding mask index (pose_idx - 1)

    Raises:
        ValueError: If pose_idx is 0 or negative
    """
    if pose_idx < 0:
        raise ValueError(f"Pose index must be non-negative, got {pose_idx}")
    if pose_idx == 0:
        raise ValueError("Pose frame 0 has no corresponding mask")

    return pose_idx - 1


def get_valid_frame_indices(num_frames: int) -> np.ndarray:
    """Get valid frame indices that have corresponding masks.

    Excludes frame 0 since it has no corresponding mask.

    Args:
        num_frames: Total number of frames (poses)

    Returns:
        Array of valid frame indices [1, 2, ..., num_frames-1]
    """
    return np.arange(1, num_frames)


class KeypointsReader(BaseHDF5Reader):
    """Memory-mapped reader for keypoints HDF5 files.

    Provides efficient access to 2D body keypoints stored in HDF5 format,
    using SWMR (single-writer-multiple-reader) mode for multi-process safety.

    The keypoints file contains T-1 frames for T pose frames, following the
    same frame alignment as masks (frame 0 has no corresponding keypoints).

    Keypoints are stored as (T-1, 54) where 54 = 18 keypoints × 3 (x, y, confidence).
    This reader reshapes them to (18, 3) per frame following the OpenPose 18-point
    body model convention.

    Args:
        keypoints_path: Path to keypoints.hdf5 file

    Raises:
        FileNotFoundError: If keypoints_path does not exist
    """

    def __init__(self, keypoints_path: Union[str, Path]) -> None:
        super().__init__(keypoints_path, "keypoints")

    def _process_item(self, idx: int) -> np.ndarray:
        """Get keypoints for a single frame, reshaped to (18, 3).

        Args:
            idx: Frame index (0-indexed, corresponds to mask indices)

        Returns:
            (18, 3) float32 array where each row is [x, y, confidence] for a keypoint
        """
        flat_keypoints = np.array(self._dataset[idx], dtype=np.float32)
        return flat_keypoints.reshape(18, 3)
