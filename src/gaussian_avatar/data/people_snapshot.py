"""PeopleSnapshot dataset implementation.

This module provides the main dataset class for loading and serving data
from the PeopleSnapshot dataset for training Gaussian avatars.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from gaussian_avatar.data.base import BaseDataset, SubjectData, compute_split_indices
from gaussian_avatar.data.camera import (
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    load_camera,
    scale_intrinsic,
    undistort_image,
)
from gaussian_avatar.data.smpl_loaders import (
    KeypointsReader,
    MaskReader,
    load_consensus,
    load_poses,
    pose_to_mask_index,
)
from gaussian_avatar.data.video import FrameCache, extract_frame, get_video_frame_count

def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Preprocess RGB image for training.

    Resizes the image to target size, normalizes pixel values to [0, 1],
    and converts from HWC to CHW format.

    Args:
        image: (H, W, 3) uint8 RGB image
        target_size: (H, W) target resolution

    Returns:
        (3, H, W) float32 tensor in [0, 1]
    """
    h, w = target_size
    image = cv2.resize(image, (w, h))

    image_tensor = torch.from_numpy(image).float() / 255.0

    image_tensor = image_tensor.permute(2, 0, 1)

    return image_tensor

def preprocess_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
    smooth: bool = True,
) -> torch.Tensor:
    """Preprocess binary mask for training.

    Optionally applies erosion and blur at original resolution to clean up
    mask edges, then resizes to target size.

    Args:
        mask: (H, W) int8 array with values in {0, 1}
        target_size: (H, W) target resolution
        smooth: When True (default), apply 5×5 erosion and 3×3 Gaussian
            blur before resizing. When False, only resize and normalize.

    Returns:
        (H, W) float32 tensor in [0, 1]
    """
    # Convert to uint8 [0, 255] for OpenCV morphology/filter operations.
    # Handles both {0, 1} binary masks and already-scaled {0, 255} masks
    # (e.g. from undistort_image which scales binary masks before undistortion).
    if mask.max() > 1:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask.astype(np.uint8) * 255

    if smooth:
        # Erode to remove thin mask artifacts at original resolution
        erode_kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.erode(mask_u8, erode_kernel, iterations=1)

        # Blur to soften edges at original resolution
        mask_u8 = cv2.GaussianBlur(mask_u8, (3, 3), 0)

    h, w = target_size
    mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_tensor = torch.from_numpy(mask_u8).float() / 255.0

    return mask_tensor

class PeopleSnapshotDataset(BaseDataset):
    """Dataset for PeopleSnapshot monocular video sequences.

    Loads and serves synchronized (image, mask, pose, camera) tuples for training
    Gaussian avatars. Supports single or multiple subjects, train/test splits,
    and optional frame caching.

    Args:
        data_root: Path to people_snapshot_public directory
        subjects: List of subject names, single subject name, or "all" to load all
        split: Data split to use ("train", "test", or "all")
        image_size: Target (H, W) for resizing images
        cache_frames: Whether to cache extracted frames to disk
        cache_dir: Directory for frame cache (default: data_root/.cache)
        undistort: Whether to undistort images using camera_k
        load_keypoints: Whether to load 2D keypoints
        frame_indices: Optional list of specific frame indices to use. When provided,
            overrides the split logic entirely. All indices must be >= 1 since
            frame 0 has no corresponding mask.
        preprocess_masks: Whether to apply erosion and Gaussian blur to masks
            before resizing (default True).

    Raises:
        FileNotFoundError: If data_root does not exist
        ValueError: If any specified subject does not exist, or if frame_indices
            contains invalid values
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        subjects: Union[List[str], str] = "all",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        cache_frames: bool = True,
        cache_dir: Optional[str] = None,
        undistort: bool = False,
        load_keypoints: bool = False,
        frame_indices: Optional[List[int]] = None,
        preprocess_masks: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        self.image_size = image_size
        self.split = split
        self.cache_frames = cache_frames
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / ".cache"
        self.undistort = undistort
        self.load_keypoints = load_keypoints
        self.frame_indices = frame_indices
        self.preprocess_masks = preprocess_masks

        if frame_indices is not None:
            if not frame_indices:
                raise ValueError("frame_indices must be non-empty")
            for idx in frame_indices:
                if idx < 1:
                    raise ValueError(
                        f"frame_indices must all be >= 1 (frame 0 has no mask), got {idx}"
                    )

        self._subjects = self._parse_subjects(subjects)

        self._subject_data: Dict[str, SubjectData] = {}
        self._mask_readers: Dict[str, MaskReader] = {}
        self._keypoints_readers: Dict[str, KeypointsReader] = {}
        self._poses_data: Dict[str, Dict[str, np.ndarray]] = {}

        for subject in self._subjects:
            self._load_subject(subject)

        # Initialize frame cache eagerly to avoid race conditions with
        # multi-worker DataLoader (each worker gets a copy via fork/spawn,
        # but lazy init in __getitem__ could cause concurrent writes).
        if self.cache_frames:
            self._frame_cache = FrameCache(self.cache_dir)

        self._build_index_mapping()

    @property
    def subjects(self) -> List[str]:
        """Return list of loaded subject names."""
        return self._subjects

    def _parse_subjects(self, subjects: Union[List[str], str]) -> List[str]:
        """Parse subject parameter into list of subject names.

        Args:
            subjects: "all", a single subject name, or list of subject names

        Returns:
            List of subject names to load

        Raises:
            ValueError: If any specified subject does not exist
        """
        if subjects == "all":
            # Find all subject directories
            subject_dirs = [
                d.name for d in self.data_root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            return sorted(subject_dirs)

        if isinstance(subjects, str):
            subjects = [subjects]

        for subject in subjects:
            subject_dir = self.data_root / subject
            if not subject_dir.exists():
                raise ValueError(f"Subject not found: {subject}")

        return list(subjects)

    def _load_subject(self, subject: str) -> None:
        """Load all data for a single subject.

        Args:
            subject: Subject name to load
        """
        subject_dir = self.data_root / subject

        camera_data = load_camera(subject_dir / "camera.pkl")
        intrinsic = build_intrinsic_matrix(
            camera_data["focal"],
            camera_data["principal"],
        )
        extrinsic = build_extrinsic_matrix(
            camera_data["rotation"],
            camera_data["translation"],
        )
        distortion = camera_data["distortion"]
        original_size = (camera_data["height"], camera_data["width"])

        consensus_data = load_consensus(subject_dir / "consensus.pkl")
        betas = consensus_data["betas"].astype(np.float32)
        v_personal = consensus_data["v_personal"].astype(np.float32)

        poses_path = subject_dir / "reconstructed_poses.hdf5"
        poses_data = load_poses(poses_path)
        num_frames = len(poses_data["poses"])
        self._poses_data[subject] = poses_data

        # Cap num_frames at video length to avoid out-of-range frame access
        video_path = subject_dir / f"{subject}.mp4"
        if video_path.exists():
            video_frames = get_video_frame_count(video_path)
            num_frames = min(num_frames, video_frames)

        mask_path = subject_dir / "masks.hdf5"
        self._mask_readers[subject] = MaskReader(mask_path)

        if self.load_keypoints:
            keypoints_path = subject_dir / "keypoints.hdf5"
            self._keypoints_readers[subject] = KeypointsReader(keypoints_path)

        train_indices, test_indices = compute_split_indices(
            num_frames, start_frame=1
        )

        self._subject_data[subject] = SubjectData(
            name=subject,
            betas=betas,
            v_personal=v_personal,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            distortion=distortion,
            image_size=original_size,
            num_frames=num_frames,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    def _build_index_mapping(self) -> None:
        """Build global index mapping to (subject, local_frame_idx)."""
        self._index_mapping: List[Tuple[str, int]] = []

        for subject in self._subjects:
            subject_data = self._subject_data[subject]

            if self.frame_indices is not None:
                # Use explicitly provided frame indices, overriding split logic
                for idx in self.frame_indices:
                    if idx >= subject_data.num_frames:
                        raise ValueError(
                            f"frame_indices contains {idx} but subject "
                            f"'{subject}' only has {subject_data.num_frames} frames"
                        )
                indices = np.array(self.frame_indices)
            elif self.split == "train":
                indices = subject_data.train_indices
            elif self.split == "test":
                indices = subject_data.test_indices
            else:  # "all"
                indices = np.concatenate([
                    subject_data.train_indices,
                    subject_data.test_indices,
                ])
                indices = np.sort(indices)

            for frame_idx in indices:
                self._index_mapping.append((subject, int(frame_idx)))

    def __len__(self) -> int:
        """Return total number of samples across all subjects."""
        return len(self._index_mapping)

    def get_sample_info(self, idx: int) -> Tuple[str, int]:
        """Get subject and local frame index for a global index.

        Args:
            idx: Global sample index

        Returns:
            Tuple of (subject_name, local_frame_idx)

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= len(self._index_mapping):
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {len(self)} samples"
            )
        return self._index_mapping[idx]

    def get_subject_for_idx(self, idx: int) -> str:
        """Get subject name for a global index.

        Args:
            idx: Global sample index

        Returns:
            Subject name string

        Raises:
            IndexError: If idx is out of bounds
        """
        subject, _ = self.get_sample_info(idx)
        return subject

    def get_subject_frame_count(self, subject: str) -> int:
        """Get the number of frames for a subject in the current split.

        Args:
            subject: Subject identifier (e.g., "male-3-casual")

        Returns:
            Number of frames for this subject in the current split

        Raises:
            KeyError: If subject is not loaded
        """
        if subject not in self._subject_data:
            raise KeyError(f"Subject not loaded: {subject}")

        # Count frames from this subject in the index mapping
        count = sum(1 for s, _ in self._index_mapping if s == subject)
        return count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary for the given index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - "image": (3, H, W) float32 tensor in [0, 1]
                - "mask": (H, W) float32 tensor in {0, 1}
                - "pose": (72,) float32 tensor of SMPL pose parameters
                - "betas": (10,) float32 tensor of SMPL shape coefficients
                - "trans": (3,) float32 tensor of root translation
                - "intrinsic": (3, 3) float32 camera intrinsic matrix (scaled for image_size)
                - "extrinsic": (4, 4) float32 world-to-camera transform
                - "frame_idx": int frame index within subject
                - "subject": str subject identifier
                - "keypoints": (18, 3) float32 tensor of 2D body keypoints (if load_keypoints=True)
                  Each row contains [x, y, confidence] following OpenPose 18-point body model.

        Raises:
            IndexError: If idx is out of bounds
        """
        subject, frame_idx = self.get_sample_info(idx)
        subject_data = self._subject_data[subject]

        image = self._load_image(subject, frame_idx)

        if self.undistort:
            image = undistort_image(
                image, subject_data.intrinsic, subject_data.distortion
            )

        image_tensor = preprocess_image(image, self.image_size)

        mask_idx = pose_to_mask_index(frame_idx)
        mask = self._mask_readers[subject][mask_idx]

        # Apply same undistortion to mask so it stays aligned with the image
        if self.undistort:
            mask = undistort_image(
                mask, subject_data.intrinsic, subject_data.distortion
            )

        mask_tensor = preprocess_mask(mask, self.image_size, smooth=self.preprocess_masks)

        scaled_intrinsic = scale_intrinsic(
            subject_data.intrinsic,
            subject_data.image_size,
            self.image_size,
        )

        sample = {
            "image": image_tensor,
            "mask": mask_tensor,
            "pose": torch.from_numpy(self._poses_data[subject]["poses"][frame_idx]),
            "betas": torch.from_numpy(subject_data.betas),
            "trans": torch.from_numpy(self._poses_data[subject]["trans"][frame_idx]),
            "intrinsic": torch.from_numpy(scaled_intrinsic),
            "extrinsic": torch.from_numpy(subject_data.extrinsic),
            "frame_idx": frame_idx,
            "subject": subject,
        }

        # Add keypoints if enabled (same frame alignment as masks)
        if self.load_keypoints:
            keypoint_idx = pose_to_mask_index(frame_idx)  # Same as mask alignment
            keypoints = self._keypoints_readers[subject][keypoint_idx]
            sample["keypoints"] = torch.from_numpy(keypoints)

        return sample

    def _load_image(self, subject: str, frame_idx: int) -> np.ndarray:
        """Load an image for a subject and frame.

        Checks the frame cache first if caching is enabled.

        Args:
            subject: Subject identifier
            frame_idx: Frame index to load

        Returns:
            (H, W, 3) uint8 RGB image
        """
        if self.cache_frames:
            cached = self._frame_cache.get(subject, frame_idx)
            if cached is not None:
                return cached

        video_path = self.data_root / subject / f"{subject}.mp4"
        image = extract_frame(video_path, frame_idx)

        # Cache if enabled (skip_existing=True prevents race conditions
        # when multiple DataLoader workers cache the same frame)
        if self.cache_frames:
            self._frame_cache.put(subject, frame_idx, image)

        return image

    def get_subject_data(self, subject: str) -> SubjectData:
        """Get all static data for a subject.

        Args:
            subject: Subject identifier (e.g., "male-3-casual")

        Returns:
            SubjectData containing betas, v_personal, camera info, etc.

        Raises:
            KeyError: If subject is not loaded
        """
        if subject not in self._subject_data:
            raise KeyError(f"Subject not loaded: {subject}")
        return self._subject_data[subject]

    def get_canonical_mesh(self, subject: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get consensus mesh vertices for a subject.

        Note: Faces are not stored per-subject and should come from SMPL model.

        Args:
            subject: Subject identifier

        Returns:
            v_personal array (6890, 3) containing vertex offsets
        """
        subject_data = self.get_subject_data(subject)
        return subject_data.v_personal, None  # Faces from SMPL model

    def close(self) -> None:
        """Close all file handles."""
        if hasattr(self, "_mask_readers"):
            for reader in self._mask_readers.values():
                reader.close()

    def __del__(self) -> None:
        """Clean up on deletion."""
        self.close()

def custom_collate(batch: List[Dict]) -> Dict:
    """Collate a batch of samples for DataLoader.

    Stacks tensor values into batched tensors and collects non-tensor values
    (like strings and ints) into lists.

    Args:
        batch: List of sample dictionaries from the dataset

    Returns:
        Dictionary with batched tensors and lists of non-tensor values

    Raises:
        ValueError: If batch is empty
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")

    result = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            # Keep non-tensors as lists (e.g., subject names, frame indices)
            result[key] = values

    return result

def _detect_dataset_format(data_root: str, subjects: Union[List[str], str]) -> str:
    """Auto-detect dataset format by checking directory structure.

    The corrected format has a cam000/ subdirectory in each subject folder,
    while the public format has video files and HDF5 files directly.

    Args:
        data_root: Path to dataset root
        subjects: Subject names to check

    Returns:
        "corrected" or "public"
    """
    root = Path(data_root)

    if subjects == "all":
        subject_dirs = [
            d for d in root.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and not d.name.startswith("__")
        ]
        if not subject_dirs:
            return "public"
        check_dir = subject_dirs[0]
    else:
        if isinstance(subjects, str):
            check_dir = root / subjects
        else:
            check_dir = root / subjects[0]

    # Corrected format has cam000/ subdirectory
    if (check_dir / "cam000").is_dir():
        return "corrected"
    return "public"

def create_dataloaders(
    data_root: str,
    subjects: Union[List[str], str],
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 1,
    num_workers: int = 4,
    dataset_format: str = "auto",
    **dataset_kwargs,
) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """Create train and test DataLoaders for PeopleSnapshot dataset.

    Factory function that creates properly configured DataLoaders with recommended
    settings for training Gaussian avatars. Supports both the original public
    format and the corrected format with pre-extracted images.

    Args:
        data_root: Path to dataset directory
        subjects: List of subject names, single subject name, or "all"
        image_size: Target (H, W) for resizing images (default: (512, 512))
        batch_size: Batch size for both loaders (default: 1 for single-image training)
        num_workers: Number of worker processes for data loading (default: 4)
        dataset_format: Dataset format to use: "auto" (detect from directory structure),
            "public" (original HDF5/video format), or "corrected" (pre-extracted PNGs
            with per-frame SMPL pickles). Default: "auto".
        **dataset_kwargs: Additional keyword arguments passed to the dataset class.
            For "public": cache_frames, cache_dir, undistort, load_keypoints, frame_indices
            For "corrected": undistort, frame_indices

    Returns:
        Tuple of (train_loader, test_loader) DataLoaders
    """
    from torch.utils.data import DataLoader

    if dataset_format == "auto":
        dataset_format = _detect_dataset_format(data_root, subjects)
        print(f"Auto-detected dataset format: {dataset_format}")

    if dataset_format == "corrected":
        from gaussian_avatar.data.people_snapshot_corrected import (
            PeopleSnapshotCorrectedDataset,
        )

        corrected_kwargs = {
            k: v for k, v in dataset_kwargs.items()
            if k in ("undistort", "frame_indices", "stop_frame", "skip", "model_type", "preprocess_masks")
        }

        train_dataset = PeopleSnapshotCorrectedDataset(
            data_root=data_root,
            subjects=subjects,
            split="train",
            image_size=image_size,
            **corrected_kwargs,
        )
        test_dataset = PeopleSnapshotCorrectedDataset(
            data_root=data_root,
            subjects=subjects,
            split="test",
            image_size=image_size,
            **corrected_kwargs,
        )
    else:
        train_dataset = PeopleSnapshotDataset(
            data_root=data_root,
            subjects=subjects,
            split="train",
            image_size=image_size,
            **dataset_kwargs,
        )
        test_dataset = PeopleSnapshotDataset(
            data_root=data_root,
            subjects=subjects,
            split="test",
            image_size=image_size,
            **dataset_kwargs,
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    # Use fewer workers for test and no shuffle for reproducible evaluation
    test_num_workers = num_workers if num_workers != 4 else 2
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    return train_loader, test_loader

