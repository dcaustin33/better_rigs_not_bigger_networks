"""MHR dataset implementation.

This module provides dataset classes for loading and serving data from
MHR (Momentum Human Rig) datasets for training Gaussian avatars.

Supports two directory layouts:

1. Standard MHR format (MHRDataset):
    <data_root>/<subject>/
        model_params.npz   — MHR parameters (model_parameters, identity_coeffs, ...)
        camera.pkl         — Camera calibration (PeopleSnapshot format)
        images/            — Per-frame PNGs: frame_000000.png, frame_000001.png, ...
        masks.npz          — Binary masks with key "masks": (T, H, W) uint8

2. Corrected PeopleSnapshot format with MHR fits (MHRCorrectedDataset):
    <data_root>/<subject>/
        cam000/
            camera.pkl     — Corrected camera calibration (rotation matrix format)
            images/        — 1-indexed PNGs: 000001.png, 000002.png, ...
            mhr/
                raw/       — Per-frame .npz: 000001.npz, 000002.npz, ...
                shape.npy  — (45,) identity coefficients
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from gaussian_avatar.data.base import BaseDataset, SubjectData, compute_split_indices
from gaussian_avatar.data.camera import (
    build_extrinsic_from_matrix,
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    load_camera,
    load_camera_corrected,
    scale_intrinsic,
    undistort_image,
)
from gaussian_avatar.data.mhr_loaders import load_mhr_params
from gaussian_avatar.data.people_snapshot import preprocess_image, preprocess_mask
from gaussian_avatar.data.smpl_loaders import MaskReader

class MHRDataset(BaseDataset):
    """Dataset for MHR monocular video sequences.

    Loads and serves synchronized (image, mask, pose, camera) tuples for
    training Gaussian avatars with MHR body model. Compatible with the
    existing trainer interface via matching batch key names.

    Args:
        data_root: Path to MHR dataset root directory
        subjects: List of subject names, single subject name, or "all"
        split: Data split ("train", "test", or "all")
        image_size: Target (H, W) for resizing images
        preprocess_masks: Whether to apply erosion and Gaussian blur to masks
            before resizing (default True).

    Raises:
        FileNotFoundError: If data_root does not exist
        ValueError: If any specified subject does not exist
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        subjects: Union[List[str], str] = "all",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        preprocess_masks: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        self.image_size = image_size
        self.split = split
        self.preprocess_masks = preprocess_masks

        self._subjects = self._parse_subjects(subjects)

        self._subject_data: Dict[str, SubjectData] = {}
        self._model_params: Dict[str, np.ndarray] = {}
        self._identity_coeffs: Dict[str, np.ndarray] = {}
        self._masks: Dict[str, np.ndarray] = {}

        for subject in self._subjects:
            self._load_subject(subject)

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
            subject_dirs = [
                d.name
                for d in self.data_root.iterdir()
                if d.is_dir()
                and not d.name.startswith(".")
                and not d.name.startswith("__")
            ]
            return sorted(subject_dirs)

        if isinstance(subjects, str):
            subjects = [subjects]

        for subject in subjects:
            if not (self.data_root / subject).exists():
                raise ValueError(f"Subject not found: {subject}")

        return list(subjects)

    def _load_subject(self, subject: str) -> None:
        """Load all data for a single subject.

        Args:
            subject: Subject name to load
        """
        subject_dir = self.data_root / subject

        params = load_mhr_params(subject_dir / "model_params.npz")
        self._model_params[subject] = params["model_parameters"]
        self._identity_coeffs[subject] = params["identity_coeffs"]
        num_frames = len(params["model_parameters"])

        camera_data = load_camera(subject_dir / "camera.pkl")
        intrinsic = build_intrinsic_matrix(
            camera_data["focal"],
            camera_data["principal"],
        )
        extrinsic = build_extrinsic_matrix(
            camera_data["rotation"],
            camera_data["translation"],
        )
        original_size = (camera_data["height"], camera_data["width"])

        masks_data = np.load(subject_dir / "masks.npz")
        self._masks[subject] = masks_data["masks"]

        train_indices, test_indices = compute_split_indices(num_frames)

        self._subject_data[subject] = SubjectData(
            name=subject,
            betas=self._identity_coeffs[subject],
            v_personal=np.zeros((0, 3), dtype=np.float32),  # MHR has no v_personal
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            distortion=camera_data["distortion"],
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

            if self.split == "train":
                indices = subject_data.train_indices
            elif self.split == "test":
                indices = subject_data.test_indices
            else:  # "all"
                indices = np.arange(subject_data.num_frames)

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary for the given index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - "image": (3, H, W) float32 tensor in [0, 1]
                - "mask": (H, W) float32 tensor in [0, 1]
                - "pose": (204,) float32 tensor of MHR model parameters
                - "betas": (45,) float32 tensor of MHR identity coefficients
                - "intrinsic": (3, 3) float32 camera intrinsic matrix (scaled)
                - "extrinsic": (4, 4) float32 world-to-camera transform
                - "frame_idx": int frame index within subject
                - "subject": str subject identifier

        Raises:
            IndexError: If idx is out of bounds
        """
        subject, frame_idx = self.get_sample_info(idx)
        subject_data = self._subject_data[subject]

        image = self._load_image(subject, frame_idx)
        image_tensor = preprocess_image(image, self.image_size)

        mask = self._masks[subject][frame_idx]
        mask_tensor = preprocess_mask(mask, self.image_size, smooth=self.preprocess_masks)

        scaled_intrinsic = scale_intrinsic(
            subject_data.intrinsic,
            subject_data.image_size,
            self.image_size,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "pose": torch.from_numpy(self._model_params[subject][frame_idx]),
            "betas": torch.from_numpy(self._identity_coeffs[subject]),
            "intrinsic": torch.from_numpy(scaled_intrinsic),
            "extrinsic": torch.from_numpy(subject_data.extrinsic),
            "frame_idx": frame_idx,
            "subject": subject,
        }

    def _load_image(self, subject: str, frame_idx: int) -> np.ndarray:
        """Load an RGB image for a subject and frame.

        Reads from images/ directory as frame_NNNNNN.png.

        Args:
            subject: Subject identifier
            frame_idx: Frame index to load

        Returns:
            (H, W, 3) uint8 RGB image

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        image_path = (
            self.data_root / subject / "images" / f"frame_{frame_idx:06d}.png"
        )
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_subject_data(self, subject: str) -> SubjectData:
        """Get all static data for a subject.

        Args:
            subject: Subject identifier

        Returns:
            SubjectData containing betas, camera info, etc.

        Raises:
            KeyError: If subject is not loaded
        """
        if subject not in self._subject_data:
            raise KeyError(f"Subject not loaded: {subject}")
        return self._subject_data[subject]

def create_mhr_dataloaders(
    data_root: str,
    subjects: Union[List[str], str],
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 1,
    num_workers: int = 4,
    preprocess_masks: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test DataLoaders for an MHR dataset.

    Args:
        data_root: Path to MHR dataset root directory
        subjects: List of subject names, single subject name, or "all"
        image_size: Target (H, W) for resizing images
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes for data loading
        preprocess_masks: Whether to apply erosion and blur to masks

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from gaussian_avatar.data.people_snapshot import custom_collate

    train_dataset = MHRDataset(
        data_root=data_root,
        subjects=subjects,
        split="train",
        image_size=image_size,
        preprocess_masks=preprocess_masks,
    )
    test_dataset = MHRDataset(
        data_root=data_root,
        subjects=subjects,
        split="test",
        image_size=image_size,
        preprocess_masks=preprocess_masks,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    return train_loader, test_loader

class MHRCorrectedDataset(BaseDataset):
    """Dataset for MHR fits on the corrected PeopleSnapshot format.

    Loads pre-extracted PNG images from cam000/images/ and per-frame MHR
    parameter .npz files from cam000/mhr/raw/. Masks are derived from
    the image alpha channel (corrected dataset uses transparent backgrounds).

    Camera intrinsics are derived from the sam3d inference that produced the
    MHR fits. The focal length is read from the NPZ ``focal_length`` field
    when available; otherwise it is estimated from the 2D/3D keypoint pairs
    stored in each frame. The principal point is set to image center, matching
    the sam3d projection convention.

    Args:
        data_root: Path to corrected PeopleSnapshot directory
        subjects: List of subject names, single subject name, or "all"
        split: Data split ("train", "test", or "all")
        image_size: Target (H, W) for resizing images
        undistort: Whether to undistort images using camera distortion coefficients
        stop_frame: Optional frame index separating train from test.
            Frames [0, stop_frame) are train, [stop_frame, N) are test.
        skip: Optional frame skip stride (e.g. skip=4 takes every 4th frame)
        preprocess_masks: Whether to apply erosion and Gaussian blur to masks
            before resizing (default True).

    Raises:
        FileNotFoundError: If data_root does not exist
        ValueError: If any specified subject lacks the expected directory structure
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        subjects: Union[List[str], str] = "all",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        undistort: bool = False,
        stop_frame: Optional[int] = None,
        skip: Optional[int] = None,
        preprocess_masks: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        self.image_size = image_size
        self.split = split
        self.undistort = undistort
        self.stop_frame = stop_frame
        self.preprocess_masks = preprocess_masks
        self.skip = skip

        if skip is not None and skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")

        self._subjects = self._parse_subjects(subjects)

        self._subject_data: Dict[str, SubjectData] = {}
        self._model_params: Dict[str, np.ndarray] = {}
        self._identity_coeffs: Dict[str, np.ndarray] = {}
        self._trans: Dict[str, np.ndarray] = {}
        self._focal_lengths: Dict[str, np.ndarray] = {}
        self._mask_readers: Dict[str, Optional[MaskReader]] = {}

        for subject in self._subjects:
            self._load_subject(subject)

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
            ValueError: If any specified subject lacks cam000/mhr/ directory
        """
        if subjects == "all":
            subject_dirs = [
                d.name
                for d in self.data_root.iterdir()
                if d.is_dir()
                and not d.name.startswith(".")
                and not d.name.startswith("__")
                and (d / "cam000" / "mhr" / "raw").is_dir()
            ]
            return sorted(subject_dirs)

        if isinstance(subjects, str):
            subjects = [subjects]

        for subject in subjects:
            mhr_dir = self.data_root / subject / "cam000" / "mhr" / "raw"
            if not mhr_dir.is_dir():
                raise ValueError(
                    f"Subject '{subject}' missing cam000/mhr/raw/ directory"
                )

        return list(subjects)

    @staticmethod
    def _estimate_focal_from_keypoints(
        kp_2d: np.ndarray, kp_3d: np.ndarray, cam_t: np.ndarray,
        cx: float, cy: float,
    ) -> Optional[float]:
        """Estimate focal length from 2D/3D keypoint correspondence.

        Uses the perspective projection model::

            x_pixel = focal * (x_3d + tx) / (z_3d + tz) + cx

        Solves for focal across all valid keypoints and returns the median.

        Args:
            kp_2d: (K, 2) projected 2D keypoints in pixels
            kp_3d: (K, 3) 3D keypoints in the same frame as pred_vertices
            cam_t: (3,) camera translation [tx, ty, tz]
            cx: Principal point x (pixels)
            cy: Principal point y (pixels)

        Returns:
            Estimated focal length in pixels, or None if estimation fails
        """
        kp_tr = kp_3d + cam_t[np.newaxis, :]
        depth = kp_tr[:, 2]
        valid = depth > 0.1

        focals = []
        for axis in [0, 1]:
            center = cx if axis == 0 else cy
            dv = kp_tr[valid, axis]
            dp = kp_2d[valid, axis] - center
            mask = np.abs(dv) > 1e-4
            if mask.sum() > 0:
                f_est = dp[mask] * depth[valid][mask] / dv[mask]
                focals.extend(f_est.tolist())

        return float(np.median(focals)) if focals else None

    def _load_subject(self, subject: str) -> None:
        """Load all data for a single subject.

        Pre-loads all per-frame MHR model parameters, camera translations,
        and focal lengths into numpy arrays. The MHR TorchScript model outputs
        vertices in centimeters with Y-up convention, but the camera calibration
        expects meters with Y-down. We embed this coordinate conversion into the
        extrinsic matrix and convert per-frame translations to mesh space.

        Focal lengths come from one of two sources, in priority order:

        1. ``focal_length`` field in the per-frame NPZ (saved by sam3d wrapper)
        2. Estimated from ``pred_keypoints_2d``/``pred_keypoints_3d`` pairs

        The principal point is set to image center to match sam3d's projection
        convention.

        Args:
            subject: Subject name to load
        """
        subject_dir = self.data_root / subject
        mhr_dir = subject_dir / "cam000" / "mhr"
        raw_dir = mhr_dir / "raw"

        # Load camera for distortion coefficients and image size
        camera_data = load_camera_corrected(subject_dir / "cam000" / "camera.pkl")
        distortion = camera_data["distortion"]
        original_size = (camera_data["height"], camera_data["width"])

        # Build extrinsic that converts MHR mesh coordinates (cm, Y-up) to
        # camera coordinates (m, Y-down). The calibrated camera extrinsic is
        # typically identity for this dataset; we replace it with a
        # scale+flip transform: diag(0.01, -0.01, -0.01, 1.0).
        extrinsic = np.array([
            [0.01,  0.0,   0.0,  0.0],
            [0.0,  -0.01,  0.0,  0.0],
            [0.0,   0.0,  -0.01, 0.0],
            [0.0,   0.0,   0.0,  1.0],
        ], dtype=np.float32)

        shape_path = mhr_dir / "shape.npy"
        identity_coeffs = np.load(str(shape_path)).astype(np.float32)
        self._identity_coeffs[subject] = identity_coeffs

        # Principal point = image center (matches sam3d projection convention)
        H, W = original_size
        cx, cy = W / 2.0, H / 2.0

        # Count and load per-frame MHR parameters, translations, focal lengths
        npz_files = sorted(raw_dir.glob("*.npz"))
        num_frames = len(npz_files)

        model_params_list = []
        trans_list = []
        focal_list = []
        for npz_path in npz_files:
            data = np.load(str(npz_path))
            model_params_list.append(
                np.array(data["mhr_model_params"], dtype=np.float32)
            )
            # Convert pred_cam_t from camera space (m, Y-down) to mesh space
            # (cm, Y-up): multiply by [100, -100, -100]
            cam_t = np.array(data["pred_cam_t"], dtype=np.float32)
            trans_list.append(cam_t * np.array([100.0, -100.0, -100.0], dtype=np.float32))

            # Extract or estimate focal length
            if "focal_length" in data:
                focal_list.append(float(data["focal_length"]))
            elif "pred_keypoints_2d" in data and "pred_keypoints_3d" in data:
                f_est = self._estimate_focal_from_keypoints(
                    data["pred_keypoints_2d"], data["pred_keypoints_3d"],
                    cam_t, cx, cy,
                )
                focal_list.append(f_est if f_est is not None else np.nan)
            else:
                focal_list.append(np.nan)

        self._model_params[subject] = np.stack(model_params_list, axis=0)  # (N, 204)
        self._trans[subject] = np.stack(trans_list, axis=0)  # (N, 3)

        # Resolve focal lengths: fill any NaN gaps with the sequence median
        focal_arr = np.array(focal_list, dtype=np.float32)
        valid_mask = ~np.isnan(focal_arr)
        if valid_mask.any():
            median_focal = float(np.median(focal_arr[valid_mask]))
            focal_arr[~valid_mask] = median_focal
        else:
            # Fallback: use camera.pkl focal length
            focal_arr[:] = float(camera_data["focal"][0])
        self._focal_lengths[subject] = focal_arr  # (N,)

        # Build intrinsic using the sequence median focal and image center.
        # Per-frame intrinsics are constructed in __getitem__ when needed.
        median_focal = float(np.median(focal_arr))
        intrinsic = build_intrinsic_matrix(
            np.array([median_focal, median_focal], dtype=np.float64),
            np.array([cx, cy], dtype=np.float64),
        )

        # Load HDF5 masks if available (preferred over alpha-derived masks)
        masks_hdf5_path = subject_dir / "masks.hdf5"
        if masks_hdf5_path.exists():
            self._mask_readers[subject] = MaskReader(masks_hdf5_path)
            print(
                f"Using HDF5 masks for subject '{subject}' "
                f"({len(self._mask_readers[subject])} masks)"
            )
        else:
            self._mask_readers[subject] = None

        train_indices, test_indices = compute_split_indices(
            num_frames, stop_frame=self.stop_frame
        )

        self._subject_data[subject] = SubjectData(
            name=subject,
            betas=identity_coeffs,
            v_personal=np.zeros((0, 3), dtype=np.float32),
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

            if self.split == "train":
                indices = subject_data.train_indices
            elif self.split == "test":
                indices = subject_data.test_indices
            else:  # "all"
                indices = np.concatenate([
                    subject_data.train_indices,
                    subject_data.test_indices,
                ])
                indices = np.sort(indices)

            if self.skip is not None and self.skip > 1:
                indices = indices[::self.skip]

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary for the given index.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - "image": (3, H, W) float32 tensor in [0, 1]
                - "mask": (H, W) float32 tensor in [0, 1]
                - "pose": (204,) float32 tensor of MHR model parameters
                - "betas": (45,) float32 tensor of MHR identity coefficients
                - "trans": (3,) float32 tensor of root translation (mesh space)
                - "intrinsic": (3, 3) float32 camera intrinsic matrix (scaled)
                - "extrinsic": (4, 4) float32 world-to-camera transform
                - "frame_idx": int frame index within subject
                - "subject": str subject identifier
        """
        subject, frame_idx = self.get_sample_info(idx)
        subject_data = self._subject_data[subject]

        # Load image (1-indexed filenames, supports JPG and PNG)
        images_dir = self.data_root / subject / "cam000" / "images"
        frame_name = f"{frame_idx + 1:06d}"
        image_path = images_dir / f"{frame_name}.jpg"
        if not image_path.exists():
            image_path = images_dir / f"{frame_name}.png"
        image_bgra = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image_bgra is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        if image_bgra.ndim == 3 and image_bgra.shape[2] == 4:
            image = cv2.cvtColor(image_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
            image_raw = image_bgra  # keep for alpha fallback
        elif image_bgra.ndim == 3 and image_bgra.shape[2] == 3:
            image = cv2.cvtColor(image_bgra, cv2.COLOR_BGR2RGB)
            image_raw = image_bgra
        else:
            raise ValueError(
                f"Unexpected image format at {image_path}: "
                f"shape={image_bgra.shape}, expected 3 or 4 channels"
            )

        # Load mask: prefer HDF5, fall back to alpha channel or all-foreground
        mask_reader = self._mask_readers.get(subject)
        if mask_reader is not None:
            mask = mask_reader[frame_idx]
        elif image_raw.ndim == 3 and image_raw.shape[2] == 4:
            alpha = image_raw[:, :, 3]
            mask = (alpha > 0).astype(np.int8)
        else:
            mask = np.ones(image.shape[:2], dtype=np.int8)

        if self.undistort:
            image = undistort_image(
                image, subject_data.intrinsic, subject_data.distortion
            )
            mask = undistort_image(
                mask, subject_data.intrinsic, subject_data.distortion
            )

        image_tensor = preprocess_image(image, self.image_size)
        mask_tensor = preprocess_mask(mask, self.image_size, smooth=self.preprocess_masks)

        # Build per-frame intrinsic from the frame's focal length and
        # image center as principal point (sam3d convention).
        H, W = subject_data.image_size
        focal = float(self._focal_lengths[subject][frame_idx])
        frame_intrinsic = build_intrinsic_matrix(
            np.array([focal, focal], dtype=np.float64),
            np.array([W / 2.0, H / 2.0], dtype=np.float64),
        )

        scaled_intrinsic = scale_intrinsic(
            frame_intrinsic,
            subject_data.image_size,
            self.image_size,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "pose": torch.from_numpy(self._model_params[subject][frame_idx]),
            "betas": torch.from_numpy(self._identity_coeffs[subject]),
            "trans": torch.from_numpy(self._trans[subject][frame_idx]),
            "intrinsic": torch.from_numpy(scaled_intrinsic),
            "extrinsic": torch.from_numpy(subject_data.extrinsic),
            "frame_idx": frame_idx,
            "subject": subject,
        }

    def get_subject_data(self, subject: str) -> SubjectData:
        """Get all static data for a subject.

        Args:
            subject: Subject identifier

        Returns:
            SubjectData containing betas, camera info, etc.

        Raises:
            KeyError: If subject is not loaded
        """
        if subject not in self._subject_data:
            raise KeyError(f"Subject not loaded: {subject}")
        return self._subject_data[subject]

def create_mhr_corrected_dataloaders(
    data_root: str,
    subjects: Union[List[str], str],
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 1,
    num_workers: int = 4,
    undistort: bool = False,
    stop_frame: Optional[int] = None,
    skip: Optional[int] = None,
    preprocess_masks: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test DataLoaders for MHR corrected format dataset.

    Args:
        data_root: Path to corrected PeopleSnapshot root directory
        subjects: List of subject names, single subject name, or "all"
        image_size: Target (H, W) for resizing images
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes for data loading
        undistort: Whether to undistort images
        stop_frame: Optional train/test split boundary
        skip: Optional frame skip stride
        preprocess_masks: Whether to apply erosion and blur to masks

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from gaussian_avatar.data.people_snapshot import custom_collate

    train_dataset = MHRCorrectedDataset(
        data_root=data_root,
        subjects=subjects,
        split="train",
        image_size=image_size,
        undistort=undistort,
        stop_frame=stop_frame,
        skip=skip,
        preprocess_masks=preprocess_masks,
    )
    test_dataset = MHRCorrectedDataset(
        data_root=data_root,
        subjects=subjects,
        split="test",
        image_size=image_size,
        undistort=undistort,
        stop_frame=stop_frame,
        skip=skip,
        preprocess_masks=preprocess_masks,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    return train_loader, test_loader
