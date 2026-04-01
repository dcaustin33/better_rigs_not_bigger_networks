"""Corrected PeopleSnapshot dataset implementation.

This module provides a dataset class for loading data from the corrected
PeopleSnapshot dataset, which uses per-frame SMPL pickles and pre-extracted
PNG images instead of the original centralized HDF5 and video format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from gaussian_avatar.data.base import BaseDataset, SubjectData, compute_split_indices
from gaussian_avatar.data.camera import (
    build_extrinsic_from_matrix,
    build_intrinsic_matrix,
    load_camera_corrected,
    scale_intrinsic,
    undistort_image,
)
from gaussian_avatar.data.smpl_loaders import MaskReader

logger = logging.getLogger(__name__)
from gaussian_avatar.data.people_snapshot import preprocess_image, preprocess_mask

class PeopleSnapshotCorrectedDataset(BaseDataset):
    """Dataset for the corrected PeopleSnapshot format.

    Loads pre-extracted PNG images and per-frame SMPL pickle files from the
    corrected PeopleSnapshot dataset. Masks are loaded from HDF5 files when
    available (``masks.hdf5`` in the subject directory), falling back to the
    image alpha channel or all-foreground for 3-channel images.

    Args:
        data_root: Path to people_snapshot_corrected directory
        subjects: List of subject names, single subject name, or "all" to load all
        split: Data split to use ("train", "test", or "all")
        image_size: Target (H, W) for resizing images
        undistort: Whether to undistort images using camera distortion coefficients
        frame_indices: Optional list of specific frame indices to use.
            When provided, overrides the split logic entirely.
        stop_frame: Optional frame index separating train from test.
            Frames [0, stop_frame) are train, [stop_frame, N) are test.
            Overrides the default 75/25 split when provided.
        skip: Optional frame skip stride. When set, only every skip-th frame
            is used. For example, skip=4 takes every 4th frame from the
            split indices. Applied after split selection / frame_indices.
        preprocess_masks: Whether to apply erosion and Gaussian blur to masks
            before resizing (default True).

    Raises:
        FileNotFoundError: If data_root does not exist
        ValueError: If any specified subject does not exist, or if skip < 1
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        subjects: Union[List[str], str] = "all",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        undistort: bool = False,
        frame_indices: Optional[List[int]] = None,
        stop_frame: Optional[int] = None,
        skip: Optional[int] = None,
        model_type: str = "smpl",
        preprocess_masks: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        self.image_size = image_size
        self.split = split
        self.undistort = undistort
        self.frame_indices = frame_indices
        self.stop_frame = stop_frame
        self.skip = skip
        self.model_type = model_type
        self.preprocess_masks = preprocess_masks

        if skip is not None and skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")

        if frame_indices is not None:
            if not frame_indices:
                raise ValueError("frame_indices must be non-empty")

        self._subjects = self._parse_subjects(subjects)

        self._subject_data: Dict[str, SubjectData] = {}
        self._poses: Dict[str, np.ndarray] = {}
        self._trans: Dict[str, np.ndarray] = {}
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

        Filters out non-subject directories like __MACOSX and validates
        subjects by checking for the cam000/ subdirectory.

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
                and (d / "cam000").is_dir()
            ]
            return sorted(subject_dirs)

        if isinstance(subjects, str):
            subjects = [subjects]

        for subject in subjects:
            subject_dir = self.data_root / subject
            if not subject_dir.exists():
                raise ValueError(f"Subject not found: {subject}")
            if not (subject_dir / "cam000").is_dir():
                raise ValueError(
                    f"Subject '{subject}' missing cam000/ directory — "
                    f"not a valid corrected dataset subject"
                )

        return list(subjects)

    def _load_subject(self, subject: str) -> None:
        """Load all data for a single subject.

        Pre-loads all per-frame SMPL parameters into numpy arrays for fast
        access during training.

        Args:
            subject: Subject name to load
        """
        subject_dir = self.data_root / subject

        camera_data = load_camera_corrected(subject_dir / "cam000" / "camera.pkl")
        intrinsic = build_intrinsic_matrix(
            camera_data["focal"],
            camera_data["principal"],
        )
        extrinsic = build_extrinsic_from_matrix(
            camera_data["rotation_matrix"],
            camera_data["translation"],
        )
        distortion = camera_data["distortion"]
        original_size = (camera_data["height"], camera_data["width"])

        # Count frames from image files (supports both JPG and PNG)
        images_dir = subject_dir / "cam000" / "images"
        image_files = sorted(
            f for f in images_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        num_frames = len(image_files)

        # Pre-load all per-frame body model pickles
        # SMPL-X uses smplxs/ directory, SMPL uses smpls/
        if self.model_type == "smplx":
            fits_dir = subject_dir / "smplxs"
            if not fits_dir.is_dir():
                raise FileNotFoundError(
                    f"SMPL-X fits directory not found: {fits_dir}. "
                    f"Expected smplxs/ for model_type='smplx'."
                )
        else:
            fits_dir = subject_dir / "smpls"

        poses_list = []
        trans_list = []
        betas = None
        v_personal = None

        for i in range(num_frames):
            # Files are 1-indexed: 000001.pkl, 000002.pkl, ...
            pkl_path = fits_dir / f"{i + 1:06d}.pkl"
            smpl_data = self._load_smpl_frame(pkl_path)

            poses_list.append(smpl_data["pose"])
            trans_list.append(smpl_data["trans"])

            # Use first frame for constant data
            if betas is None:
                betas = smpl_data["betas"]
                v_personal = smpl_data["v_personal"]

        self._poses[subject] = np.stack(poses_list, axis=0)  # (N, pose_dim)
        trans_array = np.stack(trans_list, axis=0)  # (N, 3)

        # Fix translation: SMPL fits were generated from MHR/SAM3D fits using
        # SAM3D's internal camera (varies per frame, ~1400-1700 focal), but
        # camera.pkl has the actual dataset camera (~2664 focal). Rescale each
        # frame's translation using its per-frame SAM3D camera so the mesh
        # projects correctly under the dataset camera.
        mhr_raw_dir = subject_dir / "cam000" / "mhr" / "raw"
        if mhr_raw_dir.is_dir() and self.model_type == "mhr":
            dataset_focal = camera_data["focal"]  # (2,) array [fx, fy]
            dataset_center = camera_data["principal"]  # (2,) array [cx, cy]
            trans_array = self._rescale_translations_per_frame(
                trans_array, mhr_raw_dir, num_frames,
                dataset_focal, dataset_center,
            )

        self._trans[subject] = trans_array

        # Load HDF5 masks if available (preferred over alpha-derived masks)
        masks_hdf5_path = subject_dir / "masks.hdf5"
        if masks_hdf5_path.exists():
            self._mask_readers[subject] = MaskReader(masks_hdf5_path)
            logger.info(
                "Using HDF5 masks for subject '%s' (%d masks)",
                subject, len(self._mask_readers[subject]),
            )
        else:
            self._mask_readers[subject] = None

        train_indices, test_indices = compute_split_indices(
            num_frames, stop_frame=self.stop_frame
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

    @staticmethod
    def _load_smpl_frame(pkl_path: Path) -> Dict[str, np.ndarray]:
        """Load a single per-frame SMPL or SMPL-X pickle.

        Handles both SMPL (72-dim pose) and SMPL-X (66-dim pose) formats.

        Args:
            pkl_path: Path to per-frame body model pickle file

        Returns:
            Dictionary with pose, betas, trans, v_personal arrays
        """
        import pickle

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

    @staticmethod
    def _solve_sam3d_camera_for_frame(
        npz_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the SAM3D internal camera from one raw frame.

        SAM3D uses its own projection camera (not the dataset camera.pkl)
        which varies per frame. We recover [fx, fy] and [cx, cy] by
        least-squares on matched 2D-3D keypoints.

        Args:
            npz_path: Path to a single SAM3D .npz output file

        Returns:
            Tuple of (focal, center) as (2,) numpy arrays.
        """
        sam3d = dict(np.load(npz_path, allow_pickle=True))
        kp2d = sam3d["pred_keypoints_2d"]
        kp3d = sam3d["pred_keypoints_3d"] + sam3d["pred_cam_t"][None, :]

        n = len(kp2d)
        xz = kp3d[:, 0] / kp3d[:, 2]
        yz = kp3d[:, 1] / kp3d[:, 2]
        fx, cx = np.linalg.lstsq(
            np.stack([xz, np.ones(n)], axis=1), kp2d[:, 0], rcond=None
        )[0]
        fy, cy = np.linalg.lstsq(
            np.stack([yz, np.ones(n)], axis=1), kp2d[:, 1], rcond=None
        )[0]

        return (
            np.array([fx, fy], dtype=np.float64),
            np.array([cx, cy], dtype=np.float64),
        )

    @staticmethod
    def _rescale_translations_per_frame(
        trans: np.ndarray,
        mhr_raw_dir: Path,
        num_frames: int,
        dataset_focal: np.ndarray,
        dataset_center: np.ndarray,
    ) -> np.ndarray:
        """Rescale SMPL translations using per-frame SAM3D cameras.

        The SMPL fits have translations optimized for SAM3D's internal camera,
        which varies per frame (~1400-1700 focal). For each frame we solve the
        SAM3D camera from its keypoints and rescale the translation so the mesh
        projects to the same pixel locations under the dataset camera.

        Args:
            trans: (N, 3) array of translations
            mhr_raw_dir: Path to cam000/mhr/raw/ with per-frame .npz files
            num_frames: Number of frames (matching trans rows)
            dataset_focal: (2,) dataset focal lengths [fx, fy]
            dataset_center: (2,) dataset principal point [cx, cy]

        Returns:
            (N, 3) rescaled translations
        """
        result = trans.copy()
        rescaled_count = 0

        for i in range(num_frames):
            npz_path = mhr_raw_dir / f"{i + 1:06d}.npz"
            if not npz_path.exists():
                continue

            sam3d_focal, sam3d_center = (
                PeopleSnapshotCorrectedDataset._solve_sam3d_camera_for_frame(
                    npz_path
                )
            )

            old_tx, old_ty, old_tz = trans[i]
            scale = 0.5 * (
                dataset_focal[0] / sam3d_focal[0]
                + dataset_focal[1] / sam3d_focal[1]
            )

            new_tz = old_tz * scale
            new_tx = (
                sam3d_focal[0] * old_tx / old_tz
                + sam3d_center[0]
                - dataset_center[0]
            ) * new_tz / dataset_focal[0]
            new_ty = (
                sam3d_focal[1] * old_ty / old_tz
                + sam3d_center[1]
                - dataset_center[1]
            ) * new_tz / dataset_focal[1]

            result[i] = [new_tx, new_ty, new_tz]
            rescaled_count += 1

        result = result.astype(np.float32)
        logger.info(
            "Rescaled %d/%d frame translations (per-frame SAM3D camera)",
            rescaled_count,
            num_frames,
        )
        return result

    def _build_index_mapping(self) -> None:
        """Build global index mapping to (subject, local_frame_idx)."""
        self._index_mapping: List[Tuple[str, int]] = []

        for subject in self._subjects:
            subject_data = self._subject_data[subject]

            if self.frame_indices is not None:
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

    def get_subject_for_idx(self, idx: int) -> str:
        """Get subject name for a global index.

        Args:
            idx: Global sample index

        Returns:
            Subject name string
        """
        subject, _ = self.get_sample_info(idx)
        return subject

    def get_subject_frame_count(self, subject: str) -> int:
        """Get the number of frames for a subject in the current split.

        Args:
            subject: Subject identifier

        Returns:
            Number of frames for this subject in the current split
        """
        return sum(1 for s, _ in self._index_mapping if s == subject)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary for the given index.

        Loads the PNG image and mask (from HDF5 if available, otherwise from
        image alpha channel), returning the same sample format as
        PeopleSnapshotDataset for compatibility.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - "image": (3, H, W) float32 tensor in [0, 1]
                - "mask": (H, W) float32 tensor in [0, 1]
                - "pose": (72,) float32 tensor of SMPL pose parameters
                - "betas": (10,) float32 tensor of SMPL shape coefficients
                - "trans": (3,) float32 tensor of root translation
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
        image_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image_raw is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        if image_raw.ndim == 3 and image_raw.shape[2] == 4:
            image = cv2.cvtColor(image_raw[:, :, :3], cv2.COLOR_BGR2RGB)
        elif image_raw.ndim == 3 and image_raw.shape[2] == 3:
            image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(
                f"Unexpected image format at {image_path}: "
                f"shape={image_raw.shape}, expected 3 or 4 channels"
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

        scaled_intrinsic = scale_intrinsic(
            subject_data.intrinsic,
            subject_data.image_size,
            self.image_size,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "pose": torch.from_numpy(self._poses[subject][frame_idx]),
            "betas": torch.from_numpy(subject_data.betas),
            "trans": torch.from_numpy(self._trans[subject][frame_idx]),
            "intrinsic": torch.from_numpy(scaled_intrinsic),
            "extrinsic": torch.from_numpy(subject_data.extrinsic),
            "frame_idx": frame_idx,
            "subject": subject,
        }

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
