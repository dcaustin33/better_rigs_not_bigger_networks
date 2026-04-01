"""ZJU MoCap dataset implementation with multi-model support (MHR, SMPL, SMPL-X).

Multi-view dataset where each camera directory contains separate image
and mask subdirectories, plus per-frame body model fits in a results/
subdirectory.

Supports three model types:

- **MHR** (``model_type="mhr"``): Loads raw SAM3D MHR fits from
  ``results/raw/*.npz``. Translation converted from camera space
  (m, Y-down) to mesh space (cm, Y-up). Extrinsic is Y/Z-flip.
- **SMPL** (``model_type="smpl"``): Loads fitted SMPL parameters
  from ``results/smpl/smpl_params.npz``. Translation used directly
  (m, camera Y-down). Extrinsic is identity. Pose: 72-dim.
- **SMPL-X** (``model_type="smplx"``): Loads fitted SMPL-X parameters
  from ``results/smplx/smplx_params.npz``. Translation used directly
  (m, camera Y-down). Extrinsic is identity. Pose: 66-dim.

All modes use SAM3D's per-frame focal length (from ``results/raw/``)
for the intrinsic matrix. The multi-view ``annots.npy`` R,T are NOT
used for rendering — they only enumerate available cameras.

Directory layout::

    <data_root>/<subject>/
        annots.npy                  — camera calibration (K, D, R, T per camera)
        Camera_B1/
            images/
                000000.jpg, 000001.jpg, ...   — RGB images (JPG or PNG)
            masks/
                000000.png, 000001.png, ...   — binary mask PNGs
            results/
                shape.npy           — (45,) MHR identity coefficients
                raw/
                    000000.npz      — per-frame MHR params (mhr_model_params, pred_cam_t, ...)
                smpl/
                    smpl_params.npz  — fitted SMPL params (global_orient, body_pose, betas, transl)
                smplx/
                    smplx_params.npz — fitted SMPL-X params (global_orient, body_pose, betas, transl, ...)
                    fitting_errors.npy — per-frame fitting error
        Camera_B2/
            ...
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from gaussian_avatar.data.base import BaseDataset, SubjectData
from gaussian_avatar.data.camera import (
    build_intrinsic_matrix,
    scale_intrinsic,
)
from gaussian_avatar.data.people_snapshot import preprocess_image, preprocess_mask

def load_cameras_from_annots(
    annots_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
    """Extract per-camera calibration from annots.npy.

    Args:
        annots_path: Path to annots.npy file

    Returns:
        Tuple of (camera_names, cameras_dict) where cameras_dict maps
        camera name to {K, D, R, T} numpy arrays.
    """
    annots = np.load(str(annots_path), allow_pickle=True).item()
    cams = annots["cams"]

    # Derive camera names from image paths in the first frame
    first_frame_ims = annots["ims"][0]["ims"]
    cam_names = [Path(p).parent.name for p in first_frame_ims]

    cameras = {}
    for i, name in enumerate(cam_names):
        cameras[name] = {
            "K": np.array(cams["K"][i], dtype=np.float32),
            "D": np.array(cams["D"][i], dtype=np.float32).ravel(),
            "R": np.array(cams["R"][i], dtype=np.float32),
            "T": np.array(cams["T"][i], dtype=np.float32).ravel(),
        }

    return cam_names, cameras

class ZJUMoCapDataset(BaseDataset):
    """Multi-view dataset for ZJU MoCap sequences (MHR, SMPL, or SMPL-X).

    Each sample is a (camera, frame) pair. Training uses a single camera
    (monocular), while testing uses the remaining cameras (novel-view).

    Body model parameters are loaded per-camera from Camera_BX/results/.
    Images and masks are loaded from separate subdirectories.

    Args:
        data_root: Path to ZJU MoCap directory containing subject folders
        subjects: Subject names, single name, or "all"
        split: "train", "test", or "all"
        image_size: Target (H, W) for resizing
        train_cameras: Camera names for training (e.g. ["Camera_B1"]).
            Test split uses all cameras NOT in this list.
        stop_frame: Optional frame index to truncate all cameras at.
            Both train and test splits use frames [0, stop_frame).
            When None, all available frames are used.
        skip: Optional frame skip stride
        model_type: Body model type — "mhr", "smpl", or "smplx". Determines
            which fits are loaded and how translation/extrinsics are handled.
        preprocess_masks: Whether to apply erosion and Gaussian blur to masks
            before resizing (default True).

    Raises:
        FileNotFoundError: If data_root or annots.npy not found
        ValueError: If subject, camera, or model_type not found
    """

    _VALID_MODEL_TYPES = ("mhr", "smpl", "smplx")

    def __init__(
        self,
        data_root: Union[str, Path],
        subjects: Union[List[str], str] = "all",
        split: str = "train",
        image_size: Tuple[int, int] = (512, 512),
        train_cameras: Optional[List[str]] = None,
        stop_frame: Optional[int] = None,
        skip: Optional[int] = None,
        model_type: str = "mhr",
        preprocess_masks: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        if model_type not in self._VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {self._VALID_MODEL_TYPES}, "
                f"got '{model_type}'"
            )

        self.image_size = image_size
        self.split = split
        self.train_cameras = train_cameras or ["Camera_B1"]
        self.stop_frame = stop_frame
        self.skip = skip
        self.model_type = model_type
        self.preprocess_masks = preprocess_masks

        if skip is not None and skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")

        self._subjects = self._parse_subjects(subjects)

        self._subject_data: Dict[str, SubjectData] = {}
        # Per-camera body model data:
        # MHR: {subject: {cam: {model_params, trans, focal_lengths}}}
        # SMPL/SMPL-X: {subject: {cam: {poses, trans, betas, focal_lengths}}}
        self._cam_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self._identity_coeffs: Dict[str, np.ndarray] = {}
        self._cameras: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self._all_cam_names: Dict[str, List[str]] = {}
        self._num_frames: Dict[str, int] = {}
        # Per-camera available image frame indices (handles gaps in image sequences)
        self._available_frames: Dict[str, Dict[str, set]] = {}

        for subject in self._subjects:
            self._load_subject(subject)

        self._build_index_mapping()

    @property
    def subjects(self) -> List[str]:
        """Return list of loaded subject names."""
        return self._subjects

    def _parse_subjects(self, subjects: Union[List[str], str]) -> List[str]:
        """Parse subject parameter into list of subject names."""
        if subjects == "all":
            return sorted([
                d.name for d in self.data_root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])
        if isinstance(subjects, str):
            subjects = [subjects]
        for s in subjects:
            if not (self.data_root / s).exists():
                raise ValueError(f"Subject not found: {s}")
        return list(subjects)

    def _cameras_for_split(self, subject: str) -> List[str]:
        """Return camera names for the current split."""
        all_cams = self._all_cam_names[subject]
        if self.split == "train":
            return [c for c in all_cams if c in self.train_cameras]
        elif self.split == "test":
            return [c for c in all_cams if c not in self.train_cameras]
        else:
            return all_cams

    def _load_subject(self, subject: str) -> None:
        """Load camera calibration and body model parameters for a subject."""
        subject_dir = self.data_root / subject

        annots_path = subject_dir / "annots.npy"
        if not annots_path.exists():
            raise FileNotFoundError(f"annots.npy not found in {subject_dir}")
        cam_names, cam_calibration = load_cameras_from_annots(annots_path)

        available_cams = []
        for d in sorted(subject_dir.iterdir()):
            if not (d.is_dir() and d.name.startswith("Camera_B")):
                continue
            cam_name = d.name
            if cam_name not in cam_calibration:
                continue
            if self._has_fits(d):
                available_cams.append(cam_name)

        self._all_cam_names[subject] = available_cams
        self._cameras[subject] = {n: cam_calibration[n] for n in available_cams}

        # Scan available image frame indices per camera (handles gaps)
        self._available_frames[subject] = {}
        for cam_name in available_cams:
            img_dir = subject_dir / cam_name / "images"
            if img_dir.is_dir():
                stems = set()
                for p in img_dir.iterdir():
                    if p.suffix in (".jpg", ".png"):
                        try:
                            stems.add(int(p.stem))
                        except ValueError:
                            pass
                self._available_frames[subject][cam_name] = stems

        # Load body model params per camera (only for cameras in the current split)
        split_cams = self._cameras_for_split(subject)
        self._cam_data[subject] = {}
        num_frames = None

        for cam_name in split_cams:
            if self.model_type in ("smpl", "smplx"):
                cam_data = self._load_camera_smpl_family(subject_dir, cam_name)
                n = len(cam_data["poses"])
            else:
                cam_data = self._load_camera_mhr(subject_dir, cam_name)
                n = len(cam_data["model_params"])
            self._cam_data[subject][cam_name] = cam_data
            if num_frames is None:
                num_frames = n
            else:
                num_frames = min(num_frames, n)

        if num_frames is None:
            raise ValueError(f"No cameras found for split '{self.split}' in {subject}")

        if self.stop_frame is not None:
            if self.stop_frame > num_frames:
                raise ValueError(
                    f"stop_frame {self.stop_frame} exceeds available frames "
                    f"({num_frames}) for {subject}"
                )
            num_frames = self.stop_frame

        self._num_frames[subject] = num_frames

        if self.model_type in ("smpl", "smplx"):
            # Betas from SMPL/SMPL-X fits (constant across frames, first row)
            ref_cam_data = next(iter(self._cam_data[subject].values()))
            self._identity_coeffs[subject] = ref_cam_data["betas"]
        else:
            # MHR shape coefficients from shape.npy
            first_cam = split_cams[0]
            shape_path = subject_dir / first_cam / "results" / "shape.npy"
            self._identity_coeffs[subject] = np.load(
                str(shape_path)
            ).astype(np.float32)

        first_cam = split_cams[0]
        first_img_dir = subject_dir / first_cam / "images"
        first_img_path = sorted(first_img_dir.glob("*.*"))[0]
        first_img = cv2.imread(str(first_img_path), cv2.IMREAD_UNCHANGED)
        H, W = first_img.shape[:2]

        # Both modes use SAM3D per-frame focal length for intrinsics.
        # Build a reference intrinsic from the median focal across frames.
        ref_cam_data = next(iter(self._cam_data[subject].values()))
        median_focal = float(np.median(ref_cam_data["focal_lengths"]))
        intrinsic = build_intrinsic_matrix(
            np.array([median_focal, median_focal]),
            np.array([W / 2.0, H / 2.0]),
        )

        if self.model_type in ("smpl", "smplx"):
            # SMPL/SMPL-X fits already have global_orient rotating body into
            # camera Y-down space, and transl positioning in camera space
            extrinsic = np.eye(4, dtype=np.float32)
        else:
            # MHR mesh operates in Y-up/cm; Y/Z-flip converts to camera Y-down
            extrinsic = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

        # Both train and test use the same frame range — the split is
        # purely by camera (train_cameras vs the rest).
        all_indices = np.arange(num_frames)

        self._subject_data[subject] = SubjectData(
            name=subject,
            betas=self._identity_coeffs[subject],
            v_personal=np.zeros((0, 3), dtype=np.float32),
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            distortion=np.zeros(5, dtype=np.float32),
            image_size=(H, W),
            num_frames=num_frames,
            train_indices=all_indices,
            test_indices=all_indices,
        )

    def _has_fits(self, cam_dir: Path) -> bool:
        """Check whether a camera directory has the required body model fits.

        Args:
            cam_dir: Path to a Camera_BX directory

        Returns:
            True if the required fits exist for the current model_type.
        """
        if self.model_type in ("smpl", "smplx"):
            subdir, filename = self._smpl_family_paths()
            params_path = cam_dir / "results" / subdir / filename
            # Also need raw/ for focal lengths
            raw_dir = cam_dir / "results" / "raw"
            return (
                params_path.is_file()
                and raw_dir.is_dir()
                and any(raw_dir.glob("*.npz"))
            )
        else:
            raw_dir = cam_dir / "results" / "raw"
            return raw_dir.is_dir() and any(raw_dir.glob("*.npz"))

    def _smpl_family_paths(self) -> Tuple[str, str]:
        """Return (subdir, filename) for SMPL or SMPL-X fits.

        Returns:
            Tuple of (subdirectory name, npz filename) under results/.
        """
        if self.model_type == "smpl":
            return ("smpl", "smpl_params.npz")
        return ("smplx", "smplx_params.npz")

    @staticmethod
    def _load_focal_lengths_from_raw(
        raw_dir: Path,
    ) -> np.ndarray:
        """Load per-frame SAM3D focal lengths from results/raw/.

        All modes need these because the SMPL/SMPL-X fittings were done
        against vertices positioned using SAM3D's camera.

        Args:
            raw_dir: Path to Camera_BX/results/raw/ directory

        Returns:
            (N,) float32 array of focal lengths, NaN-filled with median.
        """
        npz_files = sorted(raw_dir.glob("*.npz"))
        focal_list = []
        for npz_path in npz_files:
            data = np.load(str(npz_path), allow_pickle=True)
            if "focal_length" in data:
                focal_list.append(float(data["focal_length"]))
            else:
                focal_list.append(np.nan)

        focal_arr = np.array(focal_list, dtype=np.float32)
        valid_mask = ~np.isnan(focal_arr)
        if valid_mask.any():
            focal_arr[~valid_mask] = float(np.median(focal_arr[valid_mask]))
        return focal_arr

    def _load_camera_smpl_family(
        self, subject_dir: Path, cam_name: str,
    ) -> Dict[str, np.ndarray]:
        """Load SMPL or SMPL-X parameters for a single camera.

        Reads fitted params from ``results/<subdir>/<filename>`` (determined
        by ``_smpl_family_paths``) and per-frame focal lengths from
        ``results/raw/*.npz``.

        The NPZ file must contain ``global_orient``, ``body_pose``,
        ``betas``, and ``transl``. Pose dimensions are handled
        automatically (72 for SMPL, 66 for SMPL-X).

        Args:
            subject_dir: Path to subject directory
            cam_name: Camera name (e.g. "Camera_B1")

        Returns:
            Dict with poses (N, pose_dim), trans (N, 3), betas (10,),
            focal_lengths (N,).
        """
        subdir, filename = self._smpl_family_paths()
        params_path = (
            subject_dir / cam_name / "results" / subdir / filename
        )
        data = np.load(str(params_path), allow_pickle=True)

        # Concatenate global_orient (T,3) + body_pose (T,69 or T,63)
        global_orient = np.array(data["global_orient"], dtype=np.float32)
        body_pose = np.array(data["body_pose"], dtype=np.float32)
        poses = np.concatenate([global_orient, body_pose], axis=1)

        # Translation: already in meters, camera Y-down — use directly
        trans = np.array(data["transl"], dtype=np.float32)

        # Betas: constant across frames, take first row
        betas = np.array(data["betas"][0], dtype=np.float32)

        # Focal lengths from SAM3D raw fits
        raw_dir = subject_dir / cam_name / "results" / "raw"
        focal_lengths = self._load_focal_lengths_from_raw(raw_dir)

        if len(poses) != len(focal_lengths):
            raise ValueError(
                f"Frame count mismatch for {cam_name}: "
                f"{subdir} params has {len(poses)} frames, "
                f"raw/ has {len(focal_lengths)} .npz files"
            )

        return {
            "poses": poses,
            "trans": trans,
            "betas": betas,
            "focal_lengths": focal_lengths,
        }

    def _load_camera_mhr(
        self, subject_dir: Path, cam_name: str,
    ) -> Dict[str, np.ndarray]:
        """Load MHR parameters for a single camera.

        Args:
            subject_dir: Path to subject directory
            cam_name: Camera name (e.g. "Camera_B1")

        Returns:
            Dict with model_params (N, 204), trans (N, 3), focal_lengths (N,).
        """
        raw_dir = subject_dir / cam_name / "results" / "raw"
        npz_files = sorted(raw_dir.glob("*.npz"))

        model_params_list = []
        trans_list = []

        for npz_path in npz_files:
            data = np.load(str(npz_path), allow_pickle=True)
            model_params_list.append(
                np.array(data["mhr_model_params"], dtype=np.float32)
            )
            # Convert pred_cam_t from camera space (m, Y-down) to mesh space
            # (cm, Y-up): multiply by [100, -100, -100]
            cam_t = np.array(data["pred_cam_t"], dtype=np.float32)
            trans_list.append(cam_t * np.array([100.0, -100.0, -100.0], dtype=np.float32))

        focal_lengths = self._load_focal_lengths_from_raw(raw_dir)

        return {
            "model_params": np.stack(model_params_list),
            "trans": np.stack(trans_list),
            "focal_lengths": focal_lengths,
        }

    def _build_index_mapping(self) -> None:
        """Build global index mapping to (subject, camera, frame_idx)."""
        self._index_mapping: List[Tuple[str, str, int]] = []

        for subject in self._subjects:
            subject_data = self._subject_data[subject]
            split_cams = self._cameras_for_split(subject)

            if self.split == "train":
                indices = subject_data.train_indices
            elif self.split == "test":
                indices = subject_data.test_indices
            else:
                indices = np.arange(subject_data.num_frames)

            if self.skip is not None and self.skip > 1:
                indices = indices[::self.skip]

            for frame_idx in indices:
                for cam_name in split_cams:
                    # Skip frames missing from this camera's image directory
                    avail = self._available_frames.get(subject, {}).get(cam_name)
                    if avail is not None and int(frame_idx) not in avail:
                        continue
                    self._index_mapping.append(
                        (subject, cam_name, int(frame_idx))
                    )

    def __len__(self) -> int:
        return len(self._index_mapping)

    def get_sample_info(self, idx: int) -> Tuple[str, str, int]:
        """Get (subject, camera_name, frame_idx) for a global index."""
        if idx < 0 or idx >= len(self._index_mapping):
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {len(self)} samples"
            )
        return self._index_mapping[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample dictionary for the given index.

        Returns:
            Dictionary with image, mask, pose, betas, trans, intrinsic,
            extrinsic, frame_idx, subject keys.
        """
        subject, cam_name, frame_idx = self.get_sample_info(idx)
        subject_data = self._subject_data[subject]
        subject_dir = self.data_root / subject

        cam_dir = subject_dir / cam_name
        frame_stem = f"{frame_idx:06d}"

        # Image: try .jpg first, fall back to .png
        img_path = cam_dir / "images" / f"{frame_stem}.jpg"
        if not img_path.exists():
            img_path = cam_dir / "images" / f"{frame_stem}.png"
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Mask: grayscale PNG, threshold to binary
        mask_path = cam_dir / "masks" / f"{frame_stem}.png"
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        mask = (mask_raw > 0).astype(np.int8)

        image_tensor = preprocess_image(image, self.image_size)
        mask_tensor = preprocess_mask(mask, self.image_size, smooth=self.preprocess_masks)

        # Camera: SAM3D per-camera focal length + centered principal point.
        # Both MHR and SMPL-X use the same per-frame focal from SAM3D raw fits.
        cam_data = self._cam_data[subject][cam_name]
        focal = cam_data["focal_lengths"][frame_idx]
        H_orig, W_orig = subject_data.image_size
        intrinsic = build_intrinsic_matrix(
            np.array([focal, focal]),
            np.array([W_orig / 2.0, H_orig / 2.0]),
        )
        scaled_intrinsic = scale_intrinsic(
            intrinsic, subject_data.image_size, self.image_size
        )

        # Body model parameters and extrinsic depend on model_type
        if self.model_type in ("smpl", "smplx"):
            pose = torch.from_numpy(cam_data["poses"][frame_idx])
            trans = torch.from_numpy(cam_data["trans"][frame_idx])
            extrinsic = np.eye(4, dtype=np.float32)
        else:
            pose = torch.from_numpy(cam_data["model_params"][frame_idx])
            trans = torch.from_numpy(cam_data["trans"][frame_idx])
            extrinsic = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

        betas = torch.from_numpy(self._identity_coeffs[subject])

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "pose": pose,
            "betas": betas,
            "trans": trans,
            "intrinsic": torch.from_numpy(scaled_intrinsic),
            "extrinsic": torch.from_numpy(extrinsic),
            "frame_idx": frame_idx,
            "subject": subject,
        }

    def get_subject_data(self, subject: str) -> SubjectData:
        """Get static data for a subject."""
        if subject not in self._subject_data:
            raise KeyError(f"Subject not loaded: {subject}")
        return self._subject_data[subject]

def create_zju_mocap_dataloaders(
    data_root: str,
    subjects: Union[List[str], str],
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 1,
    num_workers: int = 4,
    train_cameras: Optional[List[str]] = None,
    stop_frame: Optional[int] = None,
    skip: Optional[int] = None,
    model_type: str = "mhr",
    preprocess_masks: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and test DataLoaders for ZJU MoCap dataset.

    Args:
        data_root: Path to ZJU MoCap dataset root
        subjects: Subject names, single name, or "all"
        image_size: Target (H, W) for resizing
        batch_size: Batch size for both loaders
        num_workers: Number of DataLoader workers
        train_cameras: Cameras for training (default ["Camera_B1"]).
            Test loader uses all other cameras.
        stop_frame: Optional frame truncation point. Both train and test
            use frames [0, stop_frame). When None, all frames are used.
        skip: Optional frame skip stride
        model_type: Body model type — "mhr", "smpl", or "smplx"
        preprocess_masks: Whether to apply erosion and blur to masks

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from gaussian_avatar.data.people_snapshot import custom_collate

    train_dataset = ZJUMoCapDataset(
        data_root=data_root,
        subjects=subjects,
        split="train",
        image_size=image_size,
        train_cameras=train_cameras,
        stop_frame=stop_frame,
        skip=skip,
        model_type=model_type,
        preprocess_masks=preprocess_masks,
    )
    test_dataset = ZJUMoCapDataset(
        data_root=data_root,
        subjects=subjects,
        split="test",
        image_size=image_size,
        train_cameras=train_cameras,
        stop_frame=stop_frame,
        skip=skip,
        model_type=model_type,
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
