"""Camera loading and matrix construction utilities.

This module provides functions for loading camera calibration data from
PeopleSnapshot dataset files and constructing camera matrices for rendering.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

def load_camera(camera_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load camera calibration data from a pickle file.

    Parses camera.pkl files from the PeopleSnapshot dataset, which contain
    camera intrinsics, distortion coefficients, and extrinsic parameters.

    Args:
        camera_path: Path to camera.pkl file

    Returns:
        Dictionary containing:
            - "focal": (2,) array of focal lengths (fx, fy) in pixels
            - "principal": (2,) array of principal point (cx, cy) in pixels
            - "distortion": (5,) array of distortion coefficients (k1, k2, p1, p2, k3)
            - "rotation": (3,) array of axis-angle rotation
            - "translation": (3,) array of translation
            - "height": int image height
            - "width": int image width

    Raises:
        FileNotFoundError: If camera_path does not exist
    """
    camera_path = Path(camera_path)
    if not camera_path.exists():
        raise FileNotFoundError(f"Camera file not found: {camera_path}")

    with open(camera_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return {
        "focal": data["camera_f"],
        "principal": data["camera_c"],
        "distortion": data["camera_k"],
        "rotation": data["camera_rt"],
        "translation": data["camera_t"],
        "height": data["height"],
        "width": data["width"],
    }

def build_intrinsic_matrix(
    focal: np.ndarray,
    principal: np.ndarray,
) -> np.ndarray:
    """Construct 3x3 camera intrinsic matrix from focal length and principal point.

    Builds the standard camera intrinsic matrix K:
        K = [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,  1]]

    Args:
        focal: (2,) array containing focal lengths (fx, fy) in pixels
        principal: (2,) array containing principal point (cx, cy) in pixels

    Returns:
        (3, 3) float32 intrinsic matrix K
    """
    fx, fy = focal
    cx, cy = principal

    K = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=np.float32,
    )

    return K

def build_extrinsic_matrix(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Construct 4x4 world-to-camera transform from rotation and translation.

    Converts axis-angle rotation to rotation matrix using Rodrigues formula
    and combines with translation into a homogeneous transformation matrix.

    Args:
        rotation: (3,) axis-angle rotation vector
        translation: (3,) translation vector

    Returns:
        (4, 4) float32 homogeneous transformation matrix
    """
    R, _ = cv2.Rodrigues(rotation)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T

def scale_intrinsic(
    intrinsic: np.ndarray,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Scale intrinsic matrix for resized images.

    Adjusts focal lengths and principal point when images are resized.
    The scaling is proportional to the ratio of target to original dimensions.

    Args:
        intrinsic: (3, 3) original intrinsic matrix
        original_size: (height, width) of original image
        target_size: (height, width) of target image

    Returns:
        (3, 3) float32 scaled intrinsic matrix
    """
    h_orig, w_orig = original_size
    h_tgt, w_tgt = target_size

    scale_x = w_tgt / w_orig
    scale_y = h_tgt / h_orig

    K = intrinsic.copy().astype(np.float32)
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y

    return K

def load_camera_corrected(camera_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load camera calibration data from a corrected PeopleSnapshot pickle file.

    Parses camera.pkl files from the corrected PeopleSnapshot dataset, which
    contain a rotation matrix (not axis-angle) and standard camera parameters.

    Args:
        camera_path: Path to camera.pkl file

    Returns:
        Dictionary containing:
            - "focal": (2,) array of focal lengths (fx, fy) in pixels
            - "principal": (2,) array of principal point (cx, cy) in pixels
            - "distortion": (5,) array of distortion coefficients (k1, k2, p1, p2, k3)
            - "rotation_matrix": (3, 3) rotation matrix
            - "translation": (3,) array of translation
            - "height": int image height
            - "width": int image width

    Raises:
        FileNotFoundError: If camera_path does not exist
    """
    camera_path = Path(camera_path)
    if not camera_path.exists():
        raise FileNotFoundError(f"Camera file not found: {camera_path}")

    with open(camera_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    return {
        "focal": data["camera_f"],
        "principal": data["camera_c"],
        "distortion": data["camera_k"],
        "rotation_matrix": data["R"],
        "translation": data["t"],
        "height": data["height"],
        "width": data["width"],
    }

def build_extrinsic_from_matrix(
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Construct 4x4 world-to-camera transform from rotation matrix and translation.

    Unlike build_extrinsic_matrix which takes axis-angle rotation, this function
    takes a pre-computed 3x3 rotation matrix directly.

    Args:
        rotation_matrix: (3, 3) rotation matrix
        translation: (3,) translation vector

    Returns:
        (4, 4) float32 homogeneous transformation matrix
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation

    return T

def undistort_image(
    image: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Remove lens distortion from an image.

    Applies inverse distortion using OpenCV's undistort function.
    The distortion model follows OpenCV convention with 5 coefficients:
    (k1, k2, p1, p2, k3) where k1, k2, k3 are radial distortion
    and p1, p2 are tangential distortion coefficients.

    Binary masks ({0, 1} int8) are scaled to {0, 255} before undistortion
    so that bilinear interpolation produces meaningful anti-aliased edge
    values. The result is returned as uint8 in [0, 255] range.

    Args:
        image: Input image array (H, W) or (H, W, C)
        intrinsic: (3, 3) camera intrinsic matrix
        distortion: (5,) distortion coefficients

    Returns:
        Undistorted image with same shape as input. For binary masks (int8
        with values in {0, 1}), returns uint8 in [0, 255] range.
    """
    # Detect binary masks: int8 with values in {0, 1}
    is_binary_mask = image.dtype == np.int8 and image.max() <= 1

    if is_binary_mask:
        # Scale to {0, 255} so bilinear interpolation produces smooth edges
        image = image.astype(np.uint8) * 255
    elif image.dtype == np.int8:
        # cv2.undistort doesn't support int8
        image = image.astype(np.uint8)

    result = cv2.undistort(image, intrinsic, distortion)

    # For binary masks, return uint8 with smooth [0, 255] values
    # (caller can threshold or use directly as soft mask)
    if not is_binary_mask and result.dtype != image.dtype:
        result = result.astype(image.dtype)

    return result
