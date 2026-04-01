"""Data loading module for Gaussian Avatar.

This module provides dataset classes and utilities for loading
PeopleSnapshot and other datasets for training Gaussian avatars.
"""

from gaussian_avatar.data.base import BaseDataset, SubjectData, compute_split_indices
from gaussian_avatar.data.constants import (
    A_POSE_ANGLE,
    MODEL_SPECS,
    SMPL_BETAS_DIM,
    SMPL_BODY_POSE_DIM,
    SMPL_FACE_COUNT,
    SMPL_JOINTS,
    SMPL_LEFT_SHOULDER_IDX,
    SMPL_POSE_DIM,
    SMPL_RIGHT_SHOULDER_IDX,
    SMPL_VERTEX_COUNT,
    SMPLX_BODY_POSE_DIM,
    SMPLX_FACE_COUNT,
    SMPLX_JOINTS,
    SMPLX_LEFT_SHOULDER_IDX,
    SMPLX_POSE_DIM,
    SMPLX_RIGHT_SHOULDER_IDX,
    SMPLX_VERTEX_COUNT,
)
from gaussian_avatar.data.camera import (
    build_extrinsic_from_matrix,
    build_extrinsic_matrix,
    build_intrinsic_matrix,
    load_camera,
    load_camera_corrected,
    scale_intrinsic,
    undistort_image,
)
from gaussian_avatar.data.smpl_loaders import (
    BaseHDF5Reader,
    KeypointsReader,
    MaskReader,
    get_valid_frame_indices,
    load_consensus,
    load_corrected_smpl_frame,
    load_poses,
    pose_to_mask_index,
)
from gaussian_avatar.data.video import (
    FrameCache,
    cache_all_frames,
    extract_frame,
    get_video_frame_count,
)
from gaussian_avatar.data.people_snapshot import (
    PeopleSnapshotDataset,
    create_dataloaders,
    custom_collate,
    preprocess_image,
    preprocess_mask,
)
from gaussian_avatar.data.people_snapshot_corrected import (
    PeopleSnapshotCorrectedDataset,
)
from gaussian_avatar.data.mhr_loaders import load_mhr_params
from gaussian_avatar.data.mhr_dataset import MHRDataset, create_mhr_dataloaders
from gaussian_avatar.data.zju_mocap import ZJUMoCapDataset, create_zju_mocap_dataloaders
from gaussian_avatar.data.visualize import (
    project_vertices,
    visualize_sample,
    visualize_sample_by_index,
    visualize_sample_from_data,
)

__all__ = [
    "BaseDataset",
    "BaseHDF5Reader",
    "SubjectData",
    "A_POSE_ANGLE",
    "MODEL_SPECS",
    "SMPL_BETAS_DIM",
    "SMPL_BODY_POSE_DIM",
    "SMPL_FACE_COUNT",
    "SMPL_JOINTS",
    "SMPL_LEFT_SHOULDER_IDX",
    "SMPL_POSE_DIM",
    "SMPL_RIGHT_SHOULDER_IDX",
    "SMPL_VERTEX_COUNT",
    "SMPLX_BODY_POSE_DIM",
    "SMPLX_FACE_COUNT",
    "SMPLX_JOINTS",
    "SMPLX_LEFT_SHOULDER_IDX",
    "SMPLX_POSE_DIM",
    "SMPLX_RIGHT_SHOULDER_IDX",
    "SMPLX_VERTEX_COUNT",
    "FrameCache",
    "KeypointsReader",
    "MaskReader",
    "MHRDataset",
    "PeopleSnapshotCorrectedDataset",
    "PeopleSnapshotDataset",
    "ZJUMoCapDataset",
    "compute_split_indices",
    "create_dataloaders",
    "create_mhr_dataloaders",
    "create_zju_mocap_dataloaders",
    "custom_collate",
    "build_extrinsic_from_matrix",
    "build_extrinsic_matrix",
    "build_intrinsic_matrix",
    "load_camera",
    "load_camera_corrected",
    "scale_intrinsic",
    "undistort_image",
    "cache_all_frames",
    "extract_frame",
    "get_video_frame_count",
    "get_valid_frame_indices",
    "load_consensus",
    "load_corrected_smpl_frame",
    "load_poses",
    "pose_to_mask_index",
    "load_mhr_params",
    "preprocess_image",
    "preprocess_mask",
    "project_vertices",
    "visualize_sample",
    "visualize_sample_by_index",
    "visualize_sample_from_data",
]
