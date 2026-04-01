"""Body model constants for SMPL, SMPL-X, and MHR.

This module contains all constants related to body models,
including model dimensions, joint indices, and canonical pose parameters.
Both data loading and mesh modules should import from here.
"""

import math

# =============================================================================
# =============================================================================

SMPL_VERTEX_COUNT = 6890
SMPL_FACE_COUNT = 13776
SMPL_POSE_DIM = 72  # 3 global orientation + 23 joints * 3 axis-angle
SMPL_BODY_POSE_DIM = 69  # 23 joints * 3 axis-angle (excludes global orient)
SMPL_BETAS_DIM = 10
SMPL_JOINTS = 24  # 23 body joints + 1 root

# SMPL joint indices (in full pose, after global orient offset)
SMPL_LEFT_SHOULDER_IDX = 16
SMPL_RIGHT_SHOULDER_IDX = 17

# =============================================================================
# =============================================================================

SMPLX_VERTEX_COUNT = 10475
SMPLX_FACE_COUNT = 20908
SMPLX_POSE_DIM = 66  # 3 global orientation + 21 body joints * 3 axis-angle
SMPLX_BODY_POSE_DIM = 63  # 21 body joints * 3 axis-angle (excludes global orient)
SMPLX_JOINTS = 55

# SMPL-X body_pose joint indices (excludes global orient, so offset by 1 from SMPL)
SMPLX_LEFT_SHOULDER_IDX = 15
SMPLX_RIGHT_SHOULDER_IDX = 16

# =============================================================================
# =============================================================================

MHR_JOINTS = 127
MHR_POSE_DIM = 204  # 136 pose + 68 skeleton (PCA-like parameterization)
MHR_SHAPE_DIM = 45  # 20 body + 20 head + 5 hands
MHR_EXPRESSION_DIM = 72  # FACS blendshapes

# LOD-dependent vertex counts (from MHR documentation)
MHR_LOD_SPECS = {
    0: {"vertices": 73639},  # High-fidelity / film
    1: {"vertices": 18439},  # Training (recommended)
    2: {"vertices": 10661},  # Mid-quality experiments
    3: {"vertices": 4899},   # Real-time / fast iteration
    4: {"vertices": 2461},   # Real-time mobile
    5: {"vertices": 971},    # Low-poly
    6: {"vertices": 595},    # Minimal
}

# =============================================================================
# =============================================================================

# A-pose angle: 45 degrees from T-pose
# Arms down at ~45 degrees reduces armpit artifacts
A_POSE_ANGLE = math.pi / 4

# =============================================================================
# =============================================================================

MODEL_SPECS = {
    "smpl": {
        "vertices": SMPL_VERTEX_COUNT,
        "joints": SMPL_JOINTS,
        "num_joints": SMPL_JOINTS,
        "faces": SMPL_FACE_COUNT,
        "pose_dim": SMPL_POSE_DIM,
        "body_pose_dim": SMPL_BODY_POSE_DIM,
        "pose_representation": "axis_angle",
    },
    "smplx": {
        "vertices": SMPLX_VERTEX_COUNT,
        "joints": SMPLX_JOINTS,
        "num_joints": SMPLX_JOINTS,
        "faces": SMPLX_FACE_COUNT,
        "pose_dim": SMPLX_POSE_DIM,
        "body_pose_dim": SMPLX_BODY_POSE_DIM,
        "pose_representation": "axis_angle",
    },
    "mhr": {
        "joints": MHR_JOINTS,
        "num_joints": MHR_JOINTS,
        "pose_dim": MHR_POSE_DIM,
        "shape_dim": MHR_SHAPE_DIM,
        "expression_dim": MHR_EXPRESSION_DIM,
        "pose_representation": "euler_xyz",
        "lod_specs": MHR_LOD_SPECS,
    },
}
