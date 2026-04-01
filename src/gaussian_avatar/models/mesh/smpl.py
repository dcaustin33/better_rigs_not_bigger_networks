"""SMPL/SMPL-X mesh implementation using the smplx library."""

import smplx
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
from torch import Tensor

from gaussian_avatar.constants import (
    A_POSE_ANGLE,
    MODEL_SPECS,
    SMPL_LEFT_SHOULDER_IDX,
    SMPL_RIGHT_SHOULDER_IDX,
    SMPLX_LEFT_SHOULDER_IDX,
    SMPLX_RIGHT_SHOULDER_IDX,
)
from gaussian_avatar.models.mesh import utils
from gaussian_avatar.models.mesh.base import BaseMesh


class SMPLMesh(BaseMesh):
    """SMPL/SMPL-X mesh implementation wrapping the smplx library.

    This class provides access to SMPL and SMPL-X body models for use
    with the Gaussian avatar system. The canonical pose is A-pose (arms
    at ~45 degrees from body) rather than T-pose.

    Args:
        model_path: Path to the directory containing model files
        model_type: Type of model ('smpl' or 'smplx')
        gender: Model gender ("male", "female", or "neutral")
        num_betas: Number of shape coefficients to use
        device: Device to place tensors on
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "smplx",
        gender: str = "neutral",
        num_betas: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        if model_type not in MODEL_SPECS:
            raise ValueError(f"model_type must be 'smpl' or 'smplx', got '{model_type}'")

        self._model_type = model_type
        self._device = device
        self._num_betas = num_betas

        self._model = smplx.create(
            model_path=model_path,
            model_type=model_type,
            gender=gender,
            num_betas=num_betas,
            use_pca=False,
        ).to(device)

        specs = MODEL_SPECS[model_type]
        self._num_vertices = specs["vertices"]
        self._num_joints = specs["joints"]
        self._canonical_vertices = self._compute_canonical_vertices()

        self._faces = self._model.faces_tensor.to(device).contiguous()
        self._lbs_weights = self._model.lbs_weights.to(device)
        self._canonical_triangle_frames: Tensor | None = None
        self.enable_grad: bool = False

    def _compute_canonical_vertices(self) -> Tensor:
        """Compute canonical vertices in A-pose.

        A-pose (arms at ~45 degrees) reduces armpit artifacts compared to T-pose.
        In SMPL coordinate system, negative Z rotation on left shoulder and
        positive Z rotation on right shoulder moves arms down from T-pose.

        Returns:
            Tensor of shape (V, 3) with canonical vertices in A-pose
        """
        if self._model_type == "smpl":
            pose = torch.zeros(1, 72, device=self._device)
            # SMPL_*_SHOULDER_IDX are overall joint indices; in the 72-dim
            # full pose vector, joint N starts at N*3 (global orient is joint 0)
            left_shoulder_start = SMPL_LEFT_SHOULDER_IDX * 3
            right_shoulder_start = SMPL_RIGHT_SHOULDER_IDX * 3
            pose[0, left_shoulder_start + 2] = -A_POSE_ANGLE
            pose[0, right_shoulder_start + 2] = A_POSE_ANGLE
        else:
            # SMPL-X body_pose excludes global orient, indices are direct
            pose = torch.zeros(1, 63, device=self._device)
            pose[0, SMPLX_LEFT_SHOULDER_IDX * 3 + 2] = -A_POSE_ANGLE
            pose[0, SMPLX_RIGHT_SHOULDER_IDX * 3 + 2] = A_POSE_ANGLE

        betas = torch.zeros(1, self._num_betas, device=self._device)

        with torch.no_grad():
            if self._model_type == "smpl":
                output = self._model(
                    betas=betas,
                    body_pose=pose[:, 3:],
                    global_orient=pose[:, :3],
                )
            else:
                output = self._model(
                    betas=betas,
                    body_pose=pose,
                    global_orient=torch.zeros(1, 3, device=self._device),
                )

        return output.vertices.squeeze(0)

    @property
    def model_type(self) -> str:
        """Return the model type identifier ('smpl' or 'smplx')."""
        return self._model_type

    @property
    def num_joints(self) -> int:
        """Return the number of joints in the skeleton."""
        return self._num_joints

    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the mesh."""
        return self._num_vertices

    @property
    def device(self) -> torch.device:
        """Return the device where mesh tensors are stored."""
        return self._device

    def get_canonical_vertices(self) -> Tensor:
        """Return vertices in canonical (A-pose) space.

        Returns:
            Tensor of shape (V, 3) where V is number of vertices
        """
        return self._canonical_vertices

    def get_faces(self) -> Tensor:
        """Return triangle connectivity.

        Returns cached, contiguous tensor for efficient memory access patterns.

        Returns:
            Tensor of shape (F, 3) with vertex indices per face
        """
        return self._faces

    def get_lbs_weights(self) -> Tensor:
        """Return skinning weights.

        Returns cached tensor on the mesh's device.

        Returns:
            Tensor of shape (V, J) where J is number of joints
        """
        return self._lbs_weights

    def get_canonical_triangle_frames(self) -> Tensor:
        """Return cached local coordinate frames for canonical mesh triangles.

        Computes frames once on first call and caches for subsequent calls.
        This avoids redundant computation when frames are needed multiple times.

        Returns:
            Tensor of shape (F, 3, 3) with orthonormal rotation matrices
        """
        if self._canonical_triangle_frames is None:
            can_verts = self.get_canonical_vertices()
            self._canonical_triangle_frames = utils.get_triangle_local_frames(
                can_verts, self._faces
            )
        return self._canonical_triangle_frames

    def _ensure_batched(
        self, pose: Tensor, shape: Tensor
    ) -> tuple[Tensor, Tensor, int, bool]:
        """Ensure pose and shape tensors have batch dimension and are on device.

        Args:
            pose: Pose parameters, shape (pose_dims,) or (B, pose_dims)
            shape: Shape coefficients, shape (num_betas,) or (B, num_betas)

        Returns:
            Tuple of (batched_pose, batched_shape, batch_size, was_batched)
        """
        is_batched = pose.dim() == 2
        if not is_batched:
            pose = pose.unsqueeze(0)
            shape = shape.unsqueeze(0)
        batch_size = pose.shape[0]
        pose = pose.to(self._device)
        shape = shape.to(self._device)
        return pose, shape, batch_size, is_batched

    def prepare_pose_params(self, pose: Tensor) -> dict:
        """Convert unified pose tensor to model-specific keyword arguments.

        Args:
            pose: Unified pose tensor of shape (pose_dims,) or (B, pose_dims)
                  SMPL: 72 dims (3 global + 69 body), SMPL-X: 66 dims (3 global + 63 body)

        Returns:
            Dictionary with global_orient and body_pose tensors
        """
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        global_orient = pose[:, :3]
        body_pose = pose[:, 3:72] if self._model_type == "smpl" else pose[:, 3:66]

        return {"global_orient": global_orient, "body_pose": body_pose}

    def get_joint_transforms(self, pose: Tensor, shape: Tensor) -> Tensor:
        """Compute per-joint transformation matrices.

        Args:
            pose: Pose parameters, shape (pose_dims,) or (B, pose_dims)
            shape: Shape coefficients, shape (num_betas,) or (B, num_betas)

        Returns:
            Tensor of shape (J, 4, 4) or (B, J, 4, 4) with 4x4 transform per joint
        """
        pose, shape, batch_size, is_batched = self._ensure_batched(pose, shape)

        pose_params = self.prepare_pose_params(pose)
        global_orient = pose_params["global_orient"]
        body_pose = pose_params["body_pose"]

        with torch.set_grad_enabled(self.enable_grad):
            v_shaped = self._model.v_template + blend_shapes(shape, self._model.shapedirs)
            J = vertices2joints(self._model.J_regressor, v_shaped)

            if self._model_type == "smpl":
                full_pose = torch.cat([global_orient, body_pose], dim=1)
            else:
                # SMPL-X requires padding for jaw, eyes, hands (99 dims)
                zeros_padding = torch.zeros(batch_size, 99, device=self._device, dtype=pose.dtype)
                full_pose = torch.cat([global_orient, body_pose, zeros_padding], dim=1)

            rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch_size, self._num_joints, 3, 3)
            _, A = batch_rigid_transform(rot_mats, J, self._model.parents)

        if not is_batched:
            A = A.squeeze(0)

        return A

    def _get_smplx_zero_params(self, batch_size: int, dtype: torch.dtype) -> dict:
        """Create zero tensors for SMPL-X optional parameters.

        Args:
            batch_size: Batch dimension size
            dtype: Tensor data type

        Returns:
            Dictionary with zero tensors for jaw, eyes, hands, and expression
        """
        zeros = lambda dim: torch.zeros(batch_size, dim, device=self._device, dtype=dtype)
        return {
            "jaw_pose": zeros(3),
            "leye_pose": zeros(3),
            "reye_pose": zeros(3),
            "left_hand_pose": zeros(45),
            "right_hand_pose": zeros(45),
            "expression": zeros(10),
        }

    def forward(self, pose: Tensor, shape: Tensor) -> Tensor:
        """Apply forward skinning to get posed vertices using smplx library.

        Args:
            pose: Pose parameters, shape (pose_dims,) or (B, pose_dims)
            shape: Shape coefficients, shape (num_betas,) or (B, num_betas)

        Returns:
            Tensor of shape (V, 3) or (B, V, 3) with posed vertices
        """
        pose, shape, batch_size, is_batched = self._ensure_batched(pose, shape)
        pose_params = self.prepare_pose_params(pose)

        model_kwargs = {
            "betas": shape,
            "global_orient": pose_params["global_orient"],
            "body_pose": pose_params["body_pose"],
        }
        if self._model_type == "smplx":
            model_kwargs.update(self._get_smplx_zero_params(batch_size, pose.dtype))

        with torch.set_grad_enabled(self.enable_grad):
            output = self._model(**model_kwargs)

        vertices = output.vertices
        if not is_batched:
            vertices = vertices.squeeze(0)

        return vertices
