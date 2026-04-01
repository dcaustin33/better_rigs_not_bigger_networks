"""MHR (Momentum Human Rig) mesh implementation using TorchScript model.

This module wraps Meta's MHR TorchScript model to provide the BaseMesh
interface for the Gaussian avatar pipeline. It uses the pre-exported
TorchScript model instead of the Python MHR/pymomentum API to avoid
memory issues with pymomentum.geometry imports.

The TorchScript model provides:
- Mesh geometry (rest vertices, faces)
- Blend shapes for identity/expression
- Parameter transform (model params -> joint params)
- Forward kinematics and skinning with pose correctives
- LBS weights in sparse format
"""

from pathlib import Path

import torch
from torch import Tensor

from gaussian_avatar.constants import MHR_EXPRESSION_DIM, MHR_JOINTS, MHR_LOD_SPECS, MHR_POSE_DIM, MHR_SHAPE_DIM
from gaussian_avatar.models.deformation.encoding import euler_xyz_to_6d
from gaussian_avatar.models.mesh import utils
from gaussian_avatar.models.mesh.base import BaseMesh


class MHRMesh(BaseMesh):
    """MHR mesh implementation wrapping a TorchScript model.

    Uses the pre-exported TorchScript model (mhr_model.pt) for mesh
    geometry, forward kinematics, and skinning with pose correctives.
    This avoids importing pymomentum.geometry which causes OOM on
    some machines.

    The model's forward() signature is:
        model(identity_coeffs, model_parameters, expression_coeffs)
        -> (vertices, skeleton_state)

    Skeleton state format per joint: [tx, ty, tz, qx, qy, qz, qw, scale]
    (xyzw quaternion convention).

    Joint parameters: 7 per joint [3 translation, 3 euler_xyz, 1 scale],
    obtained via parameter_transform from the 204 model parameters.

    Args:
        model_path: Path to directory containing mhr_model.pt
        lod: Level-of-detail (0-6). The TorchScript model is pre-built
            for a specific LOD (typically 1).
        device: Device to place tensors on
    """

    def __init__(
        self,
        model_path: str,
        lod: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        if lod not in MHR_LOD_SPECS:
            raise ValueError(f"LOD must be 0-6, got {lod}")

        self._lod = lod
        self._device = device

        # Load TorchScript model
        model_file = Path(model_path) / "mhr_model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"MHR TorchScript model not found: {model_file}")

        self._model = torch.jit.load(str(model_file), map_location=device)
        self._model.eval()

        self.enable_grad: bool = False

        ct = self._model.character_torch

        # Extract and cache mesh geometry
        self._rest_vertices = ct.mesh.rest_vertices.to(device).float()
        self._num_vertices = self._rest_vertices.shape[0]
        self._faces = ct.mesh.faces.to(device).long()
        self._num_faces = self._faces.shape[0]

        # Build dense LBS weight matrix from sparse representation
        self._lbs_weights = self._build_lbs_weights(ct.linear_blend_skinning)

        # Cache for canonical triangle frames
        self._canonical_triangle_frames: Tensor | None = None

    def _build_lbs_weights(self, lbs_module) -> Tensor:
        """Build dense (V, J) LBS weight matrix from sparse representation.

        The TorchScript model stores LBS weights as flattened sparse arrays:
        - vert_indices_flattened: vertex index per entry
        - skin_indices_flattened: joint index per entry
        - skin_weights_flattened: weight value per entry

        Args:
            lbs_module: The linear_blend_skinning submodule of character_torch

        Returns:
            Dense weight matrix (V, J) on the mesh device
        """
        V = self._num_vertices
        J = MHR_JOINTS
        weights = torch.zeros(V, J, device=self._device)

        vi = lbs_module.vert_indices_flattened.long().to(self._device)
        ji = lbs_module.skin_indices_flattened.long().to(self._device)
        wt = lbs_module.skin_weights_flattened.to(self._device)
        weights[vi, ji] = wt

        return weights

    @property
    def model_type(self) -> str:
        """Return 'mhr'."""
        return "mhr"

    @property
    def num_joints(self) -> int:
        """Return 127 (MHR joint count)."""
        return MHR_JOINTS

    @property
    def num_vertices(self) -> int:
        """Return vertex count for the loaded LOD."""
        return self._num_vertices

    @property
    def device(self) -> torch.device:
        """Return the device where mesh tensors are stored."""
        return self._device

    @property
    def lod(self) -> int:
        """Return the LOD level."""
        return self._lod

    def get_canonical_vertices(self) -> Tensor:
        """Return T-pose rest vertices.

        MHR's canonical pose is T-pose (unlike SMPL's A-pose).

        Returns:
            Tensor of shape (V, 3) with rest vertices
        """
        return self._rest_vertices

    def get_faces(self) -> Tensor:
        """Return triangle connectivity.

        Returns:
            Tensor of shape (F, 3) with vertex indices per face (int64)
        """
        return self._faces

    def get_lbs_weights(self) -> Tensor:
        """Return dense skinning weights.

        Returns:
            Tensor of shape (V, 127) with per-vertex, per-joint weights
        """
        return self._lbs_weights

    def get_canonical_triangle_frames(self) -> Tensor:
        """Return cached local coordinate frames for canonical mesh triangles.

        Computes frames once on first call and caches for subsequent calls.

        Returns:
            Tensor of shape (F, 3, 3) with orthonormal rotation matrices
        """
        if self._canonical_triangle_frames is None:
            self._canonical_triangle_frames = utils.get_triangle_local_frames(
                self._rest_vertices, self._faces
            )
        return self._canonical_triangle_frames

    def _ensure_batched(
        self, pose: Tensor, shape: Tensor
    ) -> tuple[Tensor, Tensor, int, bool]:
        """Ensure pose and shape tensors have batch dimension.

        Args:
            pose: Model parameters (204,) or (B, 204)
            shape: Identity coefficients (45,) or (B, 45)

        Returns:
            Tuple of (batched_pose, batched_shape, batch_size, was_batched)
        """
        is_batched = pose.dim() == 2
        if not is_batched:
            pose = pose.unsqueeze(0)
            shape = shape.unsqueeze(0)
        pose = pose.to(self._device)
        shape = shape.to(self._device)
        return pose, shape, pose.shape[0], is_batched

    def _run_model(
        self, pose: Tensor, shape: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the TorchScript model forward pass.

        Args:
            pose: Batched model parameters (B, 204)
            shape: Batched identity coefficients (B, 45)

        Returns:
            Tuple of (vertices (B, V, 3), skeleton_state (B, 127, 8))
        """
        expression = torch.zeros(
            pose.shape[0], MHR_EXPRESSION_DIM,
            device=self._device, dtype=pose.dtype,
        )
        return self._model(shape, pose, expression)

    def forward(self, pose: Tensor, shape: Tensor) -> Tensor:
        """Apply native MHR skinning with pose correctives.

        Overrides BaseMesh.forward() to use MHR's native forward pass
        which includes pose correctives (a non-linear MLP adding vertex
        displacements), rather than simple LBS.

        Args:
            pose: Model parameters (204,) or (B, 204)
            shape: Identity coefficients (45,) or (B, 45)

        Returns:
            Posed vertices (V, 3) or (B, V, 3)
        """
        pose, shape, batch_size, is_batched = self._ensure_batched(pose, shape)
        with torch.set_grad_enabled(self.enable_grad):
            verts, _ = self._run_model(pose, shape)

        if not is_batched:
            verts = verts.squeeze(0)

        return verts

    def get_joint_transforms(self, pose: Tensor, shape: Tensor) -> Tensor:
        """Compute per-joint 4x4 transformation matrices.

        Runs the MHR model to get skeleton state, then converts the
        per-joint [tx, ty, tz, qx, qy, qz, qw, scale] representation
        to 4x4 homogeneous transforms.

        Args:
            pose: Model parameters (204,) or (B, 204)
            shape: Identity coefficients (45,) or (B, 45)

        Returns:
            Tensor of shape (J, 4, 4) or (B, J, 4, 4)
        """
        pose, shape, batch_size, is_batched = self._ensure_batched(pose, shape)

        with torch.no_grad():
            _, skel_state = self._run_model(pose, shape)

        # skel_state: (B, 127, 8) = [tx, ty, tz, qx, qy, qz, qw, scale]
        translations = skel_state[..., :3]  # (B, J, 3)
        quats_xyzw = skel_state[..., 3:7]   # (B, J, 4) in xyzw order

        # Convert xyzw quaternions to 3x3 rotation matrices
        rot_matrices = self._quat_xyzw_to_matrix(quats_xyzw)  # (B, J, 3, 3)

        # Build 4x4 transforms
        transforms = torch.zeros(
            batch_size, MHR_JOINTS, 4, 4,
            device=self._device, dtype=skel_state.dtype,
        )
        transforms[..., :3, :3] = rot_matrices
        transforms[..., :3, 3] = translations
        transforms[..., 3, 3] = 1.0

        if not is_batched:
            transforms = transforms.squeeze(0)

        return transforms

    @staticmethod
    def _quat_xyzw_to_matrix(q: Tensor) -> Tensor:
        """Convert xyzw quaternions to 3x3 rotation matrices.

        Args:
            q: Quaternions in xyzw format (*, 4)

        Returns:
            Rotation matrices (*, 3, 3)
        """
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        x2 = x * x
        y2 = y * y
        z2 = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        R = torch.stack([
            torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1),
        ], dim=-2)

        return R

    def encode_pose_to_6d(self, pose: Tensor) -> Tensor:
        """Convert MHR model parameters to 6D rotation representation.

        Runs the parameter transform to get per-joint parameters (127 x 7),
        extracts the Euler XYZ rotations (indices 3:6 of each joint's 7 params),
        and converts to 6D representation.

        Args:
            pose: Model parameters (204,)

        Returns:
            6D rotation representation (127, 6)
        """
        ct = self._model.character_torch

        # Build full parameter vector: [identity(45) + pose(204)] = 249
        # Use zeros for identity since we only care about pose rotations
        identity = torch.zeros(1, MHR_SHAPE_DIM, device=self._device, dtype=pose.dtype)
        pose_batched = pose.unsqueeze(0) if pose.dim() == 1 else pose[:1]
        full_params = torch.cat([identity, pose_batched], dim=1)  # (1, 249)

        # parameter_transform: (1, 249) -> (1, 889) = (1, 127*7)
        joint_params = ct.parameter_transform(full_params)
        joint_params = joint_params.reshape(MHR_JOINTS, 7)

        # Extract Euler XYZ rotations (indices 3:6 of each joint's 7 params)
        euler_xyz = joint_params[:, 3:6]  # (127, 3)

        return euler_xyz_to_6d(euler_xyz)
