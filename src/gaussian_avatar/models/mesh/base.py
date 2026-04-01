"""Abstract base class for LBS-deformable triangle meshes."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from gaussian_avatar.models.deformation.encoding import axis_angle_to_6d
from gaussian_avatar.models.mesh import utils


class BaseMesh(ABC):
    """Abstract base class for LBS-deformable triangle meshes.

    This class defines the interface for any mesh that supports Linear Blend
    Skinning (LBS), enabling the Gaussian avatar system to work with SMPL,
    SMPL-X, or other body models.

    Subclasses must implement the abstract methods to provide mesh-specific
    functionality. Derived methods for triangle properties are implemented
    in this base class.
    """

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier (e.g., 'smpl', 'smplx').

        Returns:
            String identifying the underlying model type
        """
        pass

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Return the number of joints in the skeleton.

        Returns:
            Integer count of joints (e.g., 24 for SMPL, 55 for SMPL-X)
        """
        pass

    @property
    @abstractmethod
    def num_vertices(self) -> int:
        """Return the number of vertices in the mesh.

        Returns:
            Integer count of vertices (e.g., 6890 for SMPL, 10475 for SMPL-X)
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device where mesh tensors are stored.

        Returns:
            torch.device indicating CPU or CUDA device
        """
        pass

    @abstractmethod
    def get_canonical_vertices(self) -> Tensor:
        """Return vertices in canonical (unposed) space.

        Returns:
            Tensor of shape (V, 3) where V is number of vertices
        """
        pass

    @abstractmethod
    def get_faces(self) -> Tensor:
        """Return triangle connectivity.

        Returns:
            Tensor of shape (F, 3) with vertex indices per face
        """
        pass

    @abstractmethod
    def get_lbs_weights(self) -> Tensor:
        """Return skinning weights.

        Returns:
            Tensor of shape (V, J) where J is number of joints
        """
        pass

    @abstractmethod
    def get_joint_transforms(self, pose: Tensor, shape: Tensor) -> Tensor:
        """Compute per-joint transformation matrices.

        Args:
            pose: Pose parameters (depends on model)
            shape: Shape coefficients (B,)

        Returns:
            Tensor of shape (J, 4, 4) or (B, J, 4, 4) with 4x4 transform per joint
        """
        pass

    def encode_pose_to_6d(self, pose: Tensor) -> Tensor:
        """Convert pose parameters to 6D rotation representation.

        Default implementation assumes axis-angle parameterization: reshapes
        the flat pose vector into per-joint axis-angle rotations and converts
        each to 6D using the first two columns of the rotation matrix.

        The number of joints is derived from the pose vector length (P // 3),
        not from self.num_joints, because some models (e.g., SMPL-X) have more
        skeleton joints than pose parameters (55 joints but only 22 in body pose).

        Subclasses with different pose representations (e.g., Euler XYZ for MHR)
        should override this method.

        Args:
            pose: Flat pose vector (P,) where P = num_pose_joints * 3 for axis-angle

        Returns:
            6D rotation representation (J, 6) for each pose joint
        """
        num_pose_joints = pose.shape[-1] // 3
        pose_per_joint = pose.reshape(num_pose_joints, 3)
        return axis_angle_to_6d(pose_per_joint)

    def get_canonical_triangle_frames(self) -> Tensor:
        """Return local coordinate frames for canonical mesh triangles.

        Computes orthonormal frames from canonical vertices and faces.
        Subclasses may override to add caching.

        Returns:
            Tensor of shape (F, 3, 3) with orthonormal rotation matrices
        """
        return utils.get_triangle_local_frames(
            self.get_canonical_vertices(), self.get_faces()
        )

    def forward(self, pose: Tensor, shape: Tensor) -> Tensor:
        """Apply LBS forward skinning to get posed vertices.

        Args:
            pose: Pose parameters, shape (pose_dims,) or (B, pose_dims)
            shape: Shape coefficients, shape (num_betas,) or (B, num_betas)

        Returns:
            Tensor of shape (V, 3) or (B, V, 3) with posed vertices
        """
        canonical_verts = self.get_canonical_vertices()
        lbs_weights = self.get_lbs_weights()
        joint_transforms = self.get_joint_transforms(pose, shape)

        is_batched = joint_transforms.dim() == 4
        if not is_batched:
            joint_transforms = joint_transforms.unsqueeze(0)

        batch_size = joint_transforms.shape[0]
        canonical_verts = canonical_verts.unsqueeze(0).expand(batch_size, -1, -1)
        lbs_weights = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)

        weighted_transforms = torch.einsum(
            "bvj,bjmn->bvmn", lbs_weights, joint_transforms
        )

        ones = torch.ones(
            batch_size,
            canonical_verts.shape[1],
            1,
            device=canonical_verts.device,
            dtype=canonical_verts.dtype,
        )
        verts_homo = torch.cat([canonical_verts, ones], dim=-1)
        posed_homo = torch.einsum("bvmn,bvn->bvm", weighted_transforms, verts_homo)
        posed_verts = posed_homo[..., :3]

        if not is_batched:
            posed_verts = posed_verts.squeeze(0)

        return posed_verts

    def get_triangle_vertices(self, vertices: Tensor) -> Tensor:
        """Group vertices by triangle.

        Args:
            vertices: Vertex positions (V, 3) or (B, V, 3)

        Returns:
            Triangle vertices (F, 3, 3) or (B, F, 3, 3)
        """
        return utils.get_triangle_vertices(vertices, self.get_faces())

    def get_triangle_centroids(self, vertices: Tensor) -> Tensor:
        """Compute centroid of each triangle.

        Args:
            vertices: Vertex positions (V, 3) or (B, V, 3)

        Returns:
            Triangle centroids (F, 3) or (B, F, 3)
        """
        return utils.get_triangle_centroids(vertices, self.get_faces())

    def get_triangle_local_frames(self, vertices: Tensor) -> Tensor:
        """Compute local coordinate frame for each triangle.

        The local frame is an orthonormal rotation matrix [x|y|z] where:
        - x_axis: tangent along first edge (V1-V0), orthogonalized to normal
        - y_axis: bitangent (cross(z, x))
        - z_axis: triangle normal

        Args:
            vertices: Vertex positions (V, 3) or (B, V, 3)

        Returns:
            Rotation matrices (F, 3, 3) or (B, F, 3, 3) where columns are [x|y|z]
        """
        return utils.get_triangle_local_frames(vertices, self.get_faces())

    def get_triangle_transforms(self, vertices: Tensor) -> Tensor:
        """Compute full 4x4 transforms for each triangle.

        Combines rotation (local frame) with translation (centroid).

        Args:
            vertices: Vertex positions (V, 3) or (B, V, 3)

        Returns:
            Transform matrices (F, 4, 4) or (B, F, 4, 4)
        """
        return utils.get_triangle_transforms(vertices, self.get_faces())

    def get_triangle_stretches(
        self,
        canonical_vertices: Tensor,
        posed_vertices: Tensor,
        canonical_frames: Tensor | None = None,
        max_stretch: float = 2.0,
    ) -> Tensor:
        """Compute per-axis stretch factors between canonical and posed triangles.

        Stretch factors capture how each triangle axis has scaled from canonical
        to posed state.

        Args:
            canonical_vertices: Canonical vertex positions (V, 3) or (B, V, 3)
            posed_vertices: Posed vertex positions (V, 3) or (B, V, 3)
            canonical_frames: Pre-computed canonical local frames (F, 3, 3) or (B, F, 3, 3).
                If None, computed from canonical_vertices. Pass this to avoid
                redundant computation when frames are already available.
            max_stretch: Maximum allowed stretch factor per axis.
                Stretches are clamped to [1/max_stretch, max_stretch].

        Returns:
            Stretch factors (F, 3) or (B, F, 3) for each axis [x, y, z]
        """
        return utils.get_triangle_stretches(
            canonical_vertices,
            posed_vertices,
            self.get_faces(),
            canonical_frames=canonical_frames,
            max_stretch=max_stretch,
        )
