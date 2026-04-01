"""GaussianAvatar: Top-level model combining mesh, Gaussians, and deformation.

This module provides the GaussianAvatar class that orchestrates the complete
forward pass from pose/shape parameters to renderable Gaussian properties.
"""

import torch
from torch import Tensor, nn

from gaussian_avatar.configs import DeformationConfig
from gaussian_avatar.constants import MODEL_SPECS
from gaussian_avatar.models.deformation import DeformationMLP
from gaussian_avatar.models.deformation.encoding import positional_encoding
from gaussian_avatar.models.deformation.mlp import construct_quaternion_offset
from gaussian_avatar.models.gaussian import GaussianModel
from gaussian_avatar.models.mesh import BaseMesh


class GaussianAvatar(nn.Module):
    """Complete Gaussian avatar model combining mesh, Gaussians, and deformation.

    Orchestrates the forward pass from pose/shape parameters to final
    Gaussian properties ready for gsplat rendering.

    The pipeline:
    1. Mesh deformation: canonical → posed vertices via LBS
    2. Triangle properties: compute transforms and stretches
    3. MLP input: encode canonical/posed positions + pose
    4. Deformation MLP: predict pose-dependent offsets
    5. Global properties: apply offsets and transform to world space
    6. gsplat format: convert for rendering

    Args:
        mesh: Initialized mesh model (SMPLMesh or compatible BaseMesh)
        num_gaussians: Total number of Gaussians to allocate. Each is randomly
            assigned to a mesh triangle.
        deformation_config: MLP configuration (uses defaults if None)
        max_stretch: Maximum allowed per-axis stretch factor when transforming
            Gaussians from canonical to posed space. Stretches are clamped to
            [1/max_stretch, max_stretch]. Default 10.0.
        max_local_offset: When set, bounds local positions via
            ``max_local_offset * tanh(raw)``. Units are meters. None disables.
    """

    def __init__(
        self,
        mesh: BaseMesh,
        num_gaussians: int = 100000,
        deformation_config: DeformationConfig | None = None,
        disable_offsets: bool = False,
        use_sh: bool = False,
        max_sh_degree: int = 3,
        max_stretch: float = 10.0,
        max_local_offset: float | None = None,
    ):
        super().__init__()

        self.mesh = mesh
        self.disable_offsets = disable_offsets
        self.use_sh = use_sh
        self.max_sh_degree = max_sh_degree
        self.max_stretch = max_stretch

        num_faces = mesh.get_faces().shape[0]

        self.gaussian_model = GaussianModel(
            num_gaussians=num_gaussians,
            num_faces=num_faces,
            device=mesh.device,
            use_sh=use_sh,
            max_sh_degree=max_sh_degree,
            max_local_offset=max_local_offset,
        )

        self.gaussian_model.initialize_from_mesh(mesh)

        self._expected_pose_dim = MODEL_SPECS[mesh.model_type]["pose_dim"]

        if disable_offsets:
            self.deformation_mlp = None
            self._pose_joint_indices = None
        else:
            if deformation_config is None:
                deformation_config = DeformationConfig()

            self._pose_joint_indices = deformation_config.pose_joint_indices

            # Determine how many 6D joint rotations encode_pose_to_6d produces.
            # For axis-angle models (SMPL/SMPL-X): pose_dim // 3
            #   (SMPL-X has 55 skeleton joints but only 22 in the pose vector)
            # For other representations (MHR): all joints are encoded (127)
            spec = MODEL_SPECS[mesh.model_type]
            if spec.get("pose_representation") == "axis_angle":
                num_joints = spec["pose_dim"] // 3
            else:
                num_joints = spec["joints"]

            if self._pose_joint_indices is not None:
                num_joints = len(self._pose_joint_indices)

            self.deformation_mlp = DeformationMLP(
                num_frequencies=deformation_config.num_frequencies,
                num_joints=num_joints,
                hidden_dim=deformation_config.hidden_dim,
                num_layers=deformation_config.num_layers,
                activation=deformation_config.activation,
                output_init_std=deformation_config.output_init_std,
                dropout=deformation_config.dropout,
                include_canonical=deformation_config.include_canonical,
                include_betas=deformation_config.include_betas,
                num_betas=deformation_config.num_betas,
                use_sh=use_sh,
            )

        self._canonical_vertices: Tensor | None = None
        self._canonical_transforms: Tensor | None = None

    def _ensure_canonical_cache(self) -> None:
        """Compute and cache canonical mesh properties on first use."""
        if self._canonical_transforms is None:
            canonical_vertices = self.mesh.get_canonical_vertices()
            self._canonical_vertices = canonical_vertices
            self._canonical_transforms = self.mesh.get_triangle_transforms(
                canonical_vertices
            )

    def _compute_base_positions(self, transforms: Tensor) -> Tensor:
        """Compute global positions using triangle transforms and local positions.

        Handles both batched (B, F, 4, 4) and unbatched (F, 4, 4) transforms.

        Args:
            transforms: Triangle homogeneous transforms (F, 4, 4) or (B, F, 4, 4)

        Returns:
            Global positions for each Gaussian (N, 3) or (B, N, 3)
        """
        is_batched = transforms.dim() == 4
        tri_idx = self.gaussian_model.triangle_indices

        if is_batched:
            B = transforms.shape[0]
            tri_idx_expanded = tri_idx.unsqueeze(0).expand(B, -1)
            batch_idx = torch.arange(B, device=transforms.device)[:, None]
            T = transforms[batch_idx, tri_idx_expanded]

            R = T[:, :, :3, :3]
            t = T[:, :, :3, 3]

            local_pos = self.gaussian_model.local_positions.unsqueeze(0).expand(B, -1, -1)
            return torch.einsum("bnij,bnj->bni", R, local_pos) + t
        else:
            T = transforms[tri_idx]
            R = T[:, :3, :3]
            t = T[:, :3, 3]

            local_pos = self.gaussian_model.local_positions
            return torch.einsum("nij,nj->ni", R, local_pos) + t

    def get_canonical_positions(self) -> Tensor:
        """Compute global canonical positions for all Gaussians.

        Returns the actual world-space positions of Gaussians in the canonical
        (A-pose) mesh, by transforming local positions through the canonical
        triangle transforms.

        Returns:
            Global canonical positions (N, 3)
        """
        self._ensure_canonical_cache()
        return self._compute_base_positions(self._canonical_transforms)

    def _encode_pose(self, pose: Tensor) -> Tensor:
        """Convert pose parameters to 6D rotation representation.

        Delegates to the mesh's encode_pose_to_6d() method, which handles
        model-specific pose representations (axis-angle for SMPL/SMPL-X,
        Euler XYZ via FK pipeline for MHR).

        When pose_joint_indices is configured, slices the encoded pose to
        include only the selected joints, reducing MLP input dimensionality.

        Args:
            pose: Pose parameters (P,) where P depends on mesh type
                  SMPL: 72 dims, SMPL-X: 66 dims, MHR: 204 dims

        Returns:
            6D pose representation (J, 6) or (K, 6) if pose_joint_indices is set
        """
        pose_6d = self.mesh.encode_pose_to_6d(pose)  # (J, 6)
        if self._pose_joint_indices is not None:
            indices = torch.tensor(
                self._pose_joint_indices, device=pose_6d.device, dtype=torch.long
            )
            pose_6d = pose_6d[indices]  # (K, 6)
        return pose_6d

    def _format_for_gsplat(
        self,
        global_props: dict[str, Tensor],
        mlp_outputs: dict[str, Tensor],
        canonical_positions: Tensor,
        posed_base_positions: Tensor,
        canonical_rotations: Tensor,
        canonical_scales: Tensor,
    ) -> dict[str, Tensor]:
        """Convert internal representation to gsplat-ready format.

        Handles both batched and unbatched inputs based on tensor dimensions.

        Args:
            global_props: Output from GaussianModel.get_global_properties()
            mlp_outputs: Output from DeformationMLP.forward()
            canonical_positions: Gaussian positions in canonical space (N, 3)
            posed_base_positions: Gaussian positions before MLP offset (N, 3) or (B, N, 3)
            canonical_rotations: Gaussian rotations in canonical space (N, 4)
            canonical_scales: Gaussian scales in canonical space (N, 3)

        Returns:
            Dictionary with gsplat-ready properties and auxiliary outputs
        """
        is_batched = posed_base_positions.dim() == 3

        if is_batched:
            B = global_props["means"].shape[0]

            local_positions = self.gaussian_model.local_positions.unsqueeze(0).expand(B, -1, -1)
            log_scales = self.gaussian_model.local_scales.unsqueeze(0).expand(B, -1, -1)
            rotation_xyz = mlp_outputs["rotation_offsets_raw"]
        else:
            local_positions = self.gaussian_model.local_positions
            log_scales = self.gaussian_model.local_scales
            rotation_xyz = mlp_outputs["rotation_offsets_raw"]

        return {
            "means": global_props["means"],
            "quats": global_props["rotations"],
            "scales": global_props["scales"],
            "colors": global_props["colors"],
            "opacities": global_props["opacities"].squeeze(-1),
            "canonical_positions": canonical_positions,
            "posed_positions": posed_base_positions,
            "canonical_rotations": canonical_rotations,
            "canonical_scales": canonical_scales,
            "posed_rotations": global_props["rotations"],
            "posed_scales": global_props["scales"],
            "local_positions": local_positions,
            "log_scales": log_scales,
            "offsets": {
                "position": mlp_outputs["position_offsets"],
                "rotation_xyz": rotation_xyz,
                "scale": mlp_outputs["scale_offsets"],
                "color": mlp_outputs["color_offsets"],
            },
        }

    def forward(self, pose: Tensor, betas: Tensor, alpha: float | None = None) -> dict[str, Tensor]:
        """Complete forward pass from pose/shape to renderable Gaussians.

        Args:
            pose: Pose parameters (P,) where P depends on mesh type
                  SMPL: 72 dims (3 global + 69 body)
                  SMPL-X: 66 dims (3 global + 63 body)
            betas: Shape coefficients (B,), typically 10
            alpha: Hann window position for frequency annealing. None disables.

        Returns:
            Dictionary with gsplat-ready properties:
                - 'means': (N, 3) global positions
                - 'quats': (N, 4) quaternions in wxyz format
                - 'scales': (N, 3) linear scales (positive)
                - 'colors': (N, 3) RGB in [0, 1]
                - 'opacities': (N,) opacity in [0, 1]

            Also includes auxiliary outputs for loss computation:
                - 'canonical_positions': (N, 3) for AIAP
                - 'posed_positions': (N, 3) pre-offset positions for AIAP
                - 'local_positions': (N, 3) final local positions
                - 'log_scales': (N, 3) final log-scales
                - 'offsets': dict of MLP outputs for regularization
        """
        if pose.shape[-1] < self._expected_pose_dim:
            raise ValueError(
                f"Expected pose dim >= {self._expected_pose_dim} for "
                f"{self.mesh.model_type}, got {pose.shape[-1]}"
            )

        posed_vertices = self.mesh.forward(pose, betas)

        self._ensure_canonical_cache()

        posed_transforms = self.mesh.get_triangle_transforms(posed_vertices)
        stretches = self.mesh.get_triangle_stretches(
            canonical_vertices=self._canonical_vertices,
            posed_vertices=posed_vertices,
            canonical_frames=self.mesh.get_canonical_triangle_frames(),
            max_stretch=self.max_stretch,
        )

        canonical_positions = self._compute_base_positions(self._canonical_transforms)
        posed_base_positions = self._compute_base_positions(posed_transforms)

        if self.disable_offsets:
            mlp_outputs = self._zero_offsets(canonical_positions)
            global_props = self.gaussian_model.get_global_properties(
                triangle_transforms=posed_transforms,
                triangle_stretches=stretches,
            )
        else:
            pose_6d = self._encode_pose(pose)
            mlp_kwargs: dict = {}
            if self.deformation_mlp.include_canonical:
                mlp_kwargs["canonical_positions"] = canonical_positions
            if self.deformation_mlp.include_betas:
                mlp_kwargs["betas"] = betas
            mlp_outputs = self.deformation_mlp(
                posed_positions=posed_base_positions,
                pose_6d=pose_6d,
                alpha=alpha,
                **mlp_kwargs,
            )
            global_props = self.gaussian_model.get_global_properties(
                triangle_transforms=posed_transforms,
                triangle_stretches=stretches,
                position_offsets=mlp_outputs["position_offsets"],
                rotation_offsets=mlp_outputs["rotation_offsets"],
                scale_offsets=mlp_outputs["scale_offsets"],
                color_offsets=mlp_outputs["color_offsets"],
            )

        # Compute canonical rotations and scales (no MLP offsets, no deformation)
        canonical_rotations = self.gaussian_model.get_rotations(self._canonical_transforms)
        canonical_stretches = torch.ones(
            self._canonical_transforms.shape[0], 3, device=self._canonical_transforms.device
        )
        canonical_scales = self.gaussian_model.get_scales(canonical_stretches)

        return self._format_for_gsplat(
            global_props, mlp_outputs, canonical_positions, posed_base_positions,
            canonical_rotations, canonical_scales,
        )

    def forward_with_cache(
        self,
        posed_transforms: Tensor,
        stretches: Tensor,
        pose_6d: Tensor,
        betas: Tensor | None = None,
        alpha: float | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass using pre-computed mesh transforms (skips body model).

        Accepts cached mesh outputs from MeshCache instead of running the
        expensive body model forward pass and triangle computations. The
        MLP and Gaussian property computations proceed identically to forward().

        Args:
            posed_transforms: Pre-computed triangle transforms (F, 4, 4)
            stretches: Pre-computed triangle stretches (F, 3)
            pose_6d: Pre-computed 6D pose encoding (J, 6)
            betas: Shape coefficients for MLP input (when include_betas=True)
            alpha: Hann window position for frequency annealing. None disables.

        Returns:
            Same dictionary as forward() with gsplat-ready properties and
            auxiliary outputs for loss computation.
        """
        self._ensure_canonical_cache()

        canonical_positions = self._compute_base_positions(self._canonical_transforms)
        posed_base_positions = self._compute_base_positions(posed_transforms)

        if self.disable_offsets:
            mlp_outputs = self._zero_offsets(canonical_positions)
            global_props = self.gaussian_model.get_global_properties(
                triangle_transforms=posed_transforms,
                triangle_stretches=stretches,
            )
        else:
            mlp_kwargs: dict = {}
            if self.deformation_mlp.include_canonical:
                mlp_kwargs["canonical_positions"] = canonical_positions
            if self.deformation_mlp.include_betas:
                mlp_kwargs["betas"] = betas
            mlp_outputs = self.deformation_mlp(
                posed_positions=posed_base_positions,
                pose_6d=pose_6d,
                alpha=alpha,
                **mlp_kwargs,
            )
            global_props = self.gaussian_model.get_global_properties(
                triangle_transforms=posed_transforms,
                triangle_stretches=stretches,
                position_offsets=mlp_outputs["position_offsets"],
                rotation_offsets=mlp_outputs["rotation_offsets"],
                scale_offsets=mlp_outputs["scale_offsets"],
                color_offsets=mlp_outputs["color_offsets"],
            )

        canonical_rotations = self.gaussian_model.get_rotations(self._canonical_transforms)
        canonical_stretches = torch.ones(
            self._canonical_transforms.shape[0], 3, device=self._canonical_transforms.device
        )
        canonical_scales = self.gaussian_model.get_scales(canonical_stretches)

        return self._format_for_gsplat(
            global_props, mlp_outputs, canonical_positions, posed_base_positions,
            canonical_rotations, canonical_scales,
        )

    def forward_batch(self, poses: Tensor, betas: Tensor, alpha: float | None = None) -> dict[str, Tensor]:
        """Batched forward pass for multiple poses.

        Processes multiple poses in parallel for training efficiency.

        Args:
            poses: Batched pose parameters (B, P) where P depends on mesh type
                   SMPL: (B, 72), SMPL-X: (B, 66)
            betas: Shape coefficients (B, num_betas) or (num_betas,)
                   If unbatched (num_betas,), will be broadcast to all poses
            alpha: Hann window position for frequency annealing. None disables.

        Returns:
            Dictionary with batched gsplat-ready properties:
                - 'means': (B, N, 3) global positions
                - 'quats': (B, N, 4) quaternions in wxyz format
                - 'scales': (B, N, 3) linear scales (positive)
                - 'colors': (B, N, 3) RGB in [0, 1]
                - 'opacities': (B, N) opacity in [0, 1]

            Also includes auxiliary outputs for loss computation:
                - 'canonical_positions': (N, 3) for AIAP (same across batch)
                - 'posed_positions': (B, N, 3) pre-offset positions for AIAP
                - 'local_positions': (B, N, 3) final local positions
                - 'log_scales': (B, N, 3) final log-scales
                - 'offsets': dict of MLP outputs for regularization (B, N, ...)
        """
        B = poses.shape[0]

        if poses.shape[-1] < self._expected_pose_dim:
            raise ValueError(
                f"Expected pose dim >= {self._expected_pose_dim} for "
                f"{self.mesh.model_type}, got {poses.shape[-1]}"
            )

        if betas.dim() == 1:
            betas = betas.unsqueeze(0).expand(B, -1)

        posed_vertices = self.mesh.forward(poses, betas)

        self._ensure_canonical_cache()

        posed_transforms = self.mesh.get_triangle_transforms(posed_vertices)
        stretches = self.mesh.get_triangle_stretches(
            canonical_vertices=self._canonical_vertices,
            posed_vertices=posed_vertices,
            canonical_frames=self.mesh.get_canonical_triangle_frames(),
            max_stretch=self.max_stretch,
        )

        canonical_positions = self._compute_base_positions(self._canonical_transforms)
        posed_base_positions = self._compute_base_positions(posed_transforms)

        if self.disable_offsets:
            mlp_outputs = self._zero_offsets(canonical_positions, batch_size=B)
            global_props = self.gaussian_model.get_global_properties(
                triangle_transforms=posed_transforms,
                triangle_stretches=stretches,
            )
        else:
            mlp_kwargs: dict = {}
            if self.deformation_mlp.include_canonical:
                mlp_kwargs["canonical_positions"] = canonical_positions
            if self.deformation_mlp.include_betas:
                mlp_kwargs["betas"] = betas
            mlp_outputs = self._run_mlp_batched(
                posed_positions=posed_base_positions,
                poses=poses,
                alpha=alpha,
                **mlp_kwargs,
            )
            global_props = self.gaussian_model.get_global_properties(
                triangle_transforms=posed_transforms,
                triangle_stretches=stretches,
                position_offsets=mlp_outputs["position_offsets"],
                rotation_offsets=mlp_outputs["rotation_offsets"],
                scale_offsets=mlp_outputs["scale_offsets"],
                color_offsets=mlp_outputs["color_offsets"],
            )

        # Compute canonical rotations and scales (no MLP offsets, no deformation)
        canonical_rotations = self.gaussian_model.get_rotations(self._canonical_transforms)
        canonical_stretches = torch.ones(
            self._canonical_transforms.shape[0], 3, device=self._canonical_transforms.device
        )
        canonical_scales = self.gaussian_model.get_scales(canonical_stretches)

        return self._format_for_gsplat(
            global_props, mlp_outputs, canonical_positions, posed_base_positions,
            canonical_rotations, canonical_scales,
        )

    def _zero_offsets(
        self, canonical_positions: Tensor, batch_size: int | None = None
    ) -> dict[str, Tensor]:
        """Return zero-valued MLP outputs when offsets are disabled.

        Args:
            canonical_positions: Canonical Gaussian positions (N, 3)
            batch_size: If provided, return batched tensors (B, N, ...)

        Returns:
            Dictionary matching DeformationMLP output format with all zeros
        """
        N = canonical_positions.shape[0]
        device = canonical_positions.device

        color_offsets_val = None if self.use_sh else torch.zeros(
            *((batch_size, N, 3) if batch_size else (N, 3)), device=device
        )

        if batch_size is not None:
            rot_offsets = torch.zeros(batch_size, N, 4, device=device)
            rot_offsets[..., 0] = 1.0  # identity quaternion (wxyz)
            return {
                "position_offsets": torch.zeros(batch_size, N, 3, device=device),
                "rotation_offsets": rot_offsets,
                "rotation_offsets_raw": torch.zeros(batch_size, N, 3, device=device),
                "scale_offsets": torch.zeros(batch_size, N, 3, device=device),
                "color_offsets": color_offsets_val,
            }

        rot_offsets = torch.zeros(N, 4, device=device)
        rot_offsets[..., 0] = 1.0  # identity quaternion (wxyz)
        return {
            "position_offsets": torch.zeros(N, 3, device=device),
            "rotation_offsets": rot_offsets,
            "rotation_offsets_raw": torch.zeros(N, 3, device=device),
            "scale_offsets": torch.zeros(N, 3, device=device),
            "color_offsets": color_offsets_val,
        }

    def _run_mlp_batched(
        self,
        posed_positions: Tensor,
        poses: Tensor,
        alpha: float | None = None,
        canonical_positions: Tensor | None = None,
        betas: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run deformation MLP for batched poses.

        The MLP processes B*N total inputs (all Gaussians for all poses),
        then reshapes outputs to (B, N, ...).

        Args:
            posed_positions: Posed Gaussian positions (B, N, 3)
            poses: Batched poses (B, P)
            alpha: Hann window position for frequency annealing. None disables.
            canonical_positions: Canonical positions (N, 3), used when include_canonical=True
            betas: Shape coefficients (B, num_betas) or (num_betas,), used when include_betas=True

        Returns:
            Dictionary with batched MLP outputs:
                - 'position_offsets': (B, N, 3)
                - 'rotation_offsets': (B, N, 4) wxyz quaternion
                - 'scale_offsets': (B, N, 3)
                - 'color_offsets': (B, N, 3)
        """
        B = poses.shape[0]
        N = posed_positions.shape[1]

        posed_flat = posed_positions.reshape(B * N, 3)

        pose_6d_list = []
        for b in range(B):
            pose_6d_list.append(self._encode_pose(poses[b]))
        pose_6d = torch.stack(pose_6d_list, dim=0)

        pose_flat = pose_6d.reshape(B, -1)
        pose_repeated = pose_flat.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        # Prepare optional canonical/betas for flat MLP call
        canonical_flat = None
        if canonical_positions is not None:
            canonical_flat = canonical_positions.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 3)
        betas_flat = None
        if betas is not None:
            if betas.dim() == 1:
                betas_flat = betas  # (num_betas,) - handled by _run_mlp_flat
            else:
                # Expand (B, num_betas) → (B, N, num_betas) → (B*N, num_betas)
                betas_flat = betas.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        mlp_output = self._run_mlp_flat(
            posed_flat, pose_repeated, alpha=alpha,
            canonical_positions=canonical_flat, betas=betas_flat,
        )
        color_offsets = mlp_output["color_offsets"]
        return {
            "position_offsets": mlp_output["position_offsets"].reshape(B, N, 3),
            "rotation_offsets": mlp_output["rotation_offsets"].reshape(B, N, 4),
            "rotation_offsets_raw": mlp_output["rotation_offsets_raw"].reshape(B, N, 3),
            "scale_offsets": mlp_output["scale_offsets"].reshape(B, N, 3),
            "color_offsets": color_offsets.reshape(B, N, 3) if color_offsets is not None else None,
        }

    def _run_mlp_flat(
        self,
        posed_positions: Tensor,
        pose_encoded: Tensor,
        alpha: float | None = None,
        canonical_positions: Tensor | None = None,
        betas: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run MLP on pre-encoded flat inputs.

        This bypasses the MLP's internal encoding since we've already
        prepared the batched inputs.

        Args:
            posed_positions: (M, 3) where M = B*N
            pose_encoded: (M, J*6) pre-encoded and repeated pose
            alpha: Hann window position for frequency annealing. None disables.
            canonical_positions: (M, 3) canonical positions, used when include_canonical=True
            betas: Shape coefficients, used when include_betas=True

        Returns:
            MLP outputs with shapes (M, ...)
        """
        M = posed_positions.shape[0]
        dtype = posed_positions.dtype

        pe_posed = positional_encoding(
            posed_positions, self.deformation_mlp.num_frequencies, alpha=alpha
        )

        pose_encoded = pose_encoded.to(dtype=dtype)

        # Ordering must match training: [pe_canonical, pe_posed, pose, betas]
        parts = []
        if self.deformation_mlp.include_canonical and canonical_positions is not None:
            pe_canonical = positional_encoding(
                canonical_positions, self.deformation_mlp.num_frequencies, alpha=alpha
            )
            parts.append(pe_canonical)
        parts.append(pe_posed)
        parts.append(pose_encoded)
        if self.deformation_mlp.include_betas and betas is not None:
            if betas.dim() == 1:
                betas_broadcast = betas.unsqueeze(0).expand(M, -1).to(dtype=dtype)
            else:
                betas_broadcast = betas.to(dtype=dtype)  # already (M, num_betas)
            parts.append(betas_broadcast)

        x = torch.cat(parts, dim=-1)

        hidden = self.deformation_mlp.layers(x)
        output = self.deformation_mlp.output_layer(hidden)

        position_offsets = output[:, 0:3]
        rotation_xyz = output[:, 3:6]
        scale_offsets = output[:, 6:9]
        color_offsets = None if self.use_sh else output[:, 9:12]

        rotation_offsets = construct_quaternion_offset(rotation_xyz)

        return {
            "position_offsets": position_offsets,
            "rotation_offsets": rotation_offsets,
            "rotation_offsets_raw": rotation_xyz,
            "scale_offsets": scale_offsets,
            "color_offsets": color_offsets,
        }