"""GaussianModel class for parameter storage and management.

This module provides the core GaussianModel class that stores learnable
Gaussian parameters in local triangle coordinate space and manages
triangle assignments for each Gaussian.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .rotations import (
    normalize_quaternion,
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)


class GaussianModel(nn.Module):
    """Stores and manages Gaussian parameters in local triangle space.

    All Gaussian geometric properties (position, rotation, scale) are stored
    in the local coordinate frame of their parent triangle. This provides
    automatic transformation inheritance when triangles deform.

    Args:
        num_gaussians: Total number of Gaussians to allocate
        num_faces: Number of triangles in the mesh (for random assignment)
        device: Torch device for parameter allocation
        use_sh: Use spherical harmonics coefficients instead of flat RGB colors
        max_sh_degree: Maximum SH degree (0-3). Determines coefficient count:
            K = (max_sh_degree + 1)^2
        max_local_offset: When set, local positions are bounded via
            ``max_local_offset * tanh(raw_positions)``, providing a hard upper
            bound on how far Gaussians can drift from their parent triangle
            centroid. Units match the mesh coordinate system (meters for SMPL).
            None disables bounding.

    Attributes:
        local_positions: (N, 3) local position offsets from triangle centroids
        local_rotations: (N, 4) local quaternions in wxyz format
        local_scales: (N, 3) log-scale values per axis
        base_colors: (N, 3) pre-sigmoid RGB values (RGB mode only)
        sh_coefficients: (N, K, 3) SH coefficients (SH mode only)
        raw_opacities: (N, 1) pre-sigmoid opacity values
        triangle_indices: (N,) parent triangle index for each Gaussian
    """

    # SH constant: C0 = 0.5 * sqrt(1/pi)
    _SH_C0 = 0.28209479177387814

    def __init__(
        self,
        num_gaussians: int,
        num_faces: int,
        device: str | torch.device = "cpu",
        use_sh: bool = False,
        max_sh_degree: int = 3,
        max_local_offset: float | None = None,
    ) -> None:
        super().__init__()
        self._num_gaussians = num_gaussians
        self._num_faces = num_faces
        self.use_sh = use_sh
        self.max_sh_degree = max_sh_degree
        self.max_local_offset = max_local_offset
        device = torch.device(device)

        # Build instance-level param name/key mappings (mode-aware)
        self._PARAM_NAMES = [
            "_local_positions",
            "_local_rotations",
            "_local_scales",
            "_sh_coefficients" if use_sh else "_base_colors",
            "_opacity",
        ]
        if use_sh:
            self._PARAM_KEY_MAP = {
                "local_positions": "_local_positions",
                "local_rotations": "_local_rotations",
                "local_scales": "_local_scales",
                "sh_coefficients": "_sh_coefficients",
                "opacity": "_opacity",
            }
        else:
            self._PARAM_KEY_MAP = {
                "local_positions": "_local_positions",
                "local_rotations": "_local_rotations",
                "local_scales": "_local_scales",
                "base_colors": "_base_colors",
                "opacity": "_opacity",
            }

        # offset from triangle centroid in local frame
        self.register_parameter(
            "_local_positions",
            nn.Parameter(torch.zeros(num_gaussians, 3, device=device)),
        )

        # quats
        rotations = torch.zeros(num_gaussians, 4, device=device)
        rotations[:, 0] = 1.0  # w = 1 for identity quaternion
        self.register_parameter(
            "_local_rotations",
            nn.Parameter(rotations),
        )

        # log-scale, default 0 means actual scale = 1
        self.register_parameter(
            "_local_scales",
            nn.Parameter(torch.zeros(num_gaussians, 3, device=device)),
        )

        if use_sh:
            # SH coefficients: (N, K, 3) where K = (max_degree+1)^2
            K = (max_sh_degree + 1) ** 2
            sh = torch.zeros(num_gaussians, K, 3, device=device)
            # Initialize DC band to represent 0.5 gray: color = C0 * sh[0]
            sh[:, 0, :] = 0.0  # C0 * 0.0 + 0.5 (gsplat offset) = 0.5 gray
            self.register_parameter(
                "_sh_coefficients",
                nn.Parameter(sh),
            )
        else:
            # pre-sigmoid RGB, initialized for 0.5 post-activation
            self.register_parameter(
                "_base_colors",
                nn.Parameter(torch.zeros(num_gaussians, 3, device=device)),
            )

        # pre-sigmoid, initialized for 30% post-activation
        opacity_init = torch.log(torch.tensor(0.3 / 0.7))
        self.register_parameter(
            "_opacity",
            nn.Parameter(torch.full((num_gaussians, 1), opacity_init.item(), device=device)),
        )

        # Sequential assignment: cycle through faces so each gets at least
        # floor(num_gaussians / num_faces) Gaussians, then randomly assign
        # the remainder.
        full_cycles = num_gaussians // num_faces
        remainder = num_gaussians % num_faces
        parts = []
        if full_cycles > 0:
            parts.append(torch.arange(num_faces, device=device).repeat(full_cycles))
        if remainder > 0:
            perm = torch.randperm(num_faces, device=device)[:remainder]
            parts.append(perm)
        triangle_idx = torch.cat(parts) if len(parts) > 1 else parts[0]
        self.register_buffer("_triangle_idx", triangle_idx)

    @property
    def num_gaussians(self) -> int:
        """Return the total number of Gaussians."""
        return self._num_gaussians

    @property
    def local_positions(self) -> Tensor:
        """Return effective local positions (N, 3).

        When ``max_local_offset`` is set, applies tanh bounding so positions
        are in ``[-max_local_offset, max_local_offset]`` per axis.  Otherwise
        returns the raw parameters.
        """
        if self.max_local_offset is not None:
            return self.max_local_offset * torch.tanh(self._local_positions)
        return self._local_positions

    @property
    def local_rotations(self) -> Tensor:
        """Return local rotation parameters (N, 4) in wxyz format."""
        return self._local_rotations

    @property
    def local_scales(self) -> Tensor:
        """Return local log-scale parameters (N, 3)."""
        return self._local_scales

    @property
    def base_colors(self) -> Tensor:
        """Return base color parameters (N, 3) pre-sigmoid.

        Raises:
            AttributeError: If model is in SH mode (use sh_coefficients instead)
        """
        return self._base_colors

    @property
    def sh_coefficients(self) -> Tensor:
        """Return SH coefficient parameters (N, K, 3).

        Raises:
            AttributeError: If model is in RGB mode (use base_colors instead)
        """
        return self._sh_coefficients

    @property
    def raw_opacities(self) -> Tensor:
        """Return raw opacity parameters (N, 1) pre-sigmoid."""
        return self._opacity

    @property
    def triangle_indices(self) -> Tensor:
        """Return triangle assignment indices (N,)."""
        return self._triangle_idx

    def _gather_per_gaussian(
        self, data: Tensor, is_batched: bool
    ) -> Tensor:
        """Gather data for each Gaussian using triangle indices.

        Args:
            data: Per-triangle data (F, ...) or (B, F, ...)
            is_batched: Whether data has batch dimension

        Returns:
            Per-Gaussian data (N, ...) or (B, N, ...), agnostic to location, 
            rotation, or scale
        """
        if is_batched:
            B = data.shape[0]
            idx = self._triangle_idx.unsqueeze(0).expand(B, -1)
            batch_idx = torch.arange(B, device=data.device)[:, None]
            return data[batch_idx, idx]
        return data[self._triangle_idx]

    def _extract_rotation_translation(
        self, transforms: Tensor, is_batched: bool
    ) -> tuple[Tensor, Tensor]:
        """Extract rotation and translation from gathered transforms.

        Args:
            transforms: Homogeneous transforms (N, 4, 4) or (B, N, 4, 4)
            is_batched: Whether transforms are batched

        Returns:
            Tuple of (rotation (N, 3, 3) or (B, N, 3, 3),
                     translation (N, 3) or (B, N, 3))
        """
        R = transforms[..., :3, :3]
        t = transforms[..., :3, 3]
        return R, t

    def _broadcast_to_batch(self, tensor: Tensor, batch_size: int) -> Tensor:
        """Broadcast a tensor to include batch dimension.

        Args:
            tensor: Tensor without batch dim (N, ...)
            batch_size: Target batch size

        Returns:
            Tensor with batch dim (B, N, ...)
        """
        return tensor.unsqueeze(0).expand(batch_size, *[-1] * tensor.dim())

    def _collect_optimizer_states(
        self, optimizer: torch.optim.Optimizer | None, param_names: list[str]
    ) -> dict[str, tuple[nn.Parameter, dict[str, Tensor]]]:
        """Collect old parameters and their optimizer states.

        Args:
            optimizer: Optional optimizer to collect state from
            param_names: List of internal parameter names to collect

        Returns:
            Dict mapping param name to (old_param, state_dict) tuples
        """
        result = {}
        for name in param_names:
            old_param = getattr(self, name)
            old_state = {}
            if optimizer is not None and old_param in optimizer.state:
                for key, val in optimizer.state[old_param].items():
                    if isinstance(val, Tensor) and val.dim() > 0 and val.shape[0] == self._num_gaussians:
                        old_state[key] = val
                    elif isinstance(val, (int, float)) or (isinstance(val, Tensor) and val.dim() == 0):
                        old_state[key] = val
            result[name] = (old_param, old_state)
        return result

    def _update_optimizer_for_param(
        self,
        optimizer: torch.optim.Optimizer,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        new_state: dict[str, Tensor],
    ) -> None:
        """Update optimizer to reference new parameter with new state.

        Args:
            optimizer: Optimizer to update
            old_param: Old parameter to remove from optimizer
            new_param: New parameter to add to optimizer
            new_state: New state dict for the parameter
        """
        if old_param in optimizer.state:
            del optimizer.state[old_param]

        if new_state:
            optimizer.state[new_param] = new_state

        for group in optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_param:
                    group["params"][i] = new_param
                    break

    def resize(self, new_num_gaussians: int) -> None:
        """Resize all parameters and buffers to a new Gaussian count.

        Reallocates all parameter tensors and the triangle index buffer to
        match the target size. Values are uninitialized (zeros) since this
        is intended to be called before load_state_dict overwrites them.

        Args:
            new_num_gaussians: Target number of Gaussians
        """
        if new_num_gaussians == self._num_gaussians:
            return

        device = self._local_positions.device
        dtype = self._local_positions.dtype

        for name in self._PARAM_NAMES:
            old_param = getattr(self, name)
            shape = (new_num_gaussians,) + old_param.shape[1:]
            delattr(self, name)
            self.register_parameter(name, nn.Parameter(torch.zeros(shape, device=device, dtype=dtype)))

        delattr(self, "_triangle_idx")
        self.register_buffer(
            "_triangle_idx",
            torch.zeros(new_num_gaussians, device=device, dtype=torch.long),
        )
        self._num_gaussians = new_num_gaussians

    def get_colors(self) -> Tensor:
        """Return sigmoid-activated colors.

        Returns:
            Colors (N, 3) in range [0, 1]

        Raises:
            RuntimeError: If model is in SH mode (colors come from gsplat SH eval)
        """
        if self.use_sh:
            raise RuntimeError(
                "get_colors() is not available in SH mode. "
                "Colors are computed by gsplat from SH coefficients."
            )
        return torch.sigmoid(self._base_colors)

    def get_opacities(self) -> Tensor:
        """Return sigmoid-activated opacities.

        Returns:
            Opacities (N, 1) in range [0, 1]
        """
        return torch.sigmoid(self._opacity)

    def get_positions(self, triangle_transforms: Tensor) -> Tensor:
        """Transform local positions to global coordinates.

        Gathers the triangle transforms for each Gaussian using triangle_idx,
        extracts the rotation and translation, then applies:
        μ_global = R @ μ_local + t

        Args:
            triangle_transforms: Homogeneous transforms (F, 4, 4) or (B, F, 4, 4)

        Returns:
            Global positions (N, 3) or (B, N, 3)
        """
        is_batched = triangle_transforms.dim() == 4
        gathered = self._gather_per_gaussian(triangle_transforms, is_batched)
        R, t = self._extract_rotation_translation(gathered, is_batched)

        local_pos = self.local_positions
        if is_batched:
            B = triangle_transforms.shape[0]
            local_pos = self._broadcast_to_batch(local_pos, B)
            return torch.einsum("bnij,bnj->bni", R, local_pos) + t
        return torch.einsum("nij,nj->ni", R, local_pos) + t

    def get_rotations(self, triangle_transforms: Tensor) -> Tensor:
        """Transform local rotations to global coordinates.

        Composes local Gaussian rotations with triangle frame rotations.
        Algorithm:
        1. Normalize local quaternions
        2. Convert to rotation matrices
        3. Gather triangle rotation matrices using triangle_idx
        4. Compose: R_global = R_triangle @ R_local
        5. Convert back to quaternion

        Args:
            triangle_transforms: Homogeneous transforms (F, 4, 4) or (B, F, 4, 4)

        Returns:
            Global quaternions (N, 4) or (B, N, 4) in wxyz format, normalized
        """
        is_batched = triangle_transforms.dim() == 4
        gathered = self._gather_per_gaussian(triangle_transforms, is_batched)
        R_triangle, _ = self._extract_rotation_translation(gathered, is_batched)

        q_local_norm = normalize_quaternion(self._local_rotations)
        R_local = quaternion_to_matrix(q_local_norm)

        if is_batched:
            B = triangle_transforms.shape[0]
            R_local = self._broadcast_to_batch(R_local, B)
            R_global = torch.einsum("bnij,bnjk->bnik", R_triangle, R_local)
        else:
            R_global = torch.einsum("nij,njk->nik", R_triangle, R_local)

        return matrix_to_quaternion(R_global)

    def get_scales(self, triangle_stretches: Tensor) -> Tensor:
        """Transform local scales to global using triangle stretch factors.

        Applies per-axis stretch factors from triangle deformation to local
        Gaussian scales. The algorithm:
        1. Gather triangle stretches for each Gaussian using triangle_idx
        2. Convert log-scale to linear: exp(s_local)
        3. Apply element-wise stretch: s_global = stretch * s_linear

        Args:
            triangle_stretches: Per-axis stretch factors (F, 3) or (B, F, 3)

        Returns:
            Global scales (N, 3) or (B, N, 3) in linear space (not log)
        """
        is_batched = triangle_stretches.dim() == 3
        gathered_stretches = self._gather_per_gaussian(triangle_stretches, is_batched)
        s_linear = torch.exp(self._local_scales)

        if is_batched:
            B = triangle_stretches.shape[0]
            s_linear = self._broadcast_to_batch(s_linear, B)

        return gathered_stretches * s_linear

    def initialize_from_mesh(self, mesh) -> None:
        """Initialize Gaussian scales based on mesh geometry.

        Computes average edge length from the canonical mesh and sets scale
        values appropriately:
        - X and Y scales (tangent plane): log(0.5 * avg_edge)
        - Z scale (normal direction): log(0.1 * avg_edge)

        Other parameters (positions, rotations, colors, opacity) remain at
        their default values. Triangle assignments are not modified.

        Args:
            mesh: A mesh instance providing get_canonical_vertices() and
                  get_faces() methods
        """
        vertices = mesh.get_canonical_vertices()
        faces = mesh.get_faces()

        # Compute all triangle edge lengths
        v0 = vertices[faces[:, 0]]  # (F, 3)
        v1 = vertices[faces[:, 1]]  # (F, 3)
        v2 = vertices[faces[:, 2]]  # (F, 3)

        edge1_len = (v1 - v0).norm(dim=-1)  # (F,)
        edge2_len = (v2 - v1).norm(dim=-1)  # (F,)
        edge3_len = (v0 - v2).norm(dim=-1)  # (F,)

        # Average edge length across all edges of all triangles
        avg_edge = torch.cat([edge1_len, edge2_len, edge3_len]).mean()

        # Compute scale values
        s_xy = torch.log(0.5 * avg_edge)  # X and Y (tangent plane)
        s_z = torch.log(0.1 * avg_edge)   # Z (normal direction)

        # Update scales in-place while maintaining gradient tracking
        with torch.no_grad():
            self._local_scales[:, 0].fill_(s_xy.item())
            self._local_scales[:, 1].fill_(s_xy.item())
            self._local_scales[:, 2].fill_(s_z.item())

    def get_global_properties(
        self,
        triangle_transforms: Tensor,
        triangle_stretches: Tensor,
        position_offsets: Tensor | None = None,
        rotation_offsets: Tensor | None = None,
        scale_offsets: Tensor | None = None,
        color_offsets: Tensor | None = None,
        include_auxiliary: bool = False,
    ) -> dict[str, Tensor]:
        """Compute all Gaussian properties in global space for rendering.

        Transforms local Gaussian parameters to global coordinates using the
        provided triangle transforms and stretch factors. Optionally applies
        offset parameters (from a deformation MLP) in local space before
        transformation.

        Args:
            triangle_transforms: Homogeneous transforms (F, 4, 4) or (B, F, 4, 4)
            triangle_stretches: Per-axis stretch factors (F, 3) or (B, F, 3)
            position_offsets: Optional position offsets (N, 3) or (B, N, 3)
                to add to local positions before transformation
            rotation_offsets: Optional rotation offsets (N, 4) or (B, N, 4)
                as wxyz quaternions to compose with local rotations
            scale_offsets: Optional scale offsets (N, 3) or (B, N, 3)
                to add to log-scale values before exp() and stretch
            color_offsets: Optional color offsets (N, 3) or (B, N, 3)
                to add to base colors before sigmoid activation
            include_auxiliary: If True, include auxiliary outputs for debugging

        Returns:
            Dictionary with keys:
                - 'means': Global positions (N, 3) or (B, N, 3)
                - 'rotations': Global quaternions (N, 4) or (B, N, 4), wxyz, normalized
                - 'scales': Global scales (N, 3) or (B, N, 3), linear (not log)
                - 'colors': RGB values (N, 3) or (B, N, 3) in [0, 1]
                - 'opacities': Opacity values (N, 1) or (B, N, 1) in [0, 1]
            If include_auxiliary=True, also includes:
                - 'triangle_indices': Parent triangle indices (N,)
                - 'local_positions': Local position parameters (N, 3)
        """
        is_batched = triangle_transforms.dim() == 4

        if position_offsets is not None:
            # Apply offsets in local space, then transform
            means = self._transform_positions_with_offsets(
                triangle_transforms, position_offsets, is_batched
            )
        else:
            means = self.get_positions(triangle_transforms)

        if rotation_offsets is not None:
            rotations = self._transform_rotations_with_offsets(
                triangle_transforms, rotation_offsets, is_batched
            )
        else:
            rotations = self.get_rotations(triangle_transforms)

        if scale_offsets is not None:
            scales = self._transform_scales_with_offsets(
                triangle_stretches, scale_offsets, is_batched
            )
        else:
            scales = self.get_scales(triangle_stretches)

        if self.use_sh:
            # SH mode: return raw coefficients (N, K, 3); gsplat evaluates them
            colors = self._sh_coefficients
            if is_batched:
                B = triangle_transforms.shape[0]
                colors = colors.unsqueeze(0).expand(B, -1, -1, -1)
        elif color_offsets is not None:
            if is_batched:
                # Broadcast base_colors: (N, 3) -> (B, N, 3)
                B = triangle_transforms.shape[0]
                base = self._base_colors.unsqueeze(0).expand(B, -1, -1)
            else:
                base = self._base_colors
            colors = torch.sigmoid(base + color_offsets)
        else:
            colors = self.get_colors()
            if is_batched:
                # Broadcast to batch dimension
                B = triangle_transforms.shape[0]
                colors = colors.unsqueeze(0).expand(B, -1, -1)

        opacities = self.get_opacities()
        if is_batched:
            B = triangle_transforms.shape[0]
            opacities = opacities.unsqueeze(0).expand(B, -1, -1)

        result = {
            "means": means,
            "rotations": rotations,
            "scales": scales,
            "colors": colors,
            "opacities": opacities,
        }

        if include_auxiliary:
            result["triangle_indices"] = self._triangle_idx
            result["local_positions"] = self.local_positions

        return result

    def _transform_positions_with_offsets(
        self,
        triangle_transforms: Tensor,
        position_offsets: Tensor,
        is_batched: bool,
    ) -> Tensor:
        """Transform positions with offsets applied in local space.

        Args:
            triangle_transforms: (F, 4, 4) or (B, F, 4, 4)
            position_offsets: (N, 3) or (B, N, 3)
            is_batched: Whether inputs are batched

        Returns:
            Global positions (N, 3) or (B, N, 3)
        """
        gathered = self._gather_per_gaussian(triangle_transforms, is_batched)
        R, t = self._extract_rotation_translation(gathered, is_batched)

        local_pos = self.local_positions
        if is_batched:
            B = triangle_transforms.shape[0]
            local_pos = self._broadcast_to_batch(local_pos, B)
            local_pos_with_offset = local_pos + position_offsets
            return torch.einsum("bnij,bnj->bni", R, local_pos_with_offset) + t

        local_pos_with_offset = local_pos + position_offsets
        return torch.einsum("nij,nj->ni", R, local_pos_with_offset) + t

    def _transform_rotations_with_offsets(
        self,
        triangle_transforms: Tensor,
        rotation_offsets: Tensor,
        is_batched: bool,
    ) -> Tensor:
        """Transform rotations with offsets applied in local space.

        The offset is composed with local rotation: q_local_final = q_local * q_offset
        Then the result is transformed by the triangle rotation.

        Args:
            triangle_transforms: (F, 4, 4) or (B, F, 4, 4)
            rotation_offsets: (N, 4) or (B, N, 4) wxyz quaternions
            is_batched: Whether inputs are batched

        Returns:
            Global quaternions (N, 4) or (B, N, 4) in wxyz format, normalized
        """
        gathered = self._gather_per_gaussian(triangle_transforms, is_batched)
        R_triangle, _ = self._extract_rotation_translation(gathered, is_batched)

        q_local_norm = normalize_quaternion(self._local_rotations)

        if is_batched:
            B = triangle_transforms.shape[0]
            q_local = self._broadcast_to_batch(q_local_norm, B)
            q_local_final = normalize_quaternion(quaternion_multiply(q_local, rotation_offsets))
            R_local = quaternion_to_matrix(q_local_final)
            R_global = torch.einsum("bnij,bnjk->bnik", R_triangle, R_local)
        else:
            q_local_final = normalize_quaternion(quaternion_multiply(q_local_norm, rotation_offsets))
            R_local = quaternion_to_matrix(q_local_final)
            R_global = torch.einsum("nij,njk->nik", R_triangle, R_local)

        return matrix_to_quaternion(R_global)

    def _transform_scales_with_offsets(
        self,
        triangle_stretches: Tensor,
        scale_offsets: Tensor,
        is_batched: bool,
    ) -> Tensor:
        """Transform scales with offsets applied in local space.

        The offset is added to log-scale: s_local_final = s_local + s_offset
        Then exp() and stretch are applied.

        Args:
            triangle_stretches: (F, 3) or (B, F, 3)
            scale_offsets: (N, 3) or (B, N, 3)
            is_batched: Whether inputs are batched

        Returns:
            Global scales (N, 3) or (B, N, 3) in linear space
        """
        gathered_stretches = self._gather_per_gaussian(triangle_stretches, is_batched)

        if is_batched:
            B = triangle_stretches.shape[0]
            local_scales = self._broadcast_to_batch(self._local_scales, B)
            local_scales_with_offset = local_scales + scale_offsets
        else:
            local_scales_with_offset = self._local_scales + scale_offsets

        return gathered_stretches * torch.exp(local_scales_with_offset)

    def remove_gaussians(
        self,
        keep_mask: Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Remove Gaussians by rebuilding tensors to exclude pruned entries.

        This operation reduces memory usage by removing inactive Gaussians
        entirely rather than masking them. When an optimizer is provided,
        its internal state (momentum, variance) is also rebuilt to match
        the new tensor shapes.

        Args:
            keep_mask: Boolean tensor (N,) where True = keep, False = remove.
                Must have same length as current num_gaussians.
            optimizer: Optional optimizer whose state should be updated.
                If None, only parameter tensors are rebuilt.

        Raises:
            ValueError: If keep_mask shape doesn't match num_gaussians.
        """
        if keep_mask.shape[0] != self._num_gaussians:
            raise ValueError(
                f"keep_mask shape {keep_mask.shape[0]} doesn't match "
                f"num_gaussians {self._num_gaussians}"
            )

        keep_mask = keep_mask.to(self._local_positions.device)
        new_count = keep_mask.sum().item()

        old_params_and_states = self._collect_optimizer_states(optimizer, self._PARAM_NAMES)

        for name in self._PARAM_NAMES:
            old_param, old_state = old_params_and_states[name]
            new_data = old_param.data[keep_mask].contiguous()
            new_param = nn.Parameter(new_data)

            delattr(self, name)
            self.register_parameter(name, new_param)

            if optimizer is not None and old_state:
                new_state = {}
                for key, val in old_state.items():
                    if isinstance(val, Tensor) and val.dim() > 0:
                        new_state[key] = val[keep_mask].contiguous()
                    else:
                        new_state[key] = val
                self._update_optimizer_for_param(optimizer, old_param, new_param, new_state)

        new_triangle_idx = self._triangle_idx[keep_mask].contiguous()
        delattr(self, "_triangle_idx")
        self.register_buffer("_triangle_idx", new_triangle_idx)
        self._num_gaussians = new_count

    def add_gaussians(
        self,
        new_params: dict[str, Tensor],
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Add new Gaussians by extending tensors and optimizer state.

        This operation increases the number of Gaussians by concatenating new
        parameter values to existing tensors. When an optimizer is provided,
        its internal state is extended with zeros for the new entries.

        Args:
            new_params: Dictionary with tensors for each parameter type.
                Required keys:
                - 'local_positions': (M, 3) position offsets
                - 'local_rotations': (M, 4) wxyz quaternions
                - 'local_scales': (M, 3) log-scale values
                - 'base_colors': (M, 3) pre-sigmoid RGB
                - 'opacity': (M, 1) pre-sigmoid opacity
                - 'triangle_idx': (M,) parent triangle indices
                Where M is the number of new Gaussians.
            optimizer: Optional optimizer whose state should be extended.
                If None, only parameter tensors are extended.

        Raises:
            ValueError: If required keys are missing from new_params.
            ValueError: If parameter shapes are inconsistent.
            ValueError: If triangle indices are out of valid range.
        """
        required_keys = list(self._PARAM_KEY_MAP.keys()) + ["triangle_idx"]

        for key in required_keys:
            if key not in new_params:
                raise ValueError(f"Missing required key in new_params: {key}")

        num_new = new_params["local_positions"].shape[0]
        if num_new == 0:
            return

        if self.use_sh:
            K = (self.max_sh_degree + 1) ** 2
            expected_shapes = {
                "local_positions": (num_new, 3),
                "local_rotations": (num_new, 4),
                "local_scales": (num_new, 3),
                "sh_coefficients": (num_new, K, 3),
                "opacity": (num_new, 1),
                "triangle_idx": (num_new,),
            }
        else:
            expected_shapes = {
                "local_positions": (num_new, 3),
                "local_rotations": (num_new, 4),
                "local_scales": (num_new, 3),
                "base_colors": (num_new, 3),
                "opacity": (num_new, 1),
                "triangle_idx": (num_new,),
            }

        for key, expected_shape in expected_shapes.items():
            if new_params[key].shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {key}: expected {expected_shape}, "
                    f"got {new_params[key].shape}"
                )

        tri_idx = new_params["triangle_idx"]
        if (tri_idx < 0).any() or (tri_idx >= self._num_faces).any():
            raise ValueError(
                f"Triangle indices must be in range [0, {self._num_faces}), "
                f"got min={tri_idx.min().item()}, max={tri_idx.max().item()}"
            )

        device = self._local_positions.device
        old_params_and_states = self._collect_optimizer_states(optimizer, self._PARAM_NAMES)

        for new_key, internal_name in self._PARAM_KEY_MAP.items():
            old_param, old_state = old_params_and_states[internal_name]
            new_values = new_params[new_key].to(device)
            new_data = torch.cat([old_param.data, new_values], dim=0).contiguous()
            new_param = nn.Parameter(new_data)

            delattr(self, internal_name)
            self.register_parameter(internal_name, new_param)

            if optimizer is not None and old_state:
                new_state = {}
                for state_key, val in old_state.items():
                    if isinstance(val, Tensor) and val.dim() > 0:
                        zeros_shape = (num_new,) + val.shape[1:]
                        zeros = torch.zeros(zeros_shape, dtype=val.dtype, device=val.device)
                        new_state[state_key] = torch.cat([val, zeros], dim=0).contiguous()
                    else:
                        new_state[state_key] = val
                self._update_optimizer_for_param(optimizer, old_param, new_param, new_state)

        new_tri_idx = new_params["triangle_idx"].to(device)
        extended_tri_idx = torch.cat([self._triangle_idx, new_tri_idx], dim=0).contiguous()
        delattr(self, "_triangle_idx")
        self.register_buffer("_triangle_idx", extended_tri_idx)
        self._num_gaussians = self._num_gaussians + num_new
