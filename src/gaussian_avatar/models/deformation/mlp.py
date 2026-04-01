"""Deformation MLP for predicting pose-dependent Gaussian property offsets.

This module provides:
- DeformationMLP: Neural network predicting position, rotation, scale, and color offsets
- construct_quaternion_offset: Converts 3D output to normalized wxyz quaternion
"""

import torch
from torch import Tensor, nn

from gaussian_avatar.models.deformation.encoding import positional_encoding
from gaussian_avatar.models.gaussian.rotations import normalize_quaternion


def construct_quaternion_offset(delta_xyz: Tensor) -> Tensor:
    """Construct normalized quaternion from xyz components.

    Creates a quaternion with w=1 and xyz from input, then normalizes.
    When delta_xyz ≈ 0, this produces identity rotation [1, 0, 0, 0].
    Small values produce small rotations.

    Args:
        delta_xyz: Rotation offset xyz components (N, 3)

    Returns:
        Normalized quaternion in wxyz format (N, 4)
    """
    N = delta_xyz.shape[0]
    w = torch.ones(N, 1, dtype=delta_xyz.dtype, device=delta_xyz.device)
    q = torch.cat([w, delta_xyz], dim=-1)
    return normalize_quaternion(q)


class DeformationMLP(nn.Module):
    """MLP that predicts pose-dependent offsets for Gaussian properties.

    Takes encoded posed position and pose as input.
    Outputs local-space offsets for position, rotation, scale, and color.

    The network architecture is:
        Input → Linear(input_dim, hidden_dim) → SiLU
        → [Linear(hidden_dim, hidden_dim) → SiLU] × num_layers
        → Linear(hidden_dim, 12)

    Output structure (12 dims when use_sh=False, 9 dims when use_sh=True):
        - δμ (3): Position offset in local triangle space
        - δq_xyz (3): Rotation offset xyz (combined with w=1 to form quaternion)
        - δs (3): Scale offset (additive in log-space)
        - δrgb (3): Color offset (additive in pre-sigmoid space) [only when use_sh=False]
    """

    def __init__(
        self,
        num_frequencies: int = 6,
        num_joints: int = 24,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "silu",
        output_init_std: float = 0.001,
        dropout: float = 0.0,
        include_canonical: bool = False,
        include_betas: bool = False,
        num_betas: int = 10,
        use_sh: bool = False,
    ):
        """Initialize the deformation MLP.

        Args:
            num_frequencies: Positional encoding frequency bands (L)
            num_joints: Number of pose joints (24 for SMPL, 55 for SMPL-X)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function ("silu", "relu", "gelu")
            output_init_std: Standard deviation for final layer weight initialization
            dropout: Dropout probability applied after each activation (0.0 = no dropout)
            include_canonical: Include positional-encoded canonical positions in input
            include_betas: Include SMPL shape betas in input
            num_betas: Number of shape coefficients (only used when include_betas=True)
        """
        super().__init__()

        self.num_frequencies = num_frequencies
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.include_canonical = include_canonical
        self.include_betas = include_betas
        self.num_betas = num_betas
        self.use_sh = use_sh

        activation_fns = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activation_fns:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose from: {list(activation_fns.keys())}"
            )
        self.activation = activation_fns[activation]

        pe_dim = 3 + 6 * num_frequencies
        input_dim = pe_dim + num_joints * 6
        if include_canonical:
            input_dim += pe_dim
        if include_betas:
            input_dim += num_betas

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)

        # Separate output layer for small-weight initialization (near-zero initial outputs)
        # SH mode: 9 dims (no color offsets); RGB mode: 12 dims
        output_dim = 9 if use_sh else 12
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=output_init_std)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        posed_positions: Tensor,
        pose_6d: Tensor,
        alpha: float | None = None,
        canonical_positions: Tensor | None = None,
        betas: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute pose-dependent property offsets.

        Args:
            posed_positions: Gaussian positions after mesh-based transform (N, 3)
            pose_6d: Pose in 6D rotation representation (J, 6) or (J*6,)
            alpha: Hann window position for frequency annealing. None disables.
            canonical_positions: Canonical positions (N, 3), required when include_canonical=True
            betas: Shape coefficients (num_betas,), required when include_betas=True

        Returns:
            dict with keys:
                - 'position_offsets': (N, 3) δμ in local space
                - 'rotation_offsets': (N, 4) δq quaternion (wxyz, normalized)
                - 'scale_offsets': (N, 3) δs additive to log-scale
                - 'color_offsets': (N, 3) δrgb additive pre-sigmoid
        """
        N = posed_positions.shape[0]
        dtype = posed_positions.dtype

        if pose_6d.dim() == 1:
            pose_6d = pose_6d.reshape(-1, 6)

        pe_posed = positional_encoding(posed_positions, self.num_frequencies, alpha=alpha)
        pose_broadcast = pose_6d.flatten().unsqueeze(0).expand(N, -1).to(dtype=dtype)

        # Ordering must match training: [pe_canonical, pe_posed, pose, betas]
        parts = []
        if self.include_canonical and canonical_positions is not None:
            pe_canonical = positional_encoding(canonical_positions, self.num_frequencies, alpha=alpha)
            parts.append(pe_canonical)
        parts.append(pe_posed)
        parts.append(pose_broadcast)
        if self.include_betas and betas is not None:
            betas_broadcast = betas.flatten().unsqueeze(0).expand(N, -1).to(dtype=dtype)
            parts.append(betas_broadcast)

        x = torch.cat(parts, dim=-1)
        hidden = self.layers(x)
        output = self.output_layer(hidden)

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
