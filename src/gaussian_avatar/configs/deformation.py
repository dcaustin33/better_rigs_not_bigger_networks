"""Configuration dataclasses for deformation pipeline and avatar model.

This module provides:
- DeformationConfig: MLP architecture and positional encoding settings
- AvatarConfig: Complete avatar model configuration including mesh and Gaussians
"""

from dataclasses import dataclass, field

import torch


@dataclass
class DeformationConfig:
    """Configuration for the deformation MLP.

    Defines the architecture and encoding settings for the neural network
    that predicts pose-dependent property offsets.

    Args:
        num_frequencies: Number of positional encoding frequency bands (L).
            Output dimension is 3 + 6*L per position.
        hidden_dim: Hidden layer dimension in the MLP
        num_layers: Number of hidden layers in the MLP
        activation: Activation function ("silu", "relu", "gelu")
        output_init_std: Standard deviation for final layer weight initialization.
            Small values ensure near-zero initial outputs.
        dropout: Dropout probability applied after each activation in the MLP.
            0.0 disables dropout (default).
        anneal_frequencies: Enable Hann window annealing for positional encoding.
            Progressively enables higher-frequency bands during training for
            coarse-to-fine optimization (from Nerfies, Park et al. 2021).
        anneal_start_iter: Training iteration when frequency annealing begins.
        anneal_end_iter: Training iteration when all frequency bands are fully active.
        include_canonical: Include positional-encoded canonical positions in MLP input.
            Enables backward compatibility with models trained with canonical positions.
        include_betas: Include shape parameters in MLP input.
            Enables backward compatibility with models trained with betas.
        num_betas: Number of shape coefficients (10 for SMPL/SMPL-X, 45 for MHR).
            Only used when include_betas=True.
        pose_joint_indices: Optional list of joint indices to use for MLP input.
            When set, only the specified joints' 6D rotations are included
            in the MLP input, reducing dimensionality. Useful for MHR where
            127 joints would create a very large input (762 dims). When None,
            all joints are used.
    """

    num_frequencies: int = 6
    hidden_dim: int = 256
    num_layers: int = 3
    activation: str = "silu"
    output_init_std: float = 0.001
    dropout: float = 0.0
    anneal_frequencies: bool = False
    anneal_start_iter: int = 0
    anneal_end_iter: int = 10000
    include_canonical: bool = False
    include_betas: bool = False
    num_betas: int = 10
    pose_joint_indices: list[int] | None = None

    @property
    def effective_num_joints(self) -> int | None:
        """Number of joints used for MLP input when pose_joint_indices is set.

        Returns:
            Length of pose_joint_indices if set, None otherwise
        """
        if self.pose_joint_indices is not None:
            return len(self.pose_joint_indices)
        return None

    def get_frequency_alpha(self, iteration: int) -> float | None:
        """Compute alpha for Hann window frequency annealing.

        Alpha controls how many frequency bands are active. Band j is fully
        active when alpha > j+1, inactive when alpha < j, and smoothly
        transitions via a cosine window in between.

        Args:
            iteration: Current training iteration

        Returns:
            Alpha value in [0, num_frequencies], or None if annealing is disabled
        """
        if not self.anneal_frequencies:
            return None
        if self.anneal_end_iter <= self.anneal_start_iter:
            return float(self.num_frequencies)
        t = max(0, iteration - self.anneal_start_iter)
        duration = self.anneal_end_iter - self.anneal_start_iter
        alpha = self.num_frequencies * t / duration
        return min(alpha, float(self.num_frequencies))

    @property
    def input_dim(self) -> int:
        """Compute input dimension for SMPL (24 joints).

        Convenience property for backward compatibility. For other model
        types, use ``input_dim_for_joints()`` with the appropriate joint count.

        Formula:
            PE_dim = 3 + 6 * num_frequencies (39 for L=6)
            input_dim = PE_dim + 24 * 6
                      = 39 + 144 = 183 for L=6

        Returns:
            Total input dimension for the MLP
        """
        return self.input_dim_for_joints(num_joints=24)

    def input_dim_for_joints(self, num_joints: int) -> int:
        """Compute MLP input dimension for given joint count.

        The MLP input concatenates:
        - PE_dim: Positional-encoded posed positions.
          Each 3D position becomes 3 + 6*L dims (original coords + sin/cos
          at L frequency bands × 3 dims × 2 trig functions).
        - num_joints × 6: 6D rotation representation per joint (first two
          columns of rotation matrix, continuous unlike axis-angle).
        - (optional) PE_dim for canonical positions if include_canonical=True
        - (optional) num_betas if include_betas=True

        Args:
            num_joints: Number of pose joints (24 for SMPL, 55 for SMPL-X,
                127 for MHR, or a subset count when pose_joint_indices is used)

        Returns:
            Total input dimension for the MLP
        """
        pe_dim = 3 + 6 * self.num_frequencies
        dim = pe_dim + num_joints * 6
        if self.include_canonical:
            dim += pe_dim
        if self.include_betas:
            dim += self.num_betas
        return dim


@dataclass
class AvatarConfig:
    """Configuration for the complete Gaussian avatar model.

    Combines mesh, Gaussian, and deformation settings for the full pipeline.

    Args:
        model_type: Mesh model type ("smpl", "smplx", or "mhr")
        model_path: Path to the mesh model files. Convention:
            ``mesh_models/smpl`` for SMPL, ``mesh_models/models/smplx`` for
            SMPL-X, ``mesh_models/mhr`` for MHR.
        gender: Model gender ("male", "female", "neutral"). Not used by MHR
            (MHR uses shape parameters instead).
        num_gaussians: Total number of Gaussians, randomly assigned to triangles
        lod: Level-of-detail for MHR mesh (0-6). LOD 1 is recommended for
            training. Ignored for SMPL/SMPL-X.
        deformation: DeformationConfig for the MLP (uses defaults if None)
        device: Compute device ("cuda" or "cpu")
    """

    model_type: str = "smpl"
    model_path: str = "mesh_models/smpl"
    gender: str = "male"
    num_gaussians: int = 100000
    lod: int = 1
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
