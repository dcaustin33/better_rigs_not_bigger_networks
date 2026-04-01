"""AIAP (As-Isometric-As-Possible) losses for Gaussian properties.

These losses encourage Gaussians to maintain consistent relative distances
and shapes across different poses, improving generalization to novel poses.
"""

import torch
import torch.nn as nn

from gaussian_avatar.models.gaussian.rotations import quaternion_to_matrix


class NeighborCache:
    """Cache for k-nearest neighbor indices.

    Computes neighbors once from canonical positions and reuses them for
    efficient AIAP loss computation. Supports CUDA KNN acceleration with
    automatic PyTorch fallback.

    Args:
        k: Number of neighbors per Gaussian
        use_cuda_knn: Whether to use KNN_CUDA if available
    """

    def __init__(self, k: int = 5, use_cuda_knn: bool = True):
        self.k = k
        self.use_cuda_knn = use_cuda_knn
        self._neighbor_indices: torch.Tensor | None = None
        self._cuda_knn_available: bool | None = None

    def _check_cuda_knn(self) -> bool:
        """Check if CUDA KNN is available (cached)."""
        if self._cuda_knn_available is None:
            try:
                from knn_cuda import KNN  # noqa: F401

                self._cuda_knn_available = True
            except ImportError:
                self._cuda_knn_available = False
        return self._cuda_knn_available

    # KNN_CUDA has a 32-bit int overflow in its kernel when N > ~46340
    # (the index computation l*width overflows INT_MAX). Use chunked
    # torch.cdist as fallback for large point counts.
    _KNN_CUDA_MAX_POINTS = 46000

    def compute_neighbors(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute k-nearest neighbors for each Gaussian.

        Args:
            positions: (N, 3) canonical Gaussian positions

        Returns:
            (N, k) tensor of neighbor indices
        """
        N = positions.shape[0]

        if (
            self.use_cuda_knn
            and self._check_cuda_knn()
            and positions.is_cuda
            and N <= self._KNN_CUDA_MAX_POINTS
        ):
            from knn_cuda import KNN

            knn = KNN(k=self.k + 1, transpose_mode=False)

            # KNN raw kernel expects (D, N) format; transpose_mode=False passes through as-is
            pos_t = positions.t().contiguous().unsqueeze(0)  # (1, 3, N)
            _, indices = knn(pos_t, pos_t)  # (1, k+1, N)

            # Remove self from neighbors (first neighbor is self)
            # Output is (1, k+1, N), transpose to (N, k+1) then slice
            self._neighbor_indices = indices[0, 1:, :].t().contiguous()  # (N, k)
        else:
            # Chunked torch.cdist fallback to avoid N*N memory allocation
            self._neighbor_indices = self._chunked_knn(positions)

        return self._neighbor_indices

    def _chunked_knn(
        self, positions: torch.Tensor, chunk_size: int = 4096
    ) -> torch.Tensor:
        """Compute KNN using chunked torch.cdist to avoid N*N memory usage.

        Args:
            positions: (N, 3) positions
            chunk_size: number of query points per chunk

        Returns:
            (N, k) neighbor indices
        """
        N = positions.shape[0]
        k1 = self.k + 1  # include self, remove later
        all_indices = torch.empty(N, k1, dtype=torch.long, device=positions.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dists = torch.cdist(positions[start:end], positions)  # (chunk, N)
            _, topk_idx = dists.topk(k1, dim=1, largest=False)
            all_indices[start:end] = topk_idx

        return all_indices[:, 1:].contiguous()  # (N, k)

    @property
    def neighbor_indices(self) -> torch.Tensor:
        """Get cached neighbor indices.

        Returns:
            (N, k) tensor of neighbor indices

        Raises:
            RuntimeError: If neighbors haven't been computed yet
        """
        if self._neighbor_indices is None:
            raise RuntimeError("Neighbors not computed. Call compute_neighbors first.")
        return self._neighbor_indices

    def get_neighbor_pairs(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get values and their neighbor values for pairwise comparisons.

        Args:
            values: (N, D) values for each Gaussian

        Returns:
            Tuple of (values_expanded, neighbor_values) both (N, k, D)
        """
        if self._neighbor_indices is None:
            raise RuntimeError("Neighbors not computed. Call compute_neighbors first.")

        # Expand values: (N, D) -> (N, k, D)
        values_expanded = values.unsqueeze(1).expand(-1, self.k, -1)

        # Direct indexing: (N, D) indexed by (N, k) -> (N, k, D)
        neighbor_values = values[self._neighbor_indices]

        return values_expanded, neighbor_values

    @property
    def is_initialized(self) -> bool:
        """Check if neighbors have been computed."""
        return self._neighbor_indices is not None


class AIAPPositionLoss(nn.Module):
    """AIAP loss for Gaussian positions.

    Encourages consistent distances between Gaussians and their neighbors
    across different poses, promoting pose-invariant local geometry.

    Args:
        neighbor_cache: Precomputed neighbor indices
    """

    def __init__(self, neighbor_cache: NeighborCache):
        super().__init__()
        self.neighbor_cache = neighbor_cache

    def forward(
        self,
        canonical_positions: torch.Tensor,
        posed_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AIAP position loss.

        Args:
            canonical_positions: (N, 3) positions in canonical pose
            posed_positions: (N, 3) positions in current pose

        Returns:
            Scalar loss value
        """
        can_exp, can_neighbors = self.neighbor_cache.get_neighbor_pairs(
            canonical_positions
        )
        can_distances = torch.norm(can_exp - can_neighbors, dim=-1)

        posed_exp, posed_neighbors = self.neighbor_cache.get_neighbor_pairs(
            posed_positions
        )
        posed_distances = torch.norm(posed_exp - posed_neighbors, dim=-1)

        return torch.abs(can_distances - posed_distances).mean()


class AIAPCovarianceLoss(nn.Module):
    """AIAP loss for Gaussian covariances.

    Encourages consistent covariance differences between Gaussians and their
    neighbors across poses, promoting pose-invariant Gaussian shapes.

    Args:
        neighbor_cache: Precomputed neighbor indices
    """

    def __init__(self, neighbor_cache: NeighborCache):
        super().__init__()
        self.neighbor_cache = neighbor_cache

    def _compute_covariance(
        self,
        rotations: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Compute covariance matrices from rotation and scale.

        Args:
            rotations: (N, 4) quaternions (wxyz)
            scales: (N, 3) linear scales

        Returns:
            (N, 3, 3) covariance matrices
        """
        R = quaternion_to_matrix(rotations)
        S = torch.diag_embed(scales)
        S_squared = S @ S
        cov = R @ S_squared @ R.transpose(-1, -2)
        return cov

    def forward(
        self,
        canonical_rotations: torch.Tensor,
        canonical_scales: torch.Tensor,
        posed_rotations: torch.Tensor,
        posed_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AIAP covariance loss.

        Args:
            canonical_rotations: (N, 4) quaternions in canonical pose
            canonical_scales: (N, 3) scales in canonical pose
            posed_rotations: (N, 4) quaternions in current pose
            posed_scales: (N, 3) scales in current pose

        Returns:
            Scalar loss value
        """
        can_cov = self._compute_covariance(canonical_rotations, canonical_scales)
        posed_cov = self._compute_covariance(posed_rotations, posed_scales)

        can_cov_flat = can_cov.view(-1, 9)
        posed_cov_flat = posed_cov.view(-1, 9)

        can_exp, can_neighbors = self.neighbor_cache.get_neighbor_pairs(can_cov_flat)
        can_diff_norm = torch.norm(can_exp - can_neighbors, dim=-1)

        posed_exp, posed_neighbors = self.neighbor_cache.get_neighbor_pairs(
            posed_cov_flat
        )
        posed_diff_norm = torch.norm(posed_exp - posed_neighbors, dim=-1)

        return torch.abs(can_diff_norm - posed_diff_norm).mean()


class AIAPLoss(nn.Module):
    """Combined AIAP loss for position and covariance isometry.

    Encourages Gaussians to maintain consistent relative distances and shapes
    across different poses, improving generalization to novel poses.

    Args:
        k_neighbors: Number of neighbors per Gaussian
        lambda_position: Weight for position AIAP
        lambda_covariance: Weight for covariance AIAP
        use_cuda_knn: Whether to use CUDA KNN if available
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        lambda_position: float = 0.01,
        lambda_covariance: float = 0.01,
        use_cuda_knn: bool = True,
    ):
        super().__init__()
        self.lambda_position = lambda_position
        self.lambda_covariance = lambda_covariance

        self.neighbor_cache = NeighborCache(k=k_neighbors, use_cuda_knn=use_cuda_knn)
        self.position_loss = AIAPPositionLoss(self.neighbor_cache)
        self.covariance_loss = AIAPCovarianceLoss(self.neighbor_cache)

        self._initialized = False
        self._initialized_count = 0

    def initialize(self, canonical_positions: torch.Tensor):
        """Initialize neighbor cache with canonical positions.

        Must be called once before forward(), and again after densification
        changes the Gaussian count.

        Args:
            canonical_positions: (N, 3) Gaussian positions in canonical pose
        """
        self.neighbor_cache.compute_neighbors(canonical_positions)
        self._initialized = True
        self._initialized_count = canonical_positions.shape[0]

    def forward(
        self,
        canonical_positions: torch.Tensor,
        canonical_rotations: torch.Tensor,
        canonical_scales: torch.Tensor,
        posed_positions: torch.Tensor,
        posed_rotations: torch.Tensor,
        posed_scales: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute AIAP losses.

        Args:
            canonical_positions: (N, 3) positions in canonical pose
            canonical_rotations: (N, 4) quaternions in canonical pose
            canonical_scales: (N, 3) scales in canonical pose
            posed_positions: (N, 3) positions in current pose
            posed_rotations: (N, 4) quaternions in current pose
            posed_scales: (N, 3) scales in current pose

        Returns:
            Dictionary with 'aiap_position', 'aiap_covariance', and 'aiap' (total)
        """
        if not self._initialized or canonical_positions.shape[0] != self._initialized_count:
            self.initialize(canonical_positions)

        losses = {}

        losses["aiap_position"] = self.position_loss(
            canonical_positions, posed_positions
        )
        losses["aiap_covariance"] = self.covariance_loss(
            canonical_rotations,
            canonical_scales,
            posed_rotations,
            posed_scales,
        )

        losses["aiap"] = (
            self.lambda_position * losses["aiap_position"]
            + self.lambda_covariance * losses["aiap_covariance"]
        )

        return losses
