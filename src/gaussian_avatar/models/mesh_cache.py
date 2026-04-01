"""Pre-computed mesh transforms cache for training acceleration.

When per-frame parameter optimization is disabled (the default), the body model
forward pass and triangle transform computations are deterministic for a given
(subject, frame_idx). This module pre-computes and caches these results so
the training loop can skip the expensive mesh forward pass on every iteration.
"""

import logging
from typing import NamedTuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class MeshCacheEntry(NamedTuple):
    """Cached mesh outputs for a single frame.

    Args:
        posed_transforms: Per-triangle homogeneous transforms (F, 4, 4)
        stretches: Per-triangle axis stretches (F, 3)
        pose_6d: 6D rotation encoding for MLP input (J, 6)
    """

    posed_transforms: Tensor
    stretches: Tensor
    pose_6d: Tensor


class MeshCache:
    """Pre-computed mesh transforms for all dataset frames.

    Eliminates redundant GPU work by running the body model forward pass
    once per frame before training starts. The cache is keyed by
    (subject, frame_idx) and stores the three expensive outputs that
    the avatar's forward pass would otherwise recompute every iteration.

    Memory usage scales with num_frames * num_faces:
    - SMPL (~13K faces, 500 frames): ~530 MB
    - MHR LOD 1 (~36K faces, 500 frames): ~1.4 GB
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], MeshCacheEntry] = {}

    def __len__(self) -> int:
        return len(self._cache)

    @torch.no_grad()
    def populate(
        self,
        avatar: "GaussianAvatar",  # noqa: F821
        dataset: "Dataset",  # noqa: F821
        device: torch.device,
    ) -> None:
        """Run body model forward for every frame, store results.

        Iterates over the dataset's index mapping to get all (subject, frame_idx)
        pairs. For each unique pair, runs the mesh forward pass and triangle
        transform computations, storing the results for later lookup.

        Args:
            avatar: GaussianAvatar model (uses its mesh and pose encoding)
            dataset: Training dataset with _index_mapping and __getitem__
            device: Device to store cached tensors on
        """
        avatar._ensure_canonical_cache()

        # Collect unique (subject, frame_idx) pairs and their dataset indices.
        # For ZJU MoCap, _index_mapping has 3-tuples (subject, cam, frame_idx)
        # but train split uses one camera, so (subject, frame_idx) is unique.
        seen: dict[tuple[str, int], int] = {}
        for i, entry in enumerate(dataset._index_mapping):
            if len(entry) == 3:
                subject, _cam, frame_idx = entry
            else:
                subject, frame_idx = entry
            key = (subject, int(frame_idx))
            if key not in seen:
                seen[key] = i

        logger.info(
            "Populating mesh cache for %d unique frames...", len(seen)
        )

        for (subject, frame_idx), dataset_idx in seen.items():
            sample = dataset[dataset_idx]
            pose = sample["pose"].to(device)
            betas = sample["betas"].to(device)

            # Run mesh forward pass (the expensive part we're caching)
            posed_vertices = avatar.mesh.forward(pose, betas)
            posed_transforms = avatar.mesh.get_triangle_transforms(posed_vertices)
            stretches = avatar.mesh.get_triangle_stretches(
                canonical_vertices=avatar._canonical_vertices,
                posed_vertices=posed_vertices,
                canonical_frames=avatar.mesh.get_canonical_triangle_frames(),
                max_stretch=avatar.max_stretch,
            )
            pose_6d = avatar._encode_pose(pose)

            self._cache[(subject, frame_idx)] = MeshCacheEntry(
                posed_transforms=posed_transforms.detach(),
                stretches=stretches.detach(),
                pose_6d=pose_6d.detach(),
            )

        logger.info("Mesh cache populated: %d entries", len(self._cache))

    def get(self, subject: str, frame_idx: int) -> MeshCacheEntry | None:
        """Look up cached tensors for a frame.

        Args:
            subject: Subject identifier
            frame_idx: Frame index within subject

        Returns:
            MeshCacheEntry if found, None otherwise
        """
        return self._cache.get((subject, int(frame_idx)))
