"""Per-frame body model parameter optimization module.

This module provides PerFrameParameters, an nn.Module that stores learnable
offsets for per-frame pose, translation, and per-subject shape parameters.
The offsets are initialized to zero so that the model starts from the
dataset's body model fits and learns corrections during training. Works
with any body model (SMPL, SMPL-X, MHR) — dimensions are auto-detected
from the dataset.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from gaussian_avatar.configs.per_frame_params import PerFrameParamsConfig

logger = logging.getLogger(__name__)


class PerFrameParameters(nn.Module):
    """Learnable per-frame body model parameter offsets.

    Stores zero-initialized offsets for pose (per-frame), shape/betas
    (per-subject), and translation (per-frame). During training, these
    offsets are added to the dataset values before being passed to the
    avatar model. Dimensions are auto-detected from the dataset, so this
    works with any body model (SMPL 72-dim pose, MHR 204-dim pose, etc.).

    Uses nn.Embedding for per-frame parameters (pose, trans) and nn.Parameter
    for per-subject parameters (betas). Subject names are sanitized for use
    as nn.ParameterDict keys.

    Args:
        config: PerFrameParamsConfig controlling which parameters to optimize.
        dataset: A dataset with ``_index_mapping`` attribute containing
            ``(subject, frame_idx)`` or ``(subject, camera, frame_idx)`` tuples.
    """

    def __init__(
        self,
        config: PerFrameParamsConfig,
        dataset: Any,
    ) -> None:
        super().__init__()
        self.config = config

        # Discover all (subject, frame_idx) pairs from the dataset
        index_mapping: list[tuple[str, int]] = dataset._index_mapping

        # Build per-subject frame sets and find max frame index
        # index_mapping entries may be (subject, frame_idx) or
        # (subject, camera, frame_idx) depending on the dataset format
        subject_frames: dict[str, set[int]] = {}
        for entry in index_mapping:
            if len(entry) == 2:
                subject, frame_idx = entry
            elif len(entry) == 3:
                subject, _, frame_idx = entry
            else:
                raise ValueError(
                    f"Expected 2 or 3 element index_mapping entries, got {len(entry)}"
                )
            if subject not in subject_frames:
                subject_frames[subject] = set()
            subject_frames[subject].add(frame_idx)

        self._known_frames = subject_frames
        self._subject_names = sorted(subject_frames.keys())

        # Infer dimensions from dataset
        sample = dataset[0]
        pose_dim = sample["pose"].shape[0]  # 72 for SMPL
        betas_dim = sample["betas"].shape[0]  # typically 10

        # Sanitize subject names for use as ParameterDict keys
        # (nn.ParameterDict keys must be valid Python identifiers)
        self._key_to_subject: dict[str, str] = {}
        self._subject_to_key: dict[str, str] = {}
        for subject in self._subject_names:
            key = subject.replace("-", "_").replace(".", "_").replace(" ", "_")
            self._key_to_subject[key] = subject
            self._subject_to_key[subject] = key

        # Create per-frame pose offsets
        if config.optimize_pose:
            self.pose_offsets = nn.ParameterDict()
            for subject in self._subject_names:
                key = self._subject_to_key[subject]
                max_frame = max(subject_frames[subject])
                emb = nn.Embedding(max_frame + 1, pose_dim)
                nn.init.zeros_(emb.weight)
                self.pose_offsets[key] = emb.weight
            logger.info(
                "Created pose offsets for %d subjects (dim=%d)",
                len(self._subject_names),
                pose_dim,
            )

        # Create per-subject betas offsets
        if config.optimize_betas:
            self.betas_offsets = nn.ParameterDict()
            for subject in self._subject_names:
                key = self._subject_to_key[subject]
                self.betas_offsets[key] = nn.Parameter(
                    torch.zeros(betas_dim)
                )
            logger.info(
                "Created betas offsets for %d subjects (dim=%d)",
                len(self._subject_names),
                betas_dim,
            )

        # Create per-frame translation offsets
        if config.optimize_trans:
            self.trans_offsets = nn.ParameterDict()
            for subject in self._subject_names:
                key = self._subject_to_key[subject]
                max_frame = max(subject_frames[subject])
                emb = nn.Embedding(max_frame + 1, 3)
                nn.init.zeros_(emb.weight)
                self.trans_offsets[key] = emb.weight
            logger.info(
                "Created translation offsets for %d subjects (dim=3)",
                len(self._subject_names),
            )

    def apply_to_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Add learnable offsets to the batch's body model parameters.

        Only applies offsets for known (subject, frame_idx) pairs from the
        training set. Clones tensors to avoid mutating the original batch.

        Args:
            batch: Collated batch dict with keys 'pose', 'betas', 'trans',
                'subject' (list of str), and 'frame_idx' (list of int).

        Returns:
            New batch dict with offset-corrected pose/betas/trans tensors.
        """
        batch = {**batch}  # shallow copy the dict

        subjects = batch["subject"]
        frame_indices = batch["frame_idx"]
        B = len(subjects)

        if self.config.optimize_pose:
            pose = batch["pose"].clone()
            for b in range(B):
                subject = subjects[b]
                frame_idx = frame_indices[b]
                if subject in self._known_frames and frame_idx in self._known_frames[subject]:
                    key = self._subject_to_key[subject]
                    idx_tensor = torch.tensor(frame_idx, device=pose.device, dtype=torch.long)
                    pose[b] = pose[b] + self.pose_offsets[key][idx_tensor]
            batch["pose"] = pose

        if self.config.optimize_betas:
            betas = batch["betas"].clone()
            for b in range(B):
                subject = subjects[b]
                if subject in self._known_frames:
                    key = self._subject_to_key[subject]
                    betas[b] = betas[b] + self.betas_offsets[key]
            batch["betas"] = betas

        if self.config.optimize_trans and "trans" in batch and batch["trans"] is not None:
            trans = batch["trans"].clone()
            for b in range(B):
                subject = subjects[b]
                frame_idx = frame_indices[b]
                if subject in self._known_frames and frame_idx in self._known_frames[subject]:
                    key = self._subject_to_key[subject]
                    idx_tensor = torch.tensor(frame_idx, device=trans.device, dtype=torch.long)
                    trans[b] = trans[b] + self.trans_offsets[key][idx_tensor]
            batch["trans"] = trans

        return batch

    def get_param_groups(self) -> list[dict[str, Any]]:
        """Return optimizer parameter groups for enabled parameters.

        Returns:
            List of dicts with 'name', 'params', and 'lr' keys suitable
            for passing to torch.optim.Adam.
        """
        groups: list[dict[str, Any]] = []

        if self.config.optimize_pose:
            params = [p for p in self.pose_offsets.values()]
            if params:
                groups.append({
                    "name": "pose",
                    "params": params,
                    "lr": self.config.lr_pose,
                })

        if self.config.optimize_betas:
            params = [p for p in self.betas_offsets.values()]
            if params:
                groups.append({
                    "name": "betas",
                    "params": params,
                    "lr": self.config.lr_betas,
                })

        if self.config.optimize_trans:
            params = [p for p in self.trans_offsets.values()]
            if params:
                groups.append({
                    "name": "trans",
                    "params": params,
                    "lr": self.config.lr_trans,
                })

        return groups
