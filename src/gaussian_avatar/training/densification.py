"""Densification manager for adaptive Gaussian density control.

This module provides the DensificationManager class that tracks gradient
statistics and manages split, clone, and prune operations for Gaussians.
"""

import torch
from torch import Tensor

from gaussian_avatar.configs.training import DensificationConfig
from gaussian_avatar.models.gaussian import GaussianModel


class DensificationManager:
    """Manages adaptive Gaussian densification.

    Tracks gradient accumulation and executes split/clone/prune
    operations to optimize Gaussian distribution.

    Args:
        gaussian_model: GaussianModel to manage
        config: DensificationConfig with schedule and thresholds
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        config: DensificationConfig,
        debug: bool = False,
    ) -> None:
        self.gaussian_model = gaussian_model
        self.config = config
        self._debug = debug

        self._grad_accum: Tensor | None = None
        self._grad_count: Tensor | None = None
        self._reset_accumulators()

    @property
    def num_faces(self) -> int:
        """Return the number of mesh faces."""
        return self.gaussian_model._num_faces

    def _reset_accumulators(self) -> None:
        """Reset gradient accumulation buffers to zero.

        Called after densification to start fresh accumulation for
        the next densification interval.
        """
        N = self.gaussian_model.num_gaussians
        device = self.gaussian_model.local_positions.device
        self._grad_accum = torch.zeros(N, device=device)
        self._grad_count = torch.zeros(N, device=device)

    def state_dict(self) -> dict[str, Tensor]:
        """Return serializable state for checkpointing.

        Returns:
            Dictionary with grad_accum and grad_count tensors.
        """
        return {
            "grad_accum": self._grad_accum.clone(),
            "grad_count": self._grad_count.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Restore gradient accumulators from checkpoint state.

        If the saved tensor size doesn't match the current Gaussian count
        (e.g. due to densification between save and load), the accumulators
        are reset to zero at the correct size instead of crashing.

        Args:
            state: Dictionary with grad_accum and grad_count tensors.
        """
        N = self.gaussian_model.num_gaussians
        accum = state.get("grad_accum")
        count = state.get("grad_count")

        if accum is not None and count is not None and accum.shape[0] == N:
            device = self.gaussian_model.local_positions.device
            self._grad_accum = accum.to(device)
            self._grad_count = count.to(device)
        else:
            self._reset_accumulators()

    def accumulate_gradients(
        self,
        viewspace_points: Tensor,
        visibility_filter: Tensor,
        means2d: Tensor | None = None,
        use_absgrad: bool = False,
    ) -> None:
        """Accumulate viewspace position gradients.

        Called after each render during training to track which Gaussians
        have high screen-space gradients, indicating they need refinement.

        Args:
            viewspace_points: 2D screen positions (N, 2), unused for grad
                reading but kept for interface compatibility
            visibility_filter: Boolean mask of visible Gaussians (N,)
            means2d: Full means2d tensor from gsplat (C, N, 2) with
                retain_grad() already called. Gradients are read from
                this tensor since the sliced viewspace_points is a dead
                autograd branch.
            use_absgrad: If True, read absolute-value gradients from
                means2d.absgrad instead of means2d.grad. Requires the
                renderer was called with absgrad=True. Default False.
        """
        # Read gradient from the full means2d tensor (camera index 0)
        # because sliced viewspace_points doesn't receive gradients
        # through gsplat's custom backward.
        grad_source = means2d if means2d is not None else viewspace_points

        # --- DEBUG: inspect means2d values and gradients ---
        if self._debug:
            src_name = "means2d" if means2d is not None else "viewspace_points"
            print(f"\n[DEBUG accumulate_gradients] source={src_name}, absgrad={use_absgrad}")
            print(f"  grad_source shape: {grad_source.shape}")
            print(f"  grad_source requires_grad: {grad_source.requires_grad}")
            vals = grad_source if grad_source.dim() == 2 else grad_source[0]
            vis_vals = vals[visibility_filter]
            print(f"  means2d values (visible) min: {vis_vals.min(dim=0).values.tolist()}")
            print(f"  means2d values (visible) max: {vis_vals.max(dim=0).values.tolist()}")
            print(f"  means2d values (visible) mean: {vis_vals.mean(dim=0).tolist()}")
            print(f"  visibility_filter: {visibility_filter.sum().item()}/{visibility_filter.shape[0]} visible")
            print(f"  grad_source.grad is None: {grad_source.grad is None}")
            if grad_source.grad is not None:
                g = grad_source.grad if grad_source.grad.dim() == 2 else grad_source.grad[0]
                vis_g = g[visibility_filter]
                gm = vis_g.norm(dim=-1)
                print(f"  grad shape: {g.shape}")
                print(f"  grad (visible) min: {vis_g.min(dim=0).values.tolist()}")
                print(f"  grad (visible) max: {vis_g.max(dim=0).values.tolist()}")
                print(f"  grad (visible) mean: {vis_g.mean(dim=0).tolist()}")
                print(f"  grad magnitude (visible) min: {gm.min().item():.8e}")
                print(f"  grad magnitude (visible) max: {gm.max().item():.8e}")
                print(f"  grad magnitude (visible) mean: {gm.mean().item():.8e}")
        # --- END DEBUG ---

        if use_absgrad:
            # absgrad: absolute-value gradients from gsplat, prevents
            # gradient cancellation for Gaussians straddling edges
            if not hasattr(grad_source, "absgrad") or grad_source.absgrad is None:
                return
            grad = grad_source.absgrad
        else:
            if grad_source.grad is None:
                return
            grad = grad_source.grad

        if grad.dim() == 3:
            # means2d shape is (C, N, 2), take camera 0
            grad = grad[0]

        grad_magnitude = grad.norm(dim=-1)

        self._grad_accum[visibility_filter] += grad_magnitude[visibility_filter]
        self._grad_count[visibility_filter] += 1

    def get_gradient_stats(self) -> dict[str, float]:
        """Compute statistics of accumulated viewspace gradients.

        Returns average, std, max, and min of the per-Gaussian mean gradient
        magnitude (only over Gaussians that were visible at least once).

        Returns:
            Dictionary with keys: mean, std, max, min (all floats).
            Returns all zeros if no gradients have been accumulated.
        """
        visible = self._grad_count > 0
        if not visible.any():
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

        avg_grad = self._grad_accum[visible] / self._grad_count[visible]
        return {
            "mean": avg_grad.mean().item(),
            "std": avg_grad.std().item() if avg_grad.numel() > 1 else 0.0,
            "max": avg_grad.max().item(),
            "min": avg_grad.min().item(),
        }

    def _protect_triangles(
        self,
        prune_mask: Tensor,
        opacities: Tensor,
    ) -> Tensor:
        """Ensure each triangle keeps at least one Gaussian.

        When all Gaussians of a triangle would be pruned, keeps the one
        with highest opacity to maintain mesh coverage. Uses vectorized
        scatter operations for O(N) complexity instead of O(F*N).

        Args:
            prune_mask: Boolean mask of Gaussians to prune
            opacities: Current opacity values (N,) or (N, 1)

        Returns:
            Modified prune mask with triangle protection applied
        """
        opacities = opacities.squeeze()
        triangle_idx = self.gaussian_model.triangle_indices
        prune_mask = prune_mask.clone()

        N = prune_mask.shape[0]
        F = self.num_faces
        device = prune_mask.device

        # Count total and pruned Gaussians per face
        total_per_face = torch.zeros(F, device=device)
        total_per_face.scatter_add_(0, triangle_idx, torch.ones(N, device=device))

        pruned_per_face = torch.zeros(F, device=device)
        pruned_per_face.scatter_add_(0, triangle_idx, prune_mask.float())

        # Faces where all Gaussians would be pruned
        fully_pruned_faces = (pruned_per_face == total_per_face) & (total_per_face > 0)

        if not fully_pruned_faces.any():
            return prune_mask

        # Mask of Gaussians on fully-pruned faces
        on_fully_pruned = fully_pruned_faces[triangle_idx]

        # Find max opacity per fully-pruned face
        neg_inf = torch.full((F,), float('-inf'), device=device)
        candidate_opacities = torch.where(
            on_fully_pruned, opacities,
            torch.tensor(float('-inf'), device=device),
        )
        best_per_face = neg_inf.scatter_reduce(
            0, triangle_idx, candidate_opacities,
            reduce='amax', include_self=False,
        )

        # Unmark Gaussians that are the best on their fully-pruned face
        is_best = on_fully_pruned & (opacities == best_per_face[triangle_idx])
        prune_mask[is_best] = False

        return prune_mask

    def _prune(
        self,
        mask: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Remove Gaussians marked for pruning.

        Args:
            mask: Boolean tensor (N,) where True = prune, False = keep
            optimizer: Optimizer for state management

        Returns:
            Number of Gaussians removed
        """
        num_to_prune = mask.sum().item()
        if num_to_prune == 0:
            return 0

        keep_mask = ~mask
        self.gaussian_model.remove_gaussians(keep_mask, optimizer)
        return num_to_prune

    def _clone(
        self,
        mask: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Clone Gaussians with small position perturbation.

        Creates duplicates of selected Gaussians with added position
        noise to prevent identical overlapping Gaussians.

        Args:
            mask: Boolean tensor (N,) where True = clone
            optimizer: Optimizer for state management

        Returns:
            Number of Gaussians added (cloned)
        """
        indices = mask.nonzero(as_tuple=True)[0]
        n_clones = len(indices)

        if n_clones == 0:
            return 0

        new_positions = self.gaussian_model.local_positions[indices].clone()
        new_rotations = self.gaussian_model.local_rotations[indices].clone()
        new_scales = self.gaussian_model.local_scales[indices].clone()
        new_opacities = self.gaussian_model.raw_opacities[indices].clone()
        new_triangle_idx = self.gaussian_model.triangle_indices[indices].clone()

        # Prevent identical overlapping clones
        noise = torch.randn_like(new_positions) * self.config.clone_position_noise
        new_positions = new_positions + noise

        new_params = {
            "local_positions": new_positions,
            "local_rotations": new_rotations,
            "local_scales": new_scales,
            "opacity": new_opacities,
            "triangle_idx": new_triangle_idx,
        }
        if self.gaussian_model.use_sh:
            new_params["sh_coefficients"] = self.gaussian_model.sh_coefficients[indices].clone()
        else:
            new_params["base_colors"] = self.gaussian_model.base_colors[indices].clone()
        self.gaussian_model.add_gaussians(new_params, optimizer)

        return n_clones

    def _split(
        self,
        mask: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> int:
        """Split Gaussians into two smaller ones.

        Each split Gaussian is replaced with two new Gaussians that have:
        - Positions sampled from the original's distribution
        - Reduced scales (divided by 1.6)
        - Same rotations, colors, opacity, triangle assignment

        Args:
            mask: Boolean tensor (N,) where True = split
            optimizer: Optimizer for state management

        Returns:
            Number of Gaussians added (2 per split)
        """
        indices = mask.nonzero(as_tuple=True)[0]
        n_splits = len(indices)

        if n_splits == 0:
            return 0

        positions = self.gaussian_model.local_positions[indices]  # (S, 3)
        scales = torch.exp(self.gaussian_model.local_scales[indices])  # (S, 3)

        samples_1 = torch.randn_like(positions) * scales
        samples_2 = torch.randn_like(positions) * scales

        new_positions_1 = positions + samples_1
        new_positions_2 = positions + samples_2

        # Standard 3DGS split scale reduction factor
        scale_reduction = 1.6
        new_log_scales = self.gaussian_model.local_scales[indices] - torch.log(
            torch.tensor(scale_reduction, device=scales.device)
        )

        new_rotations = self.gaussian_model.local_rotations[indices]
        new_opacities = self.gaussian_model.raw_opacities[indices]
        new_triangle_idx = self.gaussian_model.triangle_indices[indices]

        new_params = {
            "local_positions": torch.cat([new_positions_1, new_positions_2], dim=0),
            "local_rotations": torch.cat([new_rotations, new_rotations.clone()], dim=0),
            "local_scales": torch.cat([new_log_scales, new_log_scales.clone()], dim=0),
            "opacity": torch.cat([new_opacities, new_opacities.clone()], dim=0),
            "triangle_idx": torch.cat([new_triangle_idx, new_triangle_idx.clone()], dim=0),
        }
        if self.gaussian_model.use_sh:
            new_sh = self.gaussian_model.sh_coefficients[indices]
            new_params["sh_coefficients"] = torch.cat([new_sh, new_sh.clone()], dim=0)
        else:
            new_colors = self.gaussian_model.base_colors[indices]
            new_params["base_colors"] = torch.cat([new_colors, new_colors.clone()], dim=0)

        keep_mask = ~mask
        self.gaussian_model.remove_gaussians(keep_mask, optimizer)
        self.gaussian_model.add_gaussians(new_params, optimizer)

        return n_splits * 2

    def reset_opacity(self) -> None:
        """Clamp Gaussian opacities to at most the configured reset value.

        Computes min(current_opacity, opacity_reset_value) in raw (pre-sigmoid)
        space. This only decreases opacity, never increases it, allowing
        pruning of high-opacity Gaussians that are not useful while preserving
        already-low opacities.
        """
        val = self.config.opacity_reset_value
        raw_cap = torch.log(torch.tensor(val / (1.0 - val))).item()
        self.gaussian_model.raw_opacities.data.clamp_(max=raw_cap)

    def densify(self, optimizer: torch.optim.Optimizer) -> dict[str, int | Tensor]:
        """Execute densification: split, clone, and prune.

        Identifies Gaussians that need refinement based on accumulated
        gradients and executes appropriate operations:
        - Split: high gradient + large scale
        - Clone: high gradient + small scale
        - Prune: low opacity or never visible during accumulation window

        Args:
            optimizer: Optimizer for state management

        Returns:
            Dictionary with counts (split, cloned, pruned) and per-face
            count tensors (split_per_face, clone_per_face).
        """
        avg_grad = self._grad_accum / self._grad_count.clamp(min=1)

        scales = torch.exp(self.gaussian_model.local_scales)
        max_scale = scales.max(dim=-1).values
        opacities = torch.sigmoid(self.gaussian_model.raw_opacities).squeeze(-1)

        high_grad = avg_grad > self.config.grad_threshold
        large_scale = max_scale > self.config.split_scale_threshold
        low_opacity = opacities < self.config.min_opacity
        never_visible = self._grad_count == 0

        split_mask = high_grad & large_scale

        # Force-split oversized Gaussians regardless of gradient
        if self.config.force_split_scale is not None:
            oversized = max_scale > self.config.force_split_scale
            split_mask = split_mask | oversized

        clone_mask = high_grad & ~split_mask
        raw_prune_mask = low_opacity | never_visible
        if self.config.protect_all_triangles:
            prune_mask = self._protect_triangles(raw_prune_mask, opacities)
        else:
            prune_mask = raw_prune_mask

        # Capture per-face counts before operations invalidate masks
        triangle_idx = self.gaussian_model.triangle_indices
        num_faces = self.num_faces
        device = split_mask.device

        split_per_face = torch.zeros(num_faces, device=device)
        split_per_face.scatter_add_(0, triangle_idx, split_mask.float())

        clone_per_face = torch.zeros(num_faces, device=device)
        clone_per_face.scatter_add_(0, triangle_idx, clone_mask.float())

        # Order matters: split first (changes indices), then clone, then prune
        stats: dict[str, int | Tensor] = {"split": 0, "cloned": 0, "pruned": 0}

        over_limit = (
            self.config.max_gaussians is not None
            and self.gaussian_model.num_gaussians >= self.config.max_gaussians
        )

        if split_mask.any() and not over_limit:
            stats["split"] = self._split(split_mask, optimizer)

        # Splits remove+add Gaussians, invalidating masks from the original tensor.
        # Remap clone candidates to their new indices in the post-split model.
        over_limit = (
            self.config.max_gaussians is not None
            and self.gaussian_model.num_gaussians >= self.config.max_gaussians
        )
        if clone_mask.any() and not over_limit:
            clone_after_split = clone_mask & ~split_mask
            if clone_after_split.any():
                keep_after_split = ~split_mask
                clone_indices_original = clone_after_split.nonzero(as_tuple=True)[0]

                new_indices = []
                for orig_idx in clone_indices_original:
                    new_idx = keep_after_split[:orig_idx].sum().item()
                    new_indices.append(new_idx)

                if new_indices:
                    new_clone_mask = torch.zeros(
                        self.gaussian_model.num_gaussians, dtype=torch.bool,
                        device=clone_mask.device
                    )
                    new_clone_mask[new_indices] = True
                    stats["cloned"] = self._clone(new_clone_mask, optimizer)

        if prune_mask.any():
            # Recompute after split/clone changed the Gaussian set
            current_opacities = torch.sigmoid(self.gaussian_model.raw_opacities).squeeze(-1)
            current_low_opacity = current_opacities < self.config.min_opacity
            if self.config.protect_all_triangles:
                current_prune_mask = self._protect_triangles(current_low_opacity, current_opacities)
            else:
                current_prune_mask = current_low_opacity
            if current_prune_mask.any():
                stats["pruned"] = self._prune(current_prune_mask, optimizer)

        self._reset_accumulators()

        stats["split_per_face"] = split_per_face
        stats["clone_per_face"] = clone_per_face

        return stats
