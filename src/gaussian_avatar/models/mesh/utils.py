"""Triangle computation utilities for mesh processing."""

import torch
from torch import Tensor


def get_triangle_vertices(vertices: Tensor, faces: Tensor) -> Tensor:
    """Group vertices by triangle.

    Args:
        vertices: Vertex positions (V, 3) or (B, V, 3)
        faces: Triangle connectivity (F, 3)

    Returns:
        Triangle vertices (F, 3, 3) or (B, F, 3, 3)
    """
    if vertices.dim() == 3:
        return vertices[:, faces]
    return vertices[faces]


def get_triangle_centroids(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute centroid of each triangle as mean of its vertices.

    Args:
        vertices: Vertex positions (V, 3) or (B, V, 3)
        faces: Triangle connectivity (F, 3)

    Returns:
        Triangle centroids (F, 3) or (B, F, 3)
    """
    return get_triangle_vertices(vertices, faces).mean(dim=-2)


def _safe_normalize(v: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Normalize vectors, returning validity mask for degenerate cases.

    Args:
        v: Vectors to normalize
        eps: Threshold below which vectors are considered degenerate

    Returns:
        Tuple of (normalized vectors, is_valid mask)
    """
    norm = v.norm(dim=-1, keepdim=True)
    is_valid = norm.squeeze(-1) >= eps
    return v / norm.clamp(min=eps), is_valid


def get_triangle_local_frames(vertices: Tensor, faces: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute orthonormal local coordinate frame for each triangle.

    Frame construction: z=normal, x=Gram-Schmidt orthogonalized edge, y=cross(z,x).
    Degenerate triangles fall back to identity rotation.

    Args:
        vertices: Vertex positions (V, 3) or (B, V, 3)
        faces: Triangle connectivity (F, 3)
        eps: Small value for numerical stability

    Returns:
        Rotation matrices (F, 3, 3) or (B, F, 3, 3) where columns are [x|y|z]
    """
    tri_verts = get_triangle_vertices(vertices, faces)

    v0, v1, v2 = tri_verts[..., 0, :], tri_verts[..., 1, :], tri_verts[..., 2, :]
    edge1, edge2 = v1 - v0, v2 - v0

    normal = torch.cross(edge1, edge2, dim=-1)
    z_axis, is_valid_normal = _safe_normalize(normal, eps)

    dot_e1_z = (edge1 * z_axis).sum(dim=-1, keepdim=True)
    x_axis_raw = edge1 - dot_e1_z * z_axis
    x_axis, is_valid_x = _safe_normalize(x_axis_raw, eps)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)

    frames = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    # Triangle is degenerate if normal or x-axis couldn't be normalized
    is_degenerate = ~is_valid_normal | ~is_valid_x
    if is_degenerate.any():
        # this should likely never happen with well formed meshes
        identity = torch.eye(3, device=frames.device, dtype=frames.dtype).expand(*frames.shape)
        frames = torch.where(
            is_degenerate.unsqueeze(-1).unsqueeze(-1).expand_as(frames),
            identity,
            frames,
        )

    return frames


def get_triangle_transforms(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute 4x4 homogeneous transforms (rotation + centroid translation) per triangle.

    Args:
        vertices: Vertex positions (V, 3) or (B, V, 3)
        faces: Triangle connectivity (F, 3)

    Returns:
        Transform matrices (F, 4, 4) or (B, F, 4, 4)
    """
    frames = get_triangle_local_frames(vertices, faces)
    centroids = get_triangle_centroids(vertices, faces)

    is_batched = vertices.dim() == 3
    shape = (vertices.shape[0], faces.shape[0], 4, 4) if is_batched else (faces.shape[0], 4, 4)

    transforms = torch.zeros(shape, device=vertices.device, dtype=vertices.dtype)
    transforms[..., 3, 3] = 1.0
    transforms[..., :3, :3] = frames
    transforms[..., :3, 3] = centroids

    return transforms


def get_triangle_stretches(
    canonical_vertices: Tensor,
    posed_vertices: Tensor,
    faces: Tensor,
    eps: float = 1e-8,
    canonical_frames: Tensor | None = None,
    max_stretch: float = 2.0,
) -> Tensor:
    """Compute per-axis stretch factors between canonical and posed triangles.

    For each axis, projects edges onto canonical local frame and computes
    the ratio of posed to canonical projections. Degenerate cases default to 1.0.

    Note: The normal axis (z) of each triangle has near-zero edge projections
    by construction (edges lie in the triangle plane). A relative threshold
    detects these degenerate axes and defaults them to 1.0. The final result
    is also clamped to [1/max_stretch, max_stretch] for numerical stability.

    Args:
        canonical_vertices: Canonical vertex positions (V, 3) or (B, V, 3)
        posed_vertices: Posed vertex positions (V, 3) or (B, V, 3)
        faces: Triangle connectivity (F, 3)
        eps: Small value for numerical stability
        canonical_frames: Pre-computed canonical local frames (F, 3, 3) or (B, F, 3, 3).
            If None, computed from canonical_vertices. Pass this to avoid
            redundant computation when frames are already available.
        max_stretch: Maximum allowed stretch factor per axis (default 2.0).
            Stretches are clamped to [1/max_stretch, max_stretch].

    Returns:
        Stretch factors (F, 3) or (B, F, 3) for each axis [x, y, z]
    """
    can_tri_verts = get_triangle_vertices(canonical_vertices, faces)
    posed_tri_verts = get_triangle_vertices(posed_vertices, faces)

    can_frames = canonical_frames if canonical_frames is not None else get_triangle_local_frames(canonical_vertices, faces)

    # Compute all 3 edges per triangle: v1-v0, v2-v1, v0-v2
    # roll(..., -1, -2) shifts vertices: [v0,v1,v2] -> [v1,v2,v0]
    can_edges = torch.roll(can_tri_verts, -1, dims=-2) - can_tri_verts
    posed_edges = torch.roll(posed_tri_verts, -1, dims=-2) - posed_tri_verts

    # Project edges onto canonical frame axes using batched matrix multiply
    # edges: (..., F, 3, 3) -> (..., F, 3, 3) [3 edges × 3 coords]
    # frames: (..., F, 3, 3) [3 axes × 3 coords] stored as columns
    # We want: projections[..., e, a] = dot(edge_e, axis_a)
    # Using matmul: edges @ frames gives (..., F, 3, 3) [3 edges × 3 axes]
    can_projections = can_edges @ can_frames
    posed_projections = posed_edges @ can_frames

    # Get maximum absolute projection per axis (across all 3 edges)
    can_max_proj = can_projections.abs().max(dim=-2).values
    posed_max_proj = posed_projections.abs().max(dim=-2).values

    stretches = posed_max_proj / can_max_proj.clamp(min=eps)

    # Detect degenerate axes using a relative threshold: any axis where the
    # canonical projection is < 1% of the triangle's max projection is
    # considered degenerate (typically the normal axis). This is more robust
    # than using a fixed absolute eps, which misses near-zero z-projections.
    tri_max_proj = can_max_proj.max(dim=-1, keepdim=True).values
    is_degenerate = can_max_proj < 0.01 * tri_max_proj
    stretches = torch.where(is_degenerate, torch.ones_like(stretches), stretches)

    return stretches.clamp(min=1.0 / max_stretch, max=max_stretch)
