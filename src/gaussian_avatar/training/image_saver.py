"""Image saving utility for training and evaluation.

Provides ImageSaver class for saving rendered and ground truth images
during training and evaluation with consistent naming conventions.
"""

import functools
import logging
import threading
from pathlib import Path
from queue import Queue

import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from torch import Tensor

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _bg_save(fn):
    """Decorator adding background dispatch to ImageSaver save methods.

    When ``self._background`` is True, snapshots all Tensor arguments to
    CPU on the calling thread, queues the actual work to the background
    worker, and returns None.  When False, calls *fn* directly.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self._background:
            return fn(self, *args, **kwargs)

        cpu_args = tuple(
            a.detach().cpu() if isinstance(a, Tensor) else a for a in args
        )
        cpu_kwargs = {
            k: v.detach().cpu() if isinstance(v, Tensor) else v
            for k, v in kwargs.items()
        }
        self._submit(lambda: fn(self, *cpu_args, **cpu_kwargs))
        return None

    return wrapper


class ImageSaver:
    """Utility for saving training and evaluation images.

    Saves rendered RGB, alpha, and optionally ground truth images
    with consistent naming conventions.

    Args:
        output_dir: Base directory for saving images
        save_gt: Whether to save ground truth images alongside renders
        background: If True, offload image encoding and disk I/O to a
            daemon worker thread so the caller is not blocked.
    """

    def __init__(self, output_dir: str | Path, save_gt: bool = True, background: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_gt = save_gt
        self._background = background
        if background:
            self._queue: Queue = Queue(maxsize=100)
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        else:
            self._queue = None
            self._thread = None

    def _worker(self) -> None:
        """Background worker loop: pull callables from queue and execute them."""
        while True:
            fn = self._queue.get()
            if fn is None:
                self._queue.task_done()
                break
            try:
                fn()
            except Exception:
                logger.exception("Background image save failed")
            self._queue.task_done()

    def _submit(self, fn) -> None:
        """Submit work to the background thread, or run inline if synchronous."""
        if self._background:
            self._queue.put(fn)
        else:
            fn()

    def flush(self) -> None:
        """Block until all queued saves complete. No-op if not background."""
        if self._queue is not None:
            self._queue.join()

    def shutdown(self) -> None:
        """Stop the background worker thread. Safe to call multiple times."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=60)
        self._thread = None

    @_bg_save
    def save_training_image(
        self,
        rendered_rgb: Tensor,
        rendered_alpha: Tensor | None,
        target_rgb: Tensor | None,
        iteration: int,
        flat: bool = False,
    ) -> dict[str, Path]:
        """Save images from a training iteration.

        Args:
            rendered_rgb: Rendered RGB image (3, H, W) in [0, 1]
            rendered_alpha: Rendered alpha/opacity (H, W) in [0, 1], optional
            target_rgb: Ground truth RGB (3, H, W) in [0, 1], optional
            iteration: Training iteration number
            flat: If True, save all images in a single ``last_n/`` directory
                with ``iter_NNNNNN_`` filename prefixes instead of per-iteration
                subdirectories.

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        saved = {}

        if flat:
            out_dir = self.output_dir / "last_n"
            out_dir.mkdir(exist_ok=True)
            prefix = f"iter_{iteration:06d}_"
        else:
            out_dir = self.output_dir / f"iter_{iteration:06d}"
            out_dir.mkdir(exist_ok=True)
            prefix = ""

        rgb_path = out_dir / f"{prefix}render_rgb.png"
        self._save_tensor_as_image(rendered_rgb, rgb_path)
        saved["render_rgb"] = rgb_path

        if rendered_alpha is not None:
            alpha_path = out_dir / f"{prefix}render_alpha.png"
            self._save_tensor_as_image(rendered_alpha.unsqueeze(0), alpha_path)
            saved["render_alpha"] = alpha_path

        if self.save_gt and target_rgb is not None:
            comparison_path = out_dir / f"{prefix}comparison.png"
            self._save_comparison(rendered_rgb, target_rgb, comparison_path)
            saved["comparison"] = comparison_path

        return saved

    @_bg_save
    def save_evaluation_image(
        self,
        rendered_rgb: Tensor,
        target_rgb: Tensor,
        frame_idx: int,
        subject: str | None = None,
    ) -> dict[str, Path]:
        """Save images from evaluation.

        Args:
            rendered_rgb: Rendered RGB image (3, H, W) in [0, 1]
            target_rgb: Ground truth RGB (3, H, W) in [0, 1]
            frame_idx: Frame index in the test set
            subject: Optional subject name for multi-subject eval

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        saved = {}

        prefix = f"frame_{frame_idx:04d}"
        if subject:
            prefix = f"{subject}_{prefix}"

        render_path = self.output_dir / f"{prefix}_render.png"
        self._save_tensor_as_image(rendered_rgb, render_path)
        saved["render"] = render_path

        gt_path = self.output_dir / f"{prefix}_gt.png"
        self._save_tensor_as_image(target_rgb, gt_path)
        saved["gt"] = gt_path

        comparison_path = self.output_dir / f"{prefix}_comparison.png"
        self._save_comparison(rendered_rgb, target_rgb, comparison_path)
        saved["comparison"] = comparison_path

        return saved

    #: Fixed colormap ceiling for LPIPS heatmaps (VGG).
    LPIPS_VMAX: float = 0.7
    #: Fixed colormap ceiling for SSIM error heatmaps (1 - SSIM).
    SSIM_VMAX: float = 0.7

    @_bg_save
    def save_lpips_heatmap(
        self,
        lpips_map: Tensor,
        frame_idx: int,
        subject: str | None = None,
    ) -> Path:
        """Save per-pixel LPIPS spatial distance as a jet-coloured heatmap.

        Args:
            lpips_map: Per-pixel LPIPS distance (H, W)
            frame_idx: Frame index in the test set
            subject: Optional subject name for multi-subject eval

        Returns:
            Path to the saved heatmap image, or None if background
        """
        return self._save_jet_heatmap(
            lpips_map, frame_idx, subject, "lpips_heatmap", vmax=self.LPIPS_VMAX
        )

    @_bg_save
    def save_ssim_heatmap(
        self,
        ssim_error_map: Tensor,
        frame_idx: int,
        subject: str | None = None,
    ) -> Path:
        """Save per-pixel SSIM error (1 - SSIM) as a jet-coloured heatmap.

        Args:
            ssim_error_map: Per-pixel SSIM error (H, W), higher = worse
            frame_idx: Frame index in the test set
            subject: Optional subject name for multi-subject eval

        Returns:
            Path to the saved heatmap image, or None if background
        """
        return self._save_jet_heatmap(
            ssim_error_map, frame_idx, subject, "ssim_heatmap", vmax=self.SSIM_VMAX
        )

    @_bg_save
    def save_heatmap(
        self, error_map: Tensor, path: Path, vmax: float | None = None
    ) -> Path:
        """Save a scalar error map as a jet-coloured heatmap.

        Public wrapper around ``_save_jet_heatmap_to_path`` that supports
        background saving.

        Args:
            error_map: Per-pixel scalar values (H, W)
            path: Output file path
            vmax: Fixed upper bound for normalization (see
                ``_save_jet_heatmap_to_path``).

        Returns:
            Path to the saved heatmap image, or None if background
        """
        return self._save_jet_heatmap_to_path(error_map, path, vmax=vmax)

    def _save_jet_heatmap(
        self,
        error_map: Tensor,
        frame_idx: int,
        subject: str | None,
        suffix: str,
        vmax: float | None = None,
    ) -> Path:
        """Normalize a scalar error map and save as a jet-coloured heatmap.

        Args:
            error_map: Per-pixel scalar values (H, W)
            frame_idx: Frame index in the test set
            subject: Optional subject name for multi-subject eval
            suffix: Filename suffix before .png (e.g. "lpips_heatmap")
            vmax: Fixed upper bound for normalization (see
                ``_save_jet_heatmap_to_path``).

        Returns:
            Path to the saved heatmap image
        """
        prefix = f"frame_{frame_idx:04d}"
        if subject:
            prefix = f"{subject}_{prefix}"

        path = self.output_dir / f"{prefix}_{suffix}.png"
        return self._save_jet_heatmap_to_path(error_map, path, vmax=vmax)

    def _save_jet_heatmap_to_path(
        self,
        error_map: Tensor,
        path: Path,
        vmax: float | None = None,
    ) -> Path:
        """Normalize a scalar error map and save as a jet-coloured heatmap.

        Args:
            error_map: Per-pixel scalar values (H, W)
            path: Output file path
            vmax: Fixed upper bound for normalization. When set, the
                colormap spans [0, vmax] so heatmaps are comparable
                across frames.  Values above *vmax* clip to red.
                When ``None``, falls back to per-image max (legacy).

        Returns:
            Path to the saved heatmap image
        """
        arr = error_map.cpu().numpy()
        actual_max = float(arr.max())
        denom = vmax if vmax is not None else max(actual_max, 1e-5)
        normalised = np.clip(arr / denom, 0, 1)
        heatmap = (cm.jet(normalised)[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(heatmap)
        draw = ImageDraw.Draw(img)
        label = f"max={actual_max:.4f}"
        if vmax is not None:
            label += f"  vmax={vmax}"
        draw.text((10, 10), label, fill=(255, 255, 255))

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    def _save_tensor_as_image(self, tensor: Tensor, path: Path) -> None:
        """Convert tensor to PIL image and save.

        Args:
            tensor: Image tensor (C, H, W) or (H, W) in [0, 1]
            path: Output file path
        """
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        img_np = (tensor.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()

        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)

        Image.fromarray(img_np).save(path)

    def _save_comparison(
        self,
        rendered: Tensor,
        target: Tensor,
        path: Path,
    ) -> None:
        """Save side-by-side comparison image.

        Args:
            rendered: Rendered image (3, H, W)
            target: Ground truth image (3, H, W)
            path: Output file path
        """
        comparison = torch.cat([target, rendered], dim=2)
        self._save_tensor_as_image(comparison, path)

    @_bg_save
    def save_canonical_density(
        self,
        vertices: Tensor,
        faces: Tensor,
        triangle_idx: Tensor,
        num_faces: int,
    ) -> dict[str, Path]:
        """Save per-face Gaussian count heatmap on the canonical mesh.

        Counts the number of Gaussians assigned to each triangle and renders
        the canonical mesh colored by relative Gaussian density.

        Args:
            vertices: Canonical mesh vertices (V, 3)
            faces: Triangle face indices (F, 3)
            triangle_idx: Per-Gaussian triangle assignment (N,)
            num_faces: Total number of faces

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        counts = torch.bincount(
            triangle_idx.cpu().long(), minlength=num_faces
        ).float().numpy()

        max_val = counts.max()
        normalized = counts / max_val if max_val > 0 else counts

        img = render_face_heatmap(vertices, faces, normalized)
        path = self.output_dir / "canonical_density.png"
        img.save(path)
        return {"canonical_density": path}

    @_bg_save
    def save_projected_density(
        self,
        means: Tensor,
        intrinsic: Tensor,
        extrinsic: Tensor,
        height: int,
        width: int,
        frame_idx: int,
        subject: str | None = None,
    ) -> dict[str, Path]:
        """Save 2D projected Gaussian density heatmap for a camera view.

        Projects all Gaussian centers into the camera view and renders a
        smoothed density heatmap showing where Gaussians concentrate.

        Args:
            means: Gaussian world-space positions (N, 3)
            intrinsic: Camera intrinsic matrix (3, 3)
            extrinsic: Camera extrinsic matrix (4, 4)
            height: Image height in pixels
            width: Image width in pixels
            frame_idx: Frame index in the test set
            subject: Optional subject name for multi-subject eval

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        img = render_projected_density(
            means=means,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            height=height,
            width=width,
        )

        prefix = f"frame_{frame_idx:04d}"
        if subject:
            prefix = f"{subject}_{prefix}"

        path = self.output_dir / f"{prefix}_projected_density.png"
        img.save(path)
        return {"projected_density": path}

    @_bg_save
    def save_offset_mesh_heatmap(
        self,
        vertices: Tensor,
        faces: Tensor,
        triangle_idx: Tensor,
        position_offsets: Tensor,
        iteration: int,
    ) -> dict[str, Path]:
        """Save per-face MLP position offset magnitude on the canonical mesh.

        Averages ||δμ|| per triangle and renders a heatmap showing where
        the deformation MLP produces the largest position corrections.

        Args:
            vertices: Canonical mesh vertices (V, 3)
            faces: Triangle face indices (F, 3)
            triangle_idx: Per-Gaussian triangle assignment (N,)
            position_offsets: MLP position offsets δμ (N, 3)
            iteration: Training iteration number

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        num_faces = faces.shape[0]
        offset_norms = position_offsets.detach().cpu().float().norm(dim=-1)  # (N,)
        tri_idx = triangle_idx.cpu().long()

        # Guard against size mismatch (can happen if densification runs
        # between the forward pass that produced offsets and this call)
        n = min(tri_idx.shape[0], offset_norms.shape[0])
        tri_idx = tri_idx[:n]
        offset_norms = offset_norms[:n]

        # Accumulate sum and count per face
        face_sum = torch.zeros(num_faces, dtype=torch.float32)
        face_count = torch.zeros(num_faces, dtype=torch.float32)
        face_sum.scatter_add_(0, tri_idx, offset_norms)
        face_count.scatter_add_(0, tri_idx, torch.ones_like(offset_norms))

        # Mean offset per face (avoid division by zero)
        face_mean = torch.where(
            face_count > 0, face_sum / face_count, torch.zeros_like(face_sum)
        ).numpy()

        # Normalize to [0, 1]
        max_val = face_mean.max()
        normalized = face_mean / max_val if max_val > 0 else face_mean

        img = render_face_heatmap(vertices, faces, normalized)

        # Add text annotation with max offset value
        draw = ImageDraw.Draw(img)
        draw.text(
            (10, 10),
            f"iter {iteration}  max ||δμ||={max_val:.4f}",
            fill=(255, 255, 255),
        )

        heatmap_dir = self.output_dir / "offset_heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        path = heatmap_dir / f"iter_{iteration:06d}_mesh.png"
        img.save(path)
        return {"offset_mesh_heatmap": path}

    @_bg_save
    def save_projected_offset_heatmap(
        self,
        means: Tensor,
        position_offsets: Tensor,
        intrinsic: Tensor,
        extrinsic: Tensor,
        height: int,
        width: int,
        iteration: int,
    ) -> dict[str, Path]:
        """Save 2D projected offset magnitude heatmap for a camera view.

        Projects Gaussian centers into the camera view and colors each
        by the magnitude of its MLP position offset ||δμ||.

        Args:
            means: Gaussian world-space positions (N, 3)
            position_offsets: MLP position offsets δμ (N, 3)
            intrinsic: Camera intrinsic matrix (3, 3)
            extrinsic: Camera extrinsic matrix (4, 4)
            height: Image height in pixels
            width: Image width in pixels
            iteration: Training iteration number

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        offset_norms = position_offsets.detach().norm(dim=-1)  # (N,)

        img = render_projected_values(
            means=means,
            values=offset_norms,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            height=height,
            width=width,
        )

        # Add annotation
        max_offset = offset_norms.max().item()
        draw = ImageDraw.Draw(img)
        draw.text(
            (10, 10),
            f"iter {iteration}  max ||δμ||={max_offset:.4f}",
            fill=(255, 255, 255),
        )

        heatmap_dir = self.output_dir / "offset_heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        path = heatmap_dir / f"iter_{iteration:06d}_projected.png"
        img.save(path)
        return {"projected_offset_heatmap": path}

    @_bg_save
    def save_densification_heatmap(
        self,
        vertices: Tensor,
        faces: Tensor,
        split_per_face: Tensor,
        clone_per_face: Tensor,
        iteration: int,
    ) -> dict[str, Path]:
        """Save per-face densification heatmaps on the canonical mesh.

        Renders the canonical pose mesh from the front with faces colored
        by split or clone count intensity. Produces separate images for
        splits and clones.

        Args:
            vertices: Canonical mesh vertices (V, 3)
            faces: Triangle face indices (F, 3)
            split_per_face: Per-face split counts (F,)
            clone_per_face: Per-face clone counts (F,)
            iteration: Training iteration number

        Returns:
            Dictionary mapping image type to saved path, or None if background
        """
        saved = {}
        heatmap_dir = self.output_dir / "densification_heatmaps"
        heatmap_dir.mkdir(exist_ok=True)

        split_counts = split_per_face.detach().cpu().float().numpy()
        clone_counts = clone_per_face.detach().cpu().float().numpy()

        for label, counts in [("split", split_counts), ("clone", clone_counts)]:
            max_val = counts.max()
            if max_val > 0:
                normalized = counts / max_val
            else:
                normalized = counts

            img = render_face_heatmap(vertices, faces, normalized)
            path = heatmap_dir / f"iter_{iteration:06d}_{label}.png"
            img.save(path)
            saved[label] = path

        return saved

    @_bg_save
    def save_mesh_overlay(
        self,
        target_rgb: Tensor,
        vertices: Tensor,
        faces: Tensor,
        intrinsic: Tensor,
        extrinsic: Tensor,
        path: Path,
    ) -> None:
        """Save side-by-side [GT | GT + SMPL wireframe overlay] image.

        Args:
            target_rgb: Ground truth RGB image (3, H, W) in [0, 1]
            vertices: Posed mesh vertices (V, 3) in world coordinates
            faces: Triangle face indices (F, 3)
            intrinsic: Camera intrinsic matrix (3, 3)
            extrinsic: Camera extrinsic matrix (4, 4)
            path: Output file path
        """
        overlay_img = render_wireframe_overlay(
            image=target_rgb,
            vertices=vertices,
            faces=faces,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )
        gt_np = (
            (target_rgb.clamp(0, 1) * 255)
            .byte()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        gt_img = Image.fromarray(gt_np)

        W = gt_img.width
        H = gt_img.height
        combined = Image.new("RGB", (W * 2, H))
        combined.paste(gt_img, (0, 0))
        combined.paste(overlay_img, (W, 0))
        combined.save(path)


def render_face_heatmap(
    vertices: Tensor,
    faces: Tensor,
    face_values: np.ndarray,
    image_size: int = 1024,
    margin: int = 40,
) -> Image.Image:
    """Render a front-view heatmap of the canonical mesh with per-face colors.

    Uses orthographic projection onto the XY plane with painter's algorithm
    (back-to-front depth sorting) for occlusion.

    Args:
        vertices: Canonical mesh vertices (V, 3)
        faces: Triangle face indices (F, 3)
        face_values: Per-face scalar values (F,) in [0, 1], where 0 = no
            activity and 1 = maximum activity
        image_size: Output image height/width in pixels
        margin: Pixel margin around the mesh

    Returns:
        PIL RGB image with faces colored by a gray-to-red gradient
    """
    verts = vertices.detach().cpu().float().numpy()
    face_idx = faces.detach().cpu().numpy()

    # Orthographic front view: project onto XY plane
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # Compute bounding box and scale to image
    draw_size = image_size - 2 * margin
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Uniform scaling to preserve aspect ratio
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = draw_size / max(x_range, y_range)

    # Center in image
    x_offset = margin + (draw_size - x_range * scale) / 2
    y_offset = margin + (draw_size - y_range * scale) / 2

    px = (x - x_min) * scale + x_offset
    py = (y_max - y) * scale + y_offset  # flip y for image coords

    # Compute per-face centroid depth for painter's algorithm sort
    tri_verts_z = z[face_idx]  # (F, 3)
    face_depth = tri_verts_z.mean(axis=1)  # (F,)

    # Sort back-to-front (smallest z = farthest in front-view, draw first)
    draw_order = np.argsort(-face_depth)

    img = Image.new("RGB", (image_size, image_size), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    for fi in draw_order:
        v0, v1, v2 = face_idx[fi]
        polygon = [(px[v0], py[v0]), (px[v1], py[v1]), (px[v2], py[v2])]

        val = float(face_values[fi])
        # Gray-to-red gradient: (60,60,60) at 0 -> (255,30,30) at 1
        r = int(60 + val * 195)
        g = int(60 - val * 30)
        b = int(60 - val * 30)
        draw.polygon(polygon, fill=(r, g, b))

    return img


def _compute_unique_edges(faces: Tensor) -> Tensor:
    """Extract deduplicated edges from face connectivity.

    Args:
        faces: Triangle face indices (F, 3)

    Returns:
        Unique edges as (E, 2) tensor with sorted vertex indices
    """
    f = faces.cpu().numpy()
    # Each triangle has 3 edges: (0,1), (1,2), (0,2)
    edges = np.concatenate(
        [
            f[:, [0, 1]],
            f[:, [1, 2]],
            f[:, [0, 2]],
        ],
        axis=0,
    )
    # Sort each edge so smaller index first, then deduplicate
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return torch.from_numpy(edges)


def render_wireframe_overlay(
    image: Tensor,
    vertices: Tensor,
    faces: Tensor,
    intrinsic: Tensor,
    extrinsic: Tensor,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Render SMPL wireframe overlay on top of an image.

    Projects posed vertices to 2D and draws mesh edges on a transparent
    overlay, then alpha-composites onto the ground truth image.

    Args:
        image: RGB image tensor (3, H, W) in [0, 1]
        vertices: Posed mesh vertices (V, 3) in world coordinates
        faces: Triangle face indices (F, 3)
        intrinsic: Camera intrinsic matrix (3, 3)
        extrinsic: Camera extrinsic matrix (4, 4)
        color: RGB color for wireframe lines
        alpha: Opacity of the wireframe overlay

    Returns:
        PIL RGB Image with wireframe drawn on top
    """
    H, W = image.shape[1], image.shape[2]

    # Project vertices: world -> camera -> pixel
    verts = vertices.detach().cpu().float()
    K = intrinsic.detach().cpu().float()
    E = extrinsic.detach().cpu().float()

    R = E[:3, :3]  # (3, 3)
    t = E[:3, 3]   # (3,)

    # Transform to camera space
    cam_pts = (R @ verts.T).T + t  # (V, 3)

    # Project to pixel coordinates
    z = cam_pts[:, 2]
    px = (K[0, 0] * cam_pts[:, 0] / z + K[0, 2]).numpy()
    py = (K[1, 1] * cam_pts[:, 1] / z + K[1, 2]).numpy()
    z_np = z.numpy()

    # Convert GT image to PIL
    img_np = (image.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
    base = Image.fromarray(img_np).convert("RGBA")

    # Create transparent overlay for wireframe
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    edges = _compute_unique_edges(faces)
    alpha_byte = int(alpha * 255)
    line_color = (*color, alpha_byte)

    for i0, i1 in edges.numpy():
        # Skip edges with endpoints behind the camera
        if z_np[i0] <= 0 or z_np[i1] <= 0:
            continue
        draw.line(
            [(px[i0], py[i0]), (px[i1], py[i1])],
            fill=line_color,
            width=1,
        )

    composited = Image.alpha_composite(base, overlay)
    return composited.convert("RGB")


def _build_density_colormap() -> np.ndarray:
    """Build a 256-entry blue-to-red colormap for density visualization.

    Returns:
        (256, 3) uint8 lookup table mapping scalar index to RGB color.
    """
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            # Blue (0) -> green (0.5)
            r = 0
            g = int(t * 2 * 255)
            b = int((1.0 - t * 2) * 255)
        else:
            # Green (0.5) -> red (1.0)
            r = int((t - 0.5) * 2 * 255)
            g = int((1.0 - (t - 0.5) * 2) * 255)
            b = 0
        lut[i] = [r, g, b]
    return lut


_DENSITY_COLORMAP = _build_density_colormap()


def _project_to_pixels(
    means: Tensor,
    intrinsic: Tensor,
    extrinsic: Tensor,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Project 3D Gaussian centers to 2D pixel coordinates.

    Args:
        means: Gaussian world-space positions (N, 3)
        intrinsic: Camera intrinsic matrix (3, 3)
        extrinsic: Camera extrinsic matrix (4, 4)
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Tuple of (px_int, py_int, valid_mask) as numpy arrays where
        px_int/py_int are clipped pixel coordinates for valid (in-front-of-camera)
        points, and valid_mask is the boolean mask into the original N points.
        Returns None if no points are in front of the camera.
    """
    verts = means.detach().cpu().float()
    K = intrinsic.detach().cpu().float()
    E = extrinsic.detach().cpu().float()

    R = E[:3, :3]
    t = E[:3, 3]
    cam_pts = (R @ verts.T).T + t  # (N, 3)
    z = cam_pts[:, 2]

    valid = z > 0.01
    if valid.sum() == 0:
        return None

    px = (K[0, 0] * cam_pts[valid, 0] / z[valid] + K[0, 2]).numpy()
    py = (K[1, 1] * cam_pts[valid, 1] / z[valid] + K[1, 2]).numpy()

    px_int = np.clip(np.round(px).astype(np.intp), 0, width - 1)
    py_int = np.clip(np.round(py).astype(np.intp), 0, height - 1)

    return px_int, py_int, valid.numpy()


def _render_heatmap(
    data: np.ndarray,
    height: int,
    width: int,
    blur_radius: int,
) -> Image.Image:
    """Normalize a 2D scalar map and render with the blue-to-red colormap.

    Args:
        data: (H, W) float32 scalar map (pre-blur)
        height: Image height
        width: Image width
        blur_radius: Gaussian blur sigma

    Returns:
        PIL RGB image
    """
    data = gaussian_filter(data, sigma=blur_radius)

    max_val = data.max()
    if max_val > 0:
        data = data / max_val

    data_uint8 = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    rgb = _DENSITY_COLORMAP[data_uint8]

    bg_mask = data < 1e-6
    rgb[bg_mask] = [30, 30, 30]

    return Image.fromarray(rgb)


def render_projected_density(
    means: Tensor,
    intrinsic: Tensor,
    extrinsic: Tensor,
    height: int,
    width: int,
    blur_radius: int = 8,
) -> Image.Image:
    """Render a 2D density heatmap of projected Gaussian centers.

    Projects all Gaussian centers into the camera view and creates a
    smoothed density heatmap showing where Gaussians concentrate in
    screen space.

    Args:
        means: Gaussian world-space positions (N, 3)
        intrinsic: Camera intrinsic matrix (3, 3)
        extrinsic: Camera extrinsic matrix (4, 4)
        height: Image height in pixels
        width: Image width in pixels
        blur_radius: Gaussian blur radius for smoothing

    Returns:
        PIL RGB image with blue-to-red density heatmap
    """
    proj = _project_to_pixels(means, intrinsic, extrinsic, height, width)
    if proj is None:
        return Image.new("RGB", (width, height), (30, 30, 30))

    px_int, py_int, _valid = proj

    density = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (py_int, px_int), 1.0)

    return _render_heatmap(density, height, width, blur_radius)


def render_projected_values(
    means: Tensor,
    values: Tensor,
    intrinsic: Tensor,
    extrinsic: Tensor,
    height: int,
    width: int,
    blur_radius: int = 4,
) -> Image.Image:
    """Render a 2D heatmap of projected Gaussian centers colored by per-Gaussian values.

    Projects all Gaussian centers into the camera view. At each pixel,
    takes the maximum value among all Gaussians projecting there, then
    applies Gaussian blur and a blue-to-red colormap.

    Args:
        means: Gaussian world-space positions (N, 3)
        values: Per-Gaussian scalar values (N,), e.g. offset magnitudes
        intrinsic: Camera intrinsic matrix (3, 3)
        extrinsic: Camera extrinsic matrix (4, 4)
        height: Image height in pixels
        width: Image width in pixels
        blur_radius: Gaussian blur radius for smoothing

    Returns:
        PIL RGB image with blue-to-red value heatmap
    """
    proj = _project_to_pixels(means, intrinsic, extrinsic, height, width)
    if proj is None:
        return Image.new("RGB", (width, height), (30, 30, 30))

    px_int, py_int, valid_mask = proj
    valid_vals = values.detach().cpu().float().numpy()[valid_mask]

    value_map = np.zeros((height, width), dtype=np.float32)
    np.maximum.at(value_map, (py_int, px_int), valid_vals)

    return _render_heatmap(value_map, height, width, blur_radius)
