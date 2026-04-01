"""Self-supervised loss for novel-pose regularization.

Perturbs training poses and camera views, then applies AIAP isometry
and soft silhouette losses to improve generalization to unseen poses.
"""

import cv2
import numpy as np
import torch
from torch import Tensor


def perturb_pose(
    pose: Tensor, noise_std: float = 0.15, global_noise_scale: float = 0.3
) -> Tensor:
    """Add Gaussian noise to pose parameters.

    Applies reduced noise to the global orientation (first 3 dims)
    relative to body joints.

    Args:
        pose: (P,) pose parameters
        noise_std: Noise standard deviation (radians for axis-angle)
        global_noise_scale: Multiplier for first 3 dims

    Returns:
        Perturbed pose (P,)
    """
    noise = torch.randn_like(pose) * noise_std
    noise[:3] *= global_noise_scale
    return pose + noise


def jitter_camera(
    extrinsic: Tensor,
    rotation_std: float = 0.02,
    translation_std: float = 0.05,
) -> Tensor:
    """Apply small random perturbation to camera extrinsic.

    Args:
        extrinsic: (4, 4) world-to-camera transform
        rotation_std: Rotation noise std (radians)
        translation_std: Translation noise std (meters)

    Returns:
        Jittered (4, 4) extrinsic
    """
    device, dtype = extrinsic.device, extrinsic.dtype

    # Small rotation via Rodrigues formula
    aa = torch.randn(3, device=device, dtype=dtype) * rotation_std
    theta = aa.norm()
    if theta > 1e-8:
        k = aa / theta
        K = torch.zeros(3, 3, device=device, dtype=dtype)
        K[0, 1], K[0, 2] = -k[2], k[1]
        K[1, 0], K[1, 2] = k[2], -k[0]
        K[2, 0], K[2, 1] = -k[1], k[0]
        dR = (
            torch.eye(3, device=device, dtype=dtype)
            + theta.sin() * K
            + (1 - theta.cos()) * (K @ K)
        )
    else:
        dR = torch.eye(3, device=device, dtype=dtype)

    dt = torch.randn(3, device=device, dtype=dtype) * translation_std

    out = extrinsic.clone()
    out[:3, :3] = dR @ extrinsic[:3, :3]
    out[:3, 3] = extrinsic[:3, 3] + dt
    return out


def rasterize_mesh_silhouette(
    vertices: Tensor,
    faces: Tensor,
    intrinsic: Tensor,
    extrinsic: Tensor,
    height: int,
    width: int,
) -> Tensor:
    """Rasterize mesh into a binary silhouette mask.

    Projects mesh vertices with the given camera and fills visible
    triangles using OpenCV.

    Args:
        vertices: (V, 3) world-space vertices
        faces: (F, 3) triangle indices
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) world-to-camera extrinsic
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        (H, W) float tensor (1 inside mesh, 0 outside)
    """
    device = vertices.device

    # Project to camera space
    V_cam = (extrinsic[:3, :3] @ vertices.T + extrinsic[:3, 3:4]).T  # (V, 3)
    z = V_cam[:, 2]

    # Keep only faces with all vertices in front of camera
    face_z = z[faces]  # (F, 3)
    valid = (face_z > 0.1).all(dim=1)

    # Project to pixel coords
    z_safe = z.clamp(min=0.1).unsqueeze(1)  # (V, 1)
    V_proj = (intrinsic @ V_cam.T).T  # (V, 3)
    V_2d = V_proj[:, :2] / z_safe  # (V, 2)

    # Rasterize with OpenCV
    verts_np = V_2d.detach().cpu().numpy().astype(np.int32)
    valid_faces_np = faces[valid].detach().cpu().numpy()

    mask_np = np.zeros((height, width), dtype=np.uint8)
    if len(valid_faces_np) > 0:
        triangles = verts_np[valid_faces_np]  # (F', 3, 2)
        cv2.fillPoly(mask_np, list(triangles), 1)

    return torch.from_numpy(mask_np.astype(np.float32)).to(device)


def compute_ssl_silhouette_loss(
    rendered_alpha: Tensor,
    mesh_mask: Tensor,
    margin_pixels: float = 3.0,
    temperature: float = 2.0,
) -> Tensor:
    """Soft silhouette loss for SSL regularization.

    Inside the mesh mask, encourages alpha to be saturated (1).
    Outside the mask, penalizes alpha with increasing weight as
    distance from the boundary grows.

    Args:
        rendered_alpha: (H, W) rendered opacity
        mesh_mask: (H, W) binary mesh silhouette (1 inside)
        margin_pixels: Slack zone outside boundary (pixels)
        temperature: Sharpness of the soft falloff

    Returns:
        Scalar loss
    """
    # Distance field: 0 inside mask, positive outside
    mask_np = mesh_mask.detach().cpu().numpy().astype(np.uint8)
    inv = (1 - mask_np).astype(np.uint8)
    dist_np = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    distance = torch.from_numpy(dist_np).to(rendered_alpha.device)

    # Inside: encourage saturation
    inside_loss = ((1.0 - rendered_alpha) * mesh_mask).mean()

    # Outside: soft distance-based penalty
    outside = 1.0 - mesh_mask
    weight = torch.sigmoid((distance - margin_pixels) / temperature)
    outside_loss = (rendered_alpha * weight * outside).mean()

    return inside_loss + outside_loss
