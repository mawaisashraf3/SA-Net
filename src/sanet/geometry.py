"""Coordinate checks and pixel/metric geometry. Ego axes: forward, left, up."""

import torch


def validate_calibration(k: torch.Tensor, transform: torch.Tensor) -> None:
    if k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("K must be 3x3 and T_ego_from_camera 4x4")
    if not torch.isfinite(k).all() or not torch.isfinite(transform).all():
        raise ValueError("Calibration contains nonfinite values")
    if abs(torch.linalg.det(k).item()) < 1e-8:
        raise ValueError("Singular intrinsics")
    r = transform[:3, :3]
    if not torch.allclose(r.T @ r, torch.eye(3, device=r.device), atol=1e-3):
        raise ValueError("Extrinsic rotation must be orthonormal")
    if abs(torch.linalg.det(r).item() - 1) > 1e-3:
        raise ValueError("Extrinsic rotation must have determinant +1")
    if not torch.allclose(transform[3], transform.new_tensor([0, 0, 0, 1]), atol=1e-6):
        raise ValueError("Invalid homogeneous transform bottom row")


def backproject(
    uv: torch.Tensor, depth: torch.Tensor, k: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """Camera-z depth, full processed-image pixel centers; accepts arbitrary leading shape."""
    q = torch.cat([uv, torch.ones_like(uv[..., :1])], -1)
    xyz = (q @ torch.linalg.inv(k).T) * depth[..., None]
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def project(points: torch.Tensor, k: torch.Tensor, transform: torch.Tensor):
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    uvw = camera @ k.T
    uv = uvw[..., :2] / uvw[..., 2:].clamp_min(1e-8)
    return uv, camera[..., 2]


def resize_matrix(
    old_height: int, old_width: int, height: int, width: int, crop_top: int = 0
) -> torch.Tensor:
    if crop_top >= old_height:
        raise ValueError("Crop removes entire image")
    sx, sy = width / old_width, height / (old_height - crop_top)
    # Center-aligned resize: new=(old+.5)*scale-.5.
    return torch.tensor(
        [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2 - sy * crop_top], [0, 0, 1]],
        dtype=torch.float32,
    )
