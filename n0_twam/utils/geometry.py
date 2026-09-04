"""Small, differentiable camera-geometry helpers for RGB motion detection.

The functions in this module deliberately do not expose 3D coordinates as model
token metadata. 3D is used only transiently to compensate camera ego motion before
comparing two RGB-D observations.

Unless stated otherwise, camera poses use the ``world_from_camera`` convention:
``p_world = pose @ p_camera``.  ``camera_from_world`` is also supported explicitly
to make pose conventions hard to confuse at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F


PoseConvention = Literal["world_from_camera", "camera_from_world"]


def _require_floating(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating point, got {tensor.dtype}.")


def _as_depth_map(depth: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a depth tensor to ``(B, H, W)`` and validate it."""

    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    _require_floating(depth, name)
    if depth.ndim == 4:
        if depth.shape[1] != 1:
            raise ValueError(
                f"{name} with four dimensions must have shape (B, 1, H, W), "
                f"got {tuple(depth.shape)}."
            )
        depth = depth[:, 0]
    if depth.ndim != 3:
        raise ValueError(
            f"{name} must have shape (B, H, W) or (B, 1, H, W), "
            f"got {tuple(depth.shape)}."
        )
    if depth.shape[0] < 1 or depth.shape[1] < 1 or depth.shape[2] < 1:
        raise ValueError(f"{name} cannot contain an empty dimension.")
    return depth


def _as_batched_matrix(
    matrix: torch.Tensor,
    batch_size: int,
    matrix_shape: tuple[int, int],
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    _require_floating(matrix, name)
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 3 or tuple(matrix.shape[-2:]) != matrix_shape:
        raise ValueError(
            f"{name} must have shape {matrix_shape} or (B, {matrix_shape[0]}, "
            f"{matrix_shape[1]}), got {tuple(matrix.shape)}."
        )
    if matrix.shape[0] not in (1, batch_size):
        raise ValueError(
            f"{name} batch dimension must be 1 or {batch_size}, "
            f"got {matrix.shape[0]}."
        )
    matrix = matrix.to(device=device, dtype=dtype)
    if matrix.shape[0] == 1 and batch_size != 1:
        matrix = matrix.expand(batch_size, -1, -1)
    return matrix


def make_normalized_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an ``align_corners=False`` sampling grid of shape ``(H, W, 2)``."""

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}.")
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def normalized_to_pixel(
    normalized_xy: torch.Tensor, image_size: Sequence[int]
) -> torch.Tensor:
    """Convert an ``align_corners=False`` normalized grid to pixel coordinates."""

    if normalized_xy.shape[-1] != 2:
        raise ValueError(
            "normalized_xy must end in an (x, y) pair, got "
            f"{tuple(normalized_xy.shape)}."
        )
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {(height, width)}.")
    x = (normalized_xy[..., 0] + 1.0) * width * 0.5 - 0.5
    y = (normalized_xy[..., 1] + 1.0) * height * 0.5 - 0.5
    return torch.stack((x, y), dim=-1)


def pixel_to_normalized(
    pixel_xy: torch.Tensor, image_size: Sequence[int]
) -> torch.Tensor:
    """Convert pixel coordinates to an ``align_corners=False`` sampling grid."""

    if pixel_xy.shape[-1] != 2:
        raise ValueError(
            f"pixel_xy must end in an (x, y) pair, got {tuple(pixel_xy.shape)}."
        )
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {(height, width)}.")
    x = (pixel_xy[..., 0] + 0.5) * (2.0 / width) - 1.0
    y = (pixel_xy[..., 1] + 0.5) * (2.0 / height) - 1.0
    return torch.stack((x, y), dim=-1)


def sample_at_grid(
    image: torch.Tensor,
    normalized_grid: torch.Tensor,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
) -> torch.Tensor:
    """Sample ``(B,C,H,W)`` at an ``(B,Hout,Wout,2)`` normalized grid."""

    if image.ndim != 4:
        raise ValueError(
            f"image must have shape (B, C, H, W), got {tuple(image.shape)}."
        )
    if normalized_grid.ndim != 4 or normalized_grid.shape[-1] != 2:
        raise ValueError(
            "normalized_grid must have shape (B, H, W, 2), got "
            f"{tuple(normalized_grid.shape)}."
        )
    if normalized_grid.shape[0] != image.shape[0]:
        raise ValueError(
            "image and normalized_grid batch sizes differ: "
            f"{image.shape[0]} vs {normalized_grid.shape[0]}."
        )
    return F.grid_sample(
        image,
        normalized_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )


def backproject_pixels(
    pixel_xy: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Backproject pixel/depth pairs to camera coordinates.

    Args:
        pixel_xy: ``(B, ..., 2)`` coordinates in pixels.
        depth: ``(B, ...)`` z-depth in camera units.
        intrinsics: ``(3,3)`` or ``(B,3,3)`` camera matrix.
    """

    if pixel_xy.ndim < 3 or pixel_xy.shape[-1] != 2:
        raise ValueError(
            f"pixel_xy must have shape (B, ..., 2), got {tuple(pixel_xy.shape)}."
        )
    if depth.shape != pixel_xy.shape[:-1]:
        raise ValueError(
            "depth must match pixel_xy without its coordinate dimension: "
            f"depth={tuple(depth.shape)}, pixels={tuple(pixel_xy.shape)}."
        )
    _require_floating(pixel_xy, "pixel_xy")
    _require_floating(depth, "depth")
    batch_size = pixel_xy.shape[0]
    intrinsics = _as_batched_matrix(
        intrinsics,
        batch_size,
        (3, 3),
        "intrinsics",
        device=pixel_xy.device,
        dtype=pixel_xy.dtype,
    )
    expand_dims = (1,) * (pixel_xy.ndim - 2)
    fx = intrinsics[:, 0, 0].view(batch_size, *expand_dims)
    fy = intrinsics[:, 1, 1].view(batch_size, *expand_dims)
    cx = intrinsics[:, 0, 2].view(batch_size, *expand_dims)
    cy = intrinsics[:, 1, 2].view(batch_size, *expand_dims)
    if torch.any(fx == 0) or torch.any(fy == 0):
        raise ValueError("intrinsics focal lengths fx and fy must be non-zero.")
    z = depth
    x = (pixel_xy[..., 0] - cx) * z / fx
    y = (pixel_xy[..., 1] - cy) * z / fy
    return torch.stack((x, y, z), dim=-1)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous transform to batched ``(..., 3)`` points."""

    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError(
            f"points must have shape (B, ..., 3), got {tuple(points.shape)}."
        )
    _require_floating(points, "points")
    transform = _as_batched_matrix(
        transform,
        points.shape[0],
        (4, 4),
        "transform",
        device=points.device,
        dtype=points.dtype,
    )
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]
    flat = points.reshape(points.shape[0], -1, 3)
    transformed = torch.bmm(flat, rotation.transpose(1, 2))
    transformed = transformed + translation[:, None, :]
    return transformed.reshape_as(points)


def project_points(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    min_depth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-space points, returning ``(pixel_xy, z, positive_z)``."""

    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError(
            f"points must have shape (B, ..., 3), got {tuple(points.shape)}."
        )
    if min_depth <= 0:
        raise ValueError(f"min_depth must be positive, got {min_depth}.")
    _require_floating(points, "points")
    batch_size = points.shape[0]
    intrinsics = _as_batched_matrix(
        intrinsics,
        batch_size,
        (3, 3),
        "intrinsics",
        device=points.device,
        dtype=points.dtype,
    )
    expand_dims = (1,) * (points.ndim - 2)
    fx = intrinsics[:, 0, 0].view(batch_size, *expand_dims)
    fy = intrinsics[:, 1, 1].view(batch_size, *expand_dims)
    cx = intrinsics[:, 0, 2].view(batch_size, *expand_dims)
    cy = intrinsics[:, 1, 2].view(batch_size, *expand_dims)
    z = points[..., 2]
    valid = torch.isfinite(points).all(dim=-1) & (z > min_depth)
    safe_z = torch.where(valid, z, torch.ones_like(z))
    u = fx * points[..., 0] / safe_z + cx
    v = fy * points[..., 1] / safe_z + cy
    return torch.stack((u, v), dim=-1), z, valid


def relative_camera_transform(
    pose_previous: torch.Tensor,
    pose_current: torch.Tensor,
    *,
    convention: PoseConvention = "world_from_camera",
) -> torch.Tensor:
    """Return the transform from current-camera to previous-camera coordinates."""

    if convention not in ("world_from_camera", "camera_from_world"):
        raise ValueError(
            "convention must be 'world_from_camera' or 'camera_from_world', "
            f"got {convention!r}."
        )
    if pose_previous.ndim == 2:
        pose_previous = pose_previous.unsqueeze(0)
    if pose_current.ndim == 2:
        pose_current = pose_current.unsqueeze(0)
    if pose_previous.ndim != 3 or pose_previous.shape[-2:] != (4, 4):
        raise ValueError(
            "pose_previous must have shape (4,4) or (B,4,4), got "
            f"{tuple(pose_previous.shape)}."
        )
    if pose_current.ndim != 3 or pose_current.shape[-2:] != (4, 4):
        raise ValueError(
            "pose_current must have shape (4,4) or (B,4,4), got "
            f"{tuple(pose_current.shape)}."
        )
    _require_floating(pose_previous, "pose_previous")
    _require_floating(pose_current, "pose_current")
    batch_size = max(pose_previous.shape[0], pose_current.shape[0])
    device = pose_current.device
    dtype = pose_current.dtype
    pose_previous = _as_batched_matrix(
        pose_previous,
        batch_size,
        (4, 4),
        "pose_previous",
        device=device,
        dtype=dtype,
    )
    pose_current = _as_batched_matrix(
        pose_current,
        batch_size,
        (4, 4),
        "pose_current",
        device=device,
        dtype=dtype,
    )
    if convention == "world_from_camera":
        return torch.linalg.inv(pose_previous) @ pose_current
    return pose_previous @ torch.linalg.inv(pose_current)


@dataclass(frozen=True)
class ReprojectionResult:
    """Backward correspondence from a current grid into the previous image."""

    previous_pixel_xy: torch.Tensor
    previous_normalized_xy: torch.Tensor
    previous_camera_depth: torch.Tensor
    current_depth: torch.Tensor
    valid: torch.Tensor


def reproject_current_grid_to_previous(
    depth_current: torch.Tensor,
    intrinsics_previous: torch.Tensor,
    pose_previous: torch.Tensor,
    pose_current: torch.Tensor,
    output_size: Sequence[int],
    *,
    intrinsics_current: torch.Tensor | None = None,
    pose_convention: PoseConvention = "world_from_camera",
    min_depth: float = 1e-6,
) -> ReprojectionResult:
    """Map a regular current-frame grid into the previous frame.

    ``output_size`` is normally the DINO patch-grid size. Current depth is sampled
    at each patch centre, backprojected, transformed by the camera poses, and then
    projected into the previous image. The returned normalized coordinates are in
    full-image coordinates and can therefore sample either previous depth or a
    uniformly aligned DINO feature grid with ``align_corners=False``.
    """

    depth_current = _as_depth_map(depth_current, "depth_current")
    if min_depth <= 0:
        raise ValueError(f"min_depth must be positive, got {min_depth}.")
    out_h, out_w = int(output_size[0]), int(output_size[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"output_size must be positive, got {(out_h, out_w)}.")

    batch_size, image_h, image_w = depth_current.shape
    compute_dtype = depth_current.dtype
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    current_depth_image = depth_current.to(dtype=compute_dtype).unsqueeze(1)
    normalized_current = (
        make_normalized_grid(
            out_h,
            out_w,
            device=depth_current.device,
            dtype=compute_dtype,
        )
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
    )
    current_pixel_xy = normalized_to_pixel(normalized_current, (image_h, image_w))
    sampled_current_depth = sample_at_grid(
        current_depth_image, normalized_current, mode="nearest"
    )[:, 0]

    intrinsics_current = (
        intrinsics_previous if intrinsics_current is None else intrinsics_current
    )
    current_points = backproject_pixels(
        current_pixel_xy, sampled_current_depth, intrinsics_current
    )
    current_to_previous = relative_camera_transform(
        pose_previous,
        pose_current,
        convention=pose_convention,
    ).to(device=depth_current.device, dtype=compute_dtype)
    previous_points = transform_points(current_points, current_to_previous)
    previous_pixel_xy, previous_z, positive_z = project_points(
        previous_points, intrinsics_previous, min_depth=min_depth
    )
    previous_normalized_xy = pixel_to_normalized(previous_pixel_xy, (image_h, image_w))

    u = previous_pixel_xy[..., 0]
    v = previous_pixel_xy[..., 1]
    in_image = (u >= 0.0) & (u <= image_w - 1) & (v >= 0.0) & (v <= image_h - 1)
    current_valid = torch.isfinite(sampled_current_depth) & (
        sampled_current_depth > min_depth
    )
    valid = current_valid & positive_z & in_image
    return ReprojectionResult(
        previous_pixel_xy=previous_pixel_xy,
        previous_normalized_xy=previous_normalized_xy,
        previous_camera_depth=previous_z,
        current_depth=sampled_current_depth,
        valid=valid,
    )


__all__ = [
    "PoseConvention",
    "ReprojectionResult",
    "backproject_pixels",
    "make_normalized_grid",
    "normalized_to_pixel",
    "pixel_to_normalized",
    "project_points",
    "relative_camera_transform",
    "reproject_current_grid_to_previous",
    "sample_at_grid",
    "transform_points",
]
