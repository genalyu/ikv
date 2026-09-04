"""Sparse RGB-motion front end for N0-TWAM.

This module stops immediately before the existing N0-TWAM input projection. It
selects raw WAN latent patches and carries a separate semantic index alongside
them. In particular, :class:`TokenIndex` is metadata and is never added to, or
concatenated with, the WAN patch values here.

The intended integration is::

    dense WAN latent -> SparsePatchGather -> existing patch_embedding_mlp
                                             (unchanged)
    TokenIndex       -> cache addressing / replacement / compression

Depth and pose are used only by :class:`EgoMotionCompensatedMotionDetector` to
remove apparent motion caused by the camera itself. They are not token indices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from n0_twam.utils.geometry import (
    PoseConvention,
    reproject_current_grid_to_previous,
    sample_at_grid,
)


def _positive_grid_size(size: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in size)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive dimensions, got {result}.")
    return result


def _dino_to_channels_first(
    tokens: torch.Tensor,
    grid_size: Sequence[int] | None,
    name: str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Accept ``(B,H,W,D)`` or flattened ``(B,N,D)`` DINO patch tokens."""

    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_floating_point(tokens):
        raise TypeError(f"{name} must be floating point, got {tokens.dtype}.")
    if tokens.ndim == 4:
        if grid_size is not None and tuple(tokens.shape[1:3]) != tuple(grid_size):
            raise ValueError(
                f"{name} grid is {tuple(tokens.shape[1:3])}, but grid_size is "
                f"{tuple(grid_size)}."
            )
        if tokens.shape[-1] < 1:
            raise ValueError(f"{name} feature dimension cannot be empty.")
        height, width = int(tokens.shape[1]), int(tokens.shape[2])
        return tokens.permute(0, 3, 1, 2).contiguous(), (height, width)
    if tokens.ndim == 3:
        if grid_size is None:
            raise ValueError(
                f"grid_size is required when {name} has flattened shape (B,N,D)."
            )
        normalized_grid = _positive_grid_size(grid_size, "grid_size")
        if len(normalized_grid) != 2:
            raise ValueError(
                f"grid_size must have two dimensions, got {normalized_grid}."
            )
        height, width = normalized_grid
        if tokens.shape[1] != height * width:
            raise ValueError(
                f"{name} has {tokens.shape[1]} tokens, expected {height * width} "
                f"for grid {(height, width)}."
            )
        if tokens.shape[-1] < 1:
            raise ValueError(f"{name} feature dimension cannot be empty.")
        mapped = tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])
        return mapped.permute(0, 3, 1, 2).contiguous(), (height, width)
    raise ValueError(
        f"{name} must have shape (B,H,W,D) or (B,N,D), got {tuple(tokens.shape)}."
    )


def _resize_max(values: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    source_size = tuple(values.shape[-2:])
    if source_size == target_size:
        return values
    if target_size[0] <= source_size[0] and target_size[1] <= source_size[1]:
        return F.adaptive_max_pool2d(values, target_size)
    return F.interpolate(values, size=target_size, mode="bilinear", align_corners=False)


def _resize_any_mask(mask: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    source_size = tuple(mask.shape[-2:])
    values = mask.to(torch.float32)
    if source_size == target_size:
        return mask
    if target_size[0] <= source_size[0] and target_size[1] <= source_size[1]:
        values = F.adaptive_max_pool2d(values, target_size)
    else:
        values = F.interpolate(values, size=target_size, mode="nearest")
    return values > 0.5


def pool_dino_to_grid(
    dino_tokens: torch.Tensor,
    target_size: Sequence[int],
    *,
    source_grid_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Pool DINO tokens to a new grid and return channels-last features.

    Average pooling is used when reducing 16x16 DINO features to the usual 8x8
    WAN-transformer grid. Bilinear interpolation handles non-integer or larger
    target grids.
    """

    mapped, _ = _dino_to_channels_first(dino_tokens, source_grid_size, "dino_tokens")
    target = _positive_grid_size(target_size, "target_size")
    if len(target) != 2:
        raise ValueError(f"target_size must have two dimensions, got {target}.")
    source = tuple(mapped.shape[-2:])
    if source != target:
        if target[0] <= source[0] and target[1] <= source[1]:
            mapped = F.adaptive_avg_pool2d(mapped, target)
        else:
            mapped = F.interpolate(
                mapped, size=target, mode="bilinear", align_corners=False
            )
    return mapped.permute(0, 2, 3, 1).contiguous()


def padded_indices_from_mask(
    mask: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    max_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a batched grid mask to padded flattened indices.

    Invalid padded positions use index ``-1`` and are identified by the returned
    boolean validity mask. If ``scores`` is supplied, selected entries are ordered
    by descending score; otherwise flattened grid order is preserved.
    """

    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        shape = tuple(mask.shape) if isinstance(mask, torch.Tensor) else None
        raise ValueError(f"mask must have shape (B, ...), got {shape}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype bool, got {mask.dtype}.")
    if max_tokens is not None and max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative, got {max_tokens}.")
    flat_mask = mask.reshape(mask.shape[0], -1)
    flat_scores = None
    if scores is not None:
        if not isinstance(scores, torch.Tensor) or scores.shape != mask.shape:
            score_shape = (
                tuple(scores.shape) if isinstance(scores, torch.Tensor) else None
            )
            raise ValueError(
                f"scores must match mask shape {tuple(mask.shape)}, got {score_shape}."
            )
        if not torch.is_floating_point(scores):
            raise TypeError(f"scores must be floating point, got {scores.dtype}.")
        if scores.device != mask.device:
            raise ValueError("scores and mask must be on the same device.")
        flat_scores = scores.reshape(scores.shape[0], -1)

    selected: list[torch.Tensor] = []
    for batch_index in range(flat_mask.shape[0]):
        indices = torch.nonzero(flat_mask[batch_index], as_tuple=False).flatten()
        if flat_scores is not None and indices.numel() > 0:
            candidate_scores = flat_scores[batch_index, indices]
            order = torch.argsort(candidate_scores, descending=True, stable=True)
            indices = indices[order]
        if max_tokens is not None:
            indices = indices[:max_tokens]
        selected.append(indices)

    padded_length = max((item.numel() for item in selected), default=0)
    indices_out = torch.full(
        (mask.shape[0], padded_length),
        -1,
        dtype=torch.long,
        device=mask.device,
    )
    valid_out = torch.zeros(
        (mask.shape[0], padded_length), dtype=torch.bool, device=mask.device
    )
    for batch_index, indices in enumerate(selected):
        count = indices.numel()
        if count:
            indices_out[batch_index, :count] = indices
            valid_out[batch_index, :count] = True
    return indices_out, valid_out


def _mask_from_padded_indices(
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    batch_size = indices.shape[0]
    flat = torch.zeros(
        batch_size,
        grid_size[0] * grid_size[1],
        dtype=torch.bool,
        device=indices.device,
    )
    for batch_index in range(batch_size):
        selected = indices[batch_index, valid_mask[batch_index]]
        if selected.numel():
            flat[batch_index, selected] = True
    return flat.reshape(batch_size, *grid_size)


@dataclass(frozen=True)
class MotionDetectionResult:
    """Motion outputs on both the DINO and WAN-transformer patch grids."""

    dino_motion_score: torch.Tensor
    dino_motion_mask: torch.Tensor
    dino_correspondence_valid: torch.Tensor
    depth_residual: torch.Tensor
    dino_distance: torch.Tensor
    wan_motion_score: torch.Tensor
    wan_motion_mask: torch.Tensor
    wan_moving_indices: torch.Tensor
    wan_indices_valid: torch.Tensor
    wan_dino_features: torch.Tensor

    @property
    def motion_mask(self) -> torch.Tensor:
        """Final, token-budgeted mask consumed by WAN patch gather."""

        return self.wan_motion_mask

    @property
    def moving_patch_indices(self) -> torch.Tensor:
        """Final padded flattened indices on the WAN transformer grid."""

        return self.wan_moving_indices

    @property
    def moving_patch_valid(self) -> torch.Tensor:
        """Validity mask for :attr:`moving_patch_indices`."""

        return self.wan_indices_valid

    def gather_current_dino(self) -> torch.Tensor:
        """Gather the current DINO index component at every selected WAN patch."""

        return gather_grid_features(
            self.wan_dino_features,
            self.wan_moving_indices,
            self.wan_indices_valid,
        )


class EgoMotionCompensatedMotionDetector(nn.Module):
    """Find moving DINO patches after compensating known camera motion.

    DINO inputs are either channels-last ``(B,Hd,Wd,D)`` maps or flattened
    ``(B,Hd*Wd,D)`` sequences (the latter requires ``dino_grid_size``). Camera
    intrinsics are for the depth/RGB image resolution, not for the DINO grid.

    A patch is moving if its reprojected depth residual exceeds
    ``depth_threshold`` or its current/warped DINO cosine distance exceeds
    ``dino_threshold``. Invalid/out-of-view correspondences are not labelled as
    motion; they remain unavailable evidence rather than false positive motion.
    """

    def __init__(
        self,
        *,
        depth_threshold: float = 0.02,
        dino_threshold: float = 0.2,
        depth_weight: float = 1.0,
        dino_weight: float = 1.0,
        dilation_radius: int = 1,
        max_tokens: int | None = None,
        min_depth: float = 1e-6,
        pose_convention: PoseConvention = "world_from_camera",
    ) -> None:
        super().__init__()
        if depth_threshold <= 0 or dino_threshold <= 0:
            raise ValueError("depth_threshold and dino_threshold must be positive.")
        if depth_weight < 0 or dino_weight < 0:
            raise ValueError("depth_weight and dino_weight must be non-negative.")
        if depth_weight == 0 and dino_weight == 0:
            raise ValueError("At least one motion score weight must be non-zero.")
        if dilation_radius < 0:
            raise ValueError("dilation_radius must be non-negative.")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be non-negative when provided.")
        if min_depth <= 0:
            raise ValueError("min_depth must be positive.")
        if pose_convention not in ("world_from_camera", "camera_from_world"):
            raise ValueError(
                "pose_convention must be 'world_from_camera' or " "'camera_from_world'."
            )
        self.depth_threshold = float(depth_threshold)
        self.dino_threshold = float(dino_threshold)
        self.depth_weight = float(depth_weight)
        self.dino_weight = float(dino_weight)
        self.dilation_radius = int(dilation_radius)
        self.max_tokens = max_tokens
        self.min_depth = float(min_depth)
        self.pose_convention = pose_convention

    def forward(
        self,
        dino_previous: torch.Tensor,
        dino_current: torch.Tensor,
        depth_previous: torch.Tensor,
        depth_current: torch.Tensor,
        camera_pose_previous: torch.Tensor,
        camera_pose_current: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        *,
        dino_grid_size: Sequence[int] | None = None,
        wan_grid_size: Sequence[int] | None = None,
        camera_intrinsics_current: torch.Tensor | None = None,
    ) -> MotionDetectionResult:
        previous_map, previous_grid = _dino_to_channels_first(
            dino_previous, dino_grid_size, "dino_previous"
        )
        current_map, current_grid = _dino_to_channels_first(
            dino_current, dino_grid_size, "dino_current"
        )
        if previous_grid != current_grid or previous_map.shape != current_map.shape:
            raise ValueError(
                "dino_previous and dino_current must have identical shapes/grids, "
                f"got {tuple(previous_map.shape)} and {tuple(current_map.shape)}."
            )
        if previous_map.device != current_map.device:
            raise ValueError("DINO tensors must be on the same device.")
        if depth_previous.ndim == 4 and depth_previous.shape[1] == 1:
            previous_depth_map = depth_previous[:, 0]
        elif depth_previous.ndim == 3:
            previous_depth_map = depth_previous
        else:
            raise ValueError(
                "depth_previous must have shape (B,H,W) or (B,1,H,W), got "
                f"{tuple(depth_previous.shape)}."
            )
        if depth_current.ndim == 4 and depth_current.shape[1] == 1:
            current_depth_map = depth_current[:, 0]
        elif depth_current.ndim == 3:
            current_depth_map = depth_current
        else:
            raise ValueError(
                "depth_current must have shape (B,H,W) or (B,1,H,W), got "
                f"{tuple(depth_current.shape)}."
            )
        if not torch.is_floating_point(
            previous_depth_map
        ) or not torch.is_floating_point(current_depth_map):
            raise TypeError("depth_previous and depth_current must be floating point.")
        if previous_depth_map.shape != current_depth_map.shape:
            raise ValueError(
                "depth_previous and depth_current must have identical shapes, got "
                f"{tuple(previous_depth_map.shape)} and "
                f"{tuple(current_depth_map.shape)}."
            )
        if previous_depth_map.shape[0] != previous_map.shape[0]:
            raise ValueError(
                "DINO/depth batch sizes differ: "
                f"{previous_map.shape[0]} vs {previous_depth_map.shape[0]}."
            )
        if previous_depth_map.device != previous_map.device:
            raise ValueError("DINO and depth tensors must be on the same device.")

        # Geometric work is kept in float32 for CPU half/bfloat16 compatibility.
        compute_dtype = previous_map.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        previous_features = previous_map.to(compute_dtype)
        current_features = current_map.to(compute_dtype)
        previous_depth = previous_depth_map.to(compute_dtype)
        current_depth = current_depth_map.to(compute_dtype)

        reprojection = reproject_current_grid_to_previous(
            current_depth,
            camera_intrinsics,
            camera_pose_previous,
            camera_pose_current,
            current_grid,
            intrinsics_current=camera_intrinsics_current,
            pose_convention=self.pose_convention,
            min_depth=self.min_depth,
        )
        warped_previous_depth = sample_at_grid(
            previous_depth.unsqueeze(1),
            reprojection.previous_normalized_xy,
            mode="bilinear",
        )[:, 0]
        warped_previous_features = sample_at_grid(
            previous_features,
            reprojection.previous_normalized_xy,
            mode="bilinear",
        )

        sampled_depth_valid = torch.isfinite(warped_previous_depth) & (
            warped_previous_depth > self.min_depth
        )
        correspondence_valid = reprojection.valid & sampled_depth_valid
        depth_residual = torch.abs(
            warped_previous_depth - reprojection.previous_camera_depth
        )
        depth_residual = torch.where(
            correspondence_valid, depth_residual, torch.zeros_like(depth_residual)
        )

        previous_norm = torch.linalg.vector_norm(warped_previous_features, dim=1)
        current_norm = torch.linalg.vector_norm(current_features, dim=1)
        dino_valid = (previous_norm > 1e-8) & (current_norm > 1e-8)
        cosine = (warped_previous_features * current_features).sum(dim=1) / (
            previous_norm * current_norm
        ).clamp_min(1e-8)
        dino_distance = (1.0 - cosine).clamp(0.0, 2.0)
        dino_distance = torch.where(
            correspondence_valid & dino_valid,
            dino_distance,
            torch.zeros_like(dino_distance),
        )

        depth_component = self.depth_weight * (depth_residual / self.depth_threshold)
        dino_component = self.dino_weight * (dino_distance / self.dino_threshold)
        motion_score = torch.maximum(depth_component, dino_component)
        motion_mask = correspondence_valid & (
            ((depth_residual > self.depth_threshold) & (self.depth_weight > 0))
            | (
                (dino_distance > self.dino_threshold)
                & dino_valid
                & (self.dino_weight > 0)
            )
        )

        if self.dilation_radius:
            kernel = 2 * self.dilation_radius + 1
            motion_mask = (
                F.max_pool2d(
                    motion_mask[:, None].to(motion_score.dtype),
                    kernel_size=kernel,
                    stride=1,
                    padding=self.dilation_radius,
                )[:, 0]
                > 0.5
            )
            motion_score = F.max_pool2d(
                motion_score[:, None],
                kernel_size=kernel,
                stride=1,
                padding=self.dilation_radius,
            )[:, 0]

        if wan_grid_size is None:
            wan_grid = current_grid
        else:
            wan_grid = _positive_grid_size(wan_grid_size, "wan_grid_size")
            if len(wan_grid) != 2:
                raise ValueError(
                    f"wan_grid_size must have two dimensions, got {wan_grid}."
                )
        wan_score = _resize_max(motion_score[:, None], wan_grid)[:, 0]
        wan_mask_before_limit = _resize_any_mask(motion_mask[:, None], wan_grid)[:, 0]
        moving_indices, indices_valid = padded_indices_from_mask(
            wan_mask_before_limit,
            scores=wan_score,
            max_tokens=self.max_tokens,
        )
        wan_mask = _mask_from_padded_indices(moving_indices, indices_valid, wan_grid)
        wan_dino = pool_dino_to_grid(
            dino_current,
            wan_grid,
            source_grid_size=dino_grid_size,
        )
        return MotionDetectionResult(
            dino_motion_score=motion_score,
            dino_motion_mask=motion_mask,
            dino_correspondence_valid=correspondence_valid,
            depth_residual=depth_residual,
            dino_distance=dino_distance,
            wan_motion_score=wan_score,
            wan_motion_mask=wan_mask,
            wan_moving_indices=moving_indices,
            wan_indices_valid=indices_valid,
            wan_dino_features=wan_dino,
        )


def patchify_latents(
    latents: torch.Tensor, patch_size: Sequence[int] = (1, 2, 2)
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Patchify ``(B,C,F,H,W)`` exactly like N0-TWAM's latent input path."""

    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        shape = tuple(latents.shape) if isinstance(latents, torch.Tensor) else None
        raise ValueError(f"latents must have shape (B,C,F,H,W), got {shape}.")
    patch = _positive_grid_size(patch_size, "patch_size")
    if len(patch) != 3:
        raise ValueError(f"patch_size must have three dimensions, got {patch}.")
    batch, channels, frames, height, width = latents.shape
    if batch < 1 or channels < 1:
        raise ValueError("latents batch and channel dimensions must be non-empty.")
    pf, ph, pw = patch
    if frames % pf or height % ph or width % pw:
        raise ValueError(
            f"latent shape {(frames, height, width)} must be divisible by "
            f"patch_size {patch}."
        )
    grid = (frames // pf, height // ph, width // pw)
    values = latents.reshape(batch, channels, grid[0], pf, grid[1], ph, grid[2], pw)
    values = values.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return values.reshape(batch, prod(grid), channels * prod(patch)), grid


def unpatchify_latents(
    patches: torch.Tensor,
    grid_shape: Sequence[int],
    patch_size: Sequence[int],
    channels: int,
) -> torch.Tensor:
    """Inverse of :func:`patchify_latents`."""

    if not isinstance(patches, torch.Tensor) or patches.ndim != 3:
        shape = tuple(patches.shape) if isinstance(patches, torch.Tensor) else None
        raise ValueError(f"patches must have shape (B,L,P), got {shape}.")
    grid = _positive_grid_size(grid_shape, "grid_shape")
    patch = _positive_grid_size(patch_size, "patch_size")
    if len(grid) != 3 or len(patch) != 3:
        raise ValueError("grid_shape and patch_size must each have three dimensions.")
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}.")
    expected_tokens = prod(grid)
    expected_width = channels * prod(patch)
    if patches.shape[1:] != (expected_tokens, expected_width):
        raise ValueError(
            f"patches trailing shape must be {(expected_tokens, expected_width)}, "
            f"got {tuple(patches.shape[1:])}."
        )
    batch = patches.shape[0]
    gf, gh, gw = grid
    pf, ph, pw = patch
    values = patches.reshape(batch, gf, gh, gw, channels, pf, ph, pw)
    values = values.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return values.reshape(batch, channels, gf * pf, gh * ph, gw * pw)


@dataclass(frozen=True)
class SparsePatchBatch:
    """Padded, batch-safe raw WAN patches selected from a dense latent."""

    values: torch.Tensor
    indices: torch.Tensor
    valid_mask: torch.Tensor
    grid_shape: tuple[int, int, int]
    latent_shape: tuple[int, int, int, int, int]
    patch_size: tuple[int, int, int] = (1, 2, 2)

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(
                f"values must have shape (B,K,P), got {tuple(self.values.shape)}."
            )
        if self.indices.ndim != 2 or self.valid_mask.ndim != 2:
            raise ValueError("indices and valid_mask must have shape (B,K).")
        if (
            self.indices.shape != self.valid_mask.shape
            or self.values.shape[:2] != self.indices.shape
        ):
            raise ValueError(
                "values, indices, and valid_mask disagree on (B,K): "
                f"{tuple(self.values.shape)}, {tuple(self.indices.shape)}, "
                f"{tuple(self.valid_mask.shape)}."
            )
        if self.indices.dtype != torch.long:
            raise TypeError(f"indices must have dtype long, got {self.indices.dtype}.")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError(
                f"valid_mask must have dtype bool, got {self.valid_mask.dtype}."
            )
        if not (self.values.device == self.indices.device == self.valid_mask.device):
            raise ValueError("values, indices, and valid_mask must share a device.")
        grid = _positive_grid_size(self.grid_shape, "grid_shape")
        patch = _positive_grid_size(self.patch_size, "patch_size")
        if len(grid) != 3 or len(patch) != 3 or len(self.latent_shape) != 5:
            raise ValueError(
                "grid_shape/patch_size must have 3 dimensions and latent_shape 5."
            )
        if self.latent_shape[0] != self.values.shape[0]:
            raise ValueError("latent_shape batch size does not match values.")
        expected_grid = tuple(
            self.latent_shape[index + 2] // patch[index] for index in range(3)
        )
        if (
            any(self.latent_shape[index + 2] % patch[index] for index in range(3))
            or expected_grid != grid
        ):
            raise ValueError(
                f"grid_shape {grid} is inconsistent with latent_shape "
                f"{self.latent_shape} and patch_size {patch}."
            )
        expected_width = self.latent_shape[1] * prod(patch)
        if self.values.shape[-1] != expected_width:
            raise ValueError(
                f"values width must be {expected_width}, got {self.values.shape[-1]}."
            )
        if torch.any(self.indices[self.valid_mask] < 0) or torch.any(
            self.indices[self.valid_mask] >= prod(grid)
        ):
            raise ValueError("A valid sparse index is outside the patch grid.")
        if torch.any(self.indices[~self.valid_mask] != -1):
            raise ValueError("Invalid/padded sparse indices must use the sentinel -1.")

    def with_values(self, values: torch.Tensor) -> "SparsePatchBatch":
        """Return the same addressing metadata carrying new predicted patch values."""

        return replace(self, values=values)


class SparsePatchGather(nn.Module):
    """Gather selected raw patches before N0-TWAM's existing input projection."""

    def __init__(self, patch_size: Sequence[int] = (1, 2, 2)) -> None:
        super().__init__()
        patch = _positive_grid_size(patch_size, "patch_size")
        if len(patch) != 3:
            raise ValueError(f"patch_size must have three dimensions, got {patch}.")
        self.patch_size = patch

    def forward(
        self,
        latents: torch.Tensor,
        *,
        motion_mask: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
        max_tokens: int | None = None,
    ) -> SparsePatchBatch:
        patches, grid = patchify_latents(latents, self.patch_size)
        batch_size = patches.shape[0]
        if motion_mask is not None:
            if indices is not None or valid_mask is not None:
                raise ValueError(
                    "Provide motion_mask or explicit indices/valid_mask, not both."
                )
            expected_shape = (batch_size, *grid)
            if motion_mask.ndim == 3 and grid[0] == 1:
                motion_mask = motion_mask[:, None]
                if scores is not None and scores.ndim == 3:
                    scores = scores[:, None]
            if tuple(motion_mask.shape) != expected_shape:
                raise ValueError(
                    f"motion_mask must have shape {expected_shape}, got "
                    f"{tuple(motion_mask.shape)}."
                )
            indices, valid_mask = padded_indices_from_mask(
                motion_mask, scores=scores, max_tokens=max_tokens
            )
        elif indices is None:
            raise ValueError("Either motion_mask or indices must be provided.")
        else:
            if max_tokens is not None or scores is not None:
                raise ValueError(
                    "scores/max_tokens are only valid when deriving indices "
                    "from a mask."
                )
            if indices.ndim != 2 or indices.shape[0] != batch_size:
                raise ValueError(
                    f"indices must have shape ({batch_size}, K), got "
                    f"{tuple(indices.shape)}."
                )
            if indices.dtype not in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError(f"indices must be integer, got {indices.dtype}.")
            indices = indices.to(device=latents.device, dtype=torch.long)
            if valid_mask is None:
                valid_mask = torch.ones_like(indices, dtype=torch.bool)
            elif valid_mask.shape != indices.shape or valid_mask.dtype != torch.bool:
                raise ValueError(
                    "valid_mask must be bool and have the same shape as indices."
                )
            else:
                valid_mask = valid_mask.to(device=latents.device)
            indices = torch.where(valid_mask, indices, torch.full_like(indices, -1))

        assert indices is not None and valid_mask is not None
        if indices.device != latents.device or valid_mask.device != latents.device:
            indices = indices.to(latents.device)
            valid_mask = valid_mask.to(latents.device)
        if torch.any(indices[valid_mask] < 0) or torch.any(
            indices[valid_mask] >= patches.shape[1]
        ):
            raise ValueError("A valid gather index is outside the latent patch grid.")
        safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
        gathered = torch.gather(
            patches,
            1,
            safe_indices[..., None].expand(-1, -1, patches.shape[-1]),
        )
        gathered = torch.where(
            valid_mask[..., None], gathered, torch.zeros_like(gathered)
        )
        return SparsePatchBatch(
            values=gathered,
            indices=torch.where(valid_mask, indices, torch.full_like(indices, -1)),
            valid_mask=valid_mask,
            grid_shape=grid,
            latent_shape=tuple(latents.shape),
            patch_size=self.patch_size,
        )


class SparsePatchScatter(nn.Module):
    """Scatter predicted sparse patches into a dense latent canvas."""

    def forward(
        self,
        sparse: SparsePatchBatch,
        *,
        base_latents: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patch_values = sparse.values if values is None else values
        if not isinstance(patch_values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor.")
        if patch_values.shape != sparse.values.shape:
            raise ValueError(
                f"values must have shape {tuple(sparse.values.shape)}, got "
                f"{tuple(patch_values.shape)}."
            )
        if patch_values.device != sparse.values.device:
            raise ValueError(
                "Replacement values and sparse metadata must share a device."
            )
        if base_latents is None:
            base_latents = torch.zeros(
                sparse.latent_shape,
                device=patch_values.device,
                dtype=patch_values.dtype,
            )
        elif tuple(base_latents.shape) != sparse.latent_shape:
            raise ValueError(
                f"base_latents must have shape {sparse.latent_shape}, got "
                f"{tuple(base_latents.shape)}."
            )
        if (
            base_latents.device != patch_values.device
            or base_latents.dtype != patch_values.dtype
        ):
            raise ValueError(
                "base_latents and sparse values must have the same device and dtype."
            )
        dense_patches, grid = patchify_latents(base_latents, sparse.patch_size)
        if grid != sparse.grid_shape:
            raise ValueError(
                "base_latents produced a grid different from sparse metadata."
            )

        batches: list[torch.Tensor] = []
        for batch_index in range(dense_patches.shape[0]):
            selected = sparse.indices[batch_index, sparse.valid_mask[batch_index]]
            if selected.numel() != torch.unique(selected).numel():
                raise ValueError(
                    "Duplicate valid indices are ambiguous during sparse scatter."
                )
            selected_values = patch_values[batch_index, sparse.valid_mask[batch_index]]
            batches.append(
                dense_patches[batch_index].index_copy(0, selected, selected_values)
            )
        output_patches = torch.stack(batches, dim=0)
        return unpatchify_latents(
            output_patches,
            sparse.grid_shape,
            sparse.patch_size,
            sparse.latent_shape[1],
        )


def gather_grid_features(
    features: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather channels-last grid metadata with padded sparse indices."""

    if features.ndim < 3:
        raise ValueError(
            f"features must have shape (B, ..., D), got {tuple(features.shape)}."
        )
    if indices.ndim != 2 or valid_mask.shape != indices.shape:
        raise ValueError("indices and valid_mask must have matching shape (B,K).")
    if indices.dtype != torch.long or valid_mask.dtype != torch.bool:
        raise TypeError("indices must be long and valid_mask must be bool.")
    if features.shape[0] != indices.shape[0]:
        raise ValueError("features and indices batch sizes differ.")
    if features.device != indices.device or indices.device != valid_mask.device:
        raise ValueError("features, indices, and valid_mask must share a device.")
    flattened = features.reshape(features.shape[0], -1, features.shape[-1])
    if torch.any(indices[valid_mask] < 0) or torch.any(
        indices[valid_mask] >= flattened.shape[1]
    ):
        raise ValueError("A valid feature index is outside the source grid.")
    safe = torch.where(valid_mask, indices, torch.zeros_like(indices))
    gathered = torch.gather(
        flattened, 1, safe[..., None].expand(-1, -1, flattened.shape[-1])
    )
    return torch.where(valid_mask[..., None], gathered, torch.zeros_like(gathered))


def _expand_sidecar_scalar(
    value: Any,
    *,
    name: str,
    batch_size: int,
    token_count: int,
    device: torch.device,
    binary: bool = False,
) -> torch.Tensor:
    """Normalize scalar/per-frame/per-token metadata to ``(F,K)``."""

    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be an integer scalar or tensor-like value."
        ) from error
    if tensor.dtype == torch.bool:
        if not binary:
            raise TypeError(f"{name} must contain integer values, not bool.")
    elif tensor.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError(f"{name} must contain integer values, got {tensor.dtype}.")

    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size, token_count)
    elif tuple(tensor.shape) == (batch_size,):
        tensor = tensor[:, None].expand(batch_size, token_count)
    elif tuple(tensor.shape) == (batch_size, 1):
        tensor = tensor.expand(batch_size, token_count)
    elif tuple(tensor.shape) != (batch_size, token_count):
        raise ValueError(
            f"{name} must be scalar, ({batch_size},), ({batch_size}, 1), or "
            f"({batch_size}, {token_count}); got {tuple(tensor.shape)}."
        )
    tensor = tensor.to(device=device, dtype=torch.long)
    if binary and torch.any((tensor != 0) & (tensor != 1)):
        raise ValueError(f"{name} must contain only 0 or 1.")
    return tensor.clone()


def _selected_presence_mask(
    value: Any,
    *,
    name: str,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Normalize a presence mask on either the WAN grid or selected tokens."""

    batch_size, token_count = indices.shape
    tensor = torch.as_tensor(value)
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype bool, got {tensor.dtype}.")
    if tensor.device != indices.device:
        raise ValueError(f"{name} and MotionDetectionResult must share a device.")
    if tensor.ndim == 0 or tuple(tensor.shape) in (
        (batch_size,),
        (batch_size, 1),
        (batch_size, token_count),
    ):
        selected = _expand_sidecar_scalar(
            tensor,
            name=name,
            batch_size=batch_size,
            token_count=token_count,
            device=indices.device,
            binary=True,
        ).bool()
    elif tuple(tensor.shape) == (batch_size, *grid_size):
        selected = gather_grid_features(tensor[..., None], indices, valid_mask)[..., 0]
    else:
        raise ValueError(
            f"{name} must be scalar, per-frame/per-token, or have WAN-grid "
            f"shape {(batch_size, *grid_size)}; got {tuple(tensor.shape)}."
        )
    return selected & valid_mask


def motion_result_to_sidecar(
    result: MotionDetectionResult,
    *,
    world_time_id: Any,
    observation_flag: Any,
    neoforce_features: torch.Tensor | None = None,
    tactile_valid: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Convert motion detection output into the canonical RGB-motion sidecar.

    The detector's batch dimension is interpreted as the sidecar frame
    dimension: ``B == F``.  Returned indices are frame-local WAN spatial
    indices with shape ``(F,K)``.  ``observation_flag`` uses ``1`` for an
    actually observed frame and ``0`` for a world-model prediction.

    NeoForce features may be supplied either on the full WAN grid
    ``(F,Hw,Ww,Dn)`` or already selected as ``(F,K,Dn)``.  Their presence
    cannot be inferred from numeric values, so a non-empty NeoForce input must
    be accompanied by ``tactile_valid``.  Omitting NeoForce yields the RGB-only
    canonical representation ``(F,K,0)`` with an all-false tactile mask.

    This function only constructs metadata.  It never adds or concatenates any
    index field to WAN content patches or their embeddings.
    """

    if not isinstance(result, MotionDetectionResult):
        raise TypeError("result must be a MotionDetectionResult.")
    indices = result.wan_moving_indices
    valid = result.wan_indices_valid
    scores = result.wan_motion_score
    motion_mask = result.wan_motion_mask
    dino_grid = result.wan_dino_features

    if indices.ndim != 2 or indices.dtype != torch.long:
        raise TypeError("wan_moving_indices must be a long tensor with shape (F,K).")
    if valid.shape != indices.shape or valid.dtype != torch.bool:
        raise TypeError("wan_indices_valid must be bool and match (F,K) indices.")
    if scores.ndim != 3 or not torch.is_floating_point(scores):
        raise TypeError("wan_motion_score must be floating point with shape (F,H,W).")
    batch_size, height, width = scores.shape
    token_count = indices.shape[1]
    if indices.shape[0] != batch_size:
        raise ValueError("WAN scores and selected indices have different frame counts.")
    if motion_mask.shape != scores.shape or motion_mask.dtype != torch.bool:
        raise TypeError("wan_motion_mask must be bool and match wan_motion_score.")
    if (
        dino_grid.ndim != 4
        or dino_grid.shape[:3] != scores.shape
        or dino_grid.shape[-1] < 1
        or not torch.is_floating_point(dino_grid)
    ):
        raise ValueError(
            "wan_dino_features must be floating point with shape (F,H,W,D), D > 0."
        )
    tensors = (indices, valid, scores, motion_mask, dino_grid)
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("All MotionDetectionResult WAN fields must share a device.")
    if torch.any(indices[~valid] != -1):
        raise ValueError("Invalid/padded motion indices must use the sentinel -1.")
    if torch.any(indices[valid] < 0) or torch.any(indices[valid] >= height * width):
        raise ValueError("A valid motion index is outside the WAN spatial grid.")
    expected_mask = _mask_from_padded_indices(indices, valid, (height, width))
    if not torch.equal(expected_mask, motion_mask):
        raise ValueError("wan_motion_mask disagrees with selected indices/validity.")

    dino = gather_grid_features(dino_grid, indices, valid)
    selected_scores = gather_grid_features(scores[..., None], indices, valid)[..., 0]
    if valid.any() and (
        not torch.isfinite(dino[valid]).all()
        or not torch.isfinite(selected_scores[valid]).all()
    ):
        raise ValueError("Valid DINO features and motion scores must be finite.")

    world_time = _expand_sidecar_scalar(
        world_time_id,
        name="world_time_id",
        batch_size=batch_size,
        token_count=token_count,
        device=indices.device,
    )
    if valid.any() and torch.any(world_time[valid] < 0):
        raise ValueError("world_time_id must be non-negative for valid tokens.")
    source = _expand_sidecar_scalar(
        observation_flag,
        name="observation_flag",
        batch_size=batch_size,
        token_count=token_count,
        device=indices.device,
        binary=True,
    )

    if neoforce_features is None:
        neoforce = torch.empty(
            (batch_size, token_count, 0),
            device=dino.device,
            dtype=dino.dtype,
        )
        if tactile_valid is not None:
            tactile = _selected_presence_mask(
                tactile_valid,
                name="tactile_valid",
                indices=indices,
                valid_mask=valid,
                grid_size=(height, width),
            )
            if tactile.any():
                raise ValueError(
                    "tactile_valid cannot be true when NeoForce is omitted."
                )
        tactile = torch.zeros_like(valid)
    else:
        if not isinstance(neoforce_features, torch.Tensor):
            raise TypeError("neoforce_features must be a torch.Tensor.")
        if not torch.is_floating_point(neoforce_features):
            raise TypeError("neoforce_features must be floating point.")
        if neoforce_features.device != indices.device:
            raise ValueError(
                "neoforce_features and MotionDetectionResult must share a device."
            )
        if neoforce_features.ndim == 4 and tuple(neoforce_features.shape[:3]) == (
            batch_size,
            height,
            width,
        ):
            neoforce = gather_grid_features(neoforce_features, indices, valid)
        elif neoforce_features.ndim == 3 and tuple(neoforce_features.shape[:2]) == (
            batch_size,
            token_count,
        ):
            neoforce = torch.where(
                valid[..., None], neoforce_features, torch.zeros_like(neoforce_features)
            )
        else:
            raise ValueError(
                "neoforce_features must have shape "
                f"{(batch_size, height, width, 'D')} or "
                f"{(batch_size, token_count, 'D')}; got "
                f"{tuple(neoforce_features.shape)}."
            )
        if neoforce.shape[-1] == 0:
            if tactile_valid is not None:
                tactile = _selected_presence_mask(
                    tactile_valid,
                    name="tactile_valid",
                    indices=indices,
                    valid_mask=valid,
                    grid_size=(height, width),
                )
                if tactile.any():
                    raise ValueError(
                        "tactile_valid cannot be true when NeoForce width is zero."
                    )
            tactile = torch.zeros_like(valid)
        else:
            if tactile_valid is None:
                raise ValueError(
                    "tactile_valid is required when NeoForce features are present."
                )
            tactile = _selected_presence_mask(
                tactile_valid,
                name="tactile_valid",
                indices=indices,
                valid_mask=valid,
                grid_size=(height, width),
            )
            if tactile.any() and not torch.isfinite(neoforce[tactile]).all():
                raise ValueError("Tactile-valid NeoForce features must be finite.")

    sidecar = {
        "motion_indices": torch.where(valid, indices, torch.full_like(indices, -1)),
        "motion_valid_mask": valid.clone(),
        "motion_scores": selected_scores.masked_fill(~valid, 0).contiguous(),
        "world_time_id": world_time.masked_fill(~valid, -1).contiguous(),
        "dino_features": dino.contiguous(),
        "neoforce_features": neoforce.contiguous(),
        "observation_flag": source.masked_fill(~valid, 0).contiguous(),
        "visual_valid": valid.clone(),
        "tactile_valid": tactile.contiguous(),
    }
    # Reuse the semantic-index invariants without coupling the index to content.
    TokenIndex(
        world_time_id=sidecar["world_time_id"],
        dino=sidecar["dino_features"],
        neoforce=sidecar["neoforce_features"],
        observation_flag=sidecar["observation_flag"],
        visual_valid=sidecar["visual_valid"],
        tactile_valid=sidecar["tactile_valid"],
    ).validate_presence(sidecar["motion_valid_mask"])
    return sidecar


@dataclass(frozen=True)
class TokenIndex:
    """Independent semantic index carried beside, never inside, content values.

    Formal index fields are ``{world_time_id, dino, neoforce,
    observation_flag}``, where ``observation_flag=1`` means observed and ``0``
    means predicted. ``visual_valid`` and ``tactile_valid`` are presence masks so
    a padded zero feature cannot be confused with a real feature. They are support
    metadata, not extra index dimensions.
    """

    world_time_id: torch.Tensor
    dino: torch.Tensor
    neoforce: torch.Tensor
    observation_flag: torch.Tensor
    visual_valid: torch.Tensor
    tactile_valid: torch.Tensor

    def __post_init__(self) -> None:
        tensors = {
            "world_time_id": self.world_time_id,
            "dino": self.dino,
            "neoforce": self.neoforce,
            "observation_flag": self.observation_flag,
            "visual_valid": self.visual_valid,
            "tactile_valid": self.tactile_valid,
        }
        if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("Every TokenIndex field must be a torch.Tensor.")
        if self.world_time_id.ndim != 2:
            raise ValueError("world_time_id must have shape (B,K).")
        if self.world_time_id.dtype == torch.bool or self.world_time_id.is_complex():
            raise TypeError("world_time_id must contain real numeric time values.")
        if (
            self.world_time_id.dtype.is_floating_point
            and not torch.isfinite(self.world_time_id).all()
        ):
            raise ValueError("world_time_id must contain only finite values.")
        shape = self.world_time_id.shape
        if self.dino.ndim != 3 or self.dino.shape[:2] != shape:
            raise ValueError("dino must have shape (B,K,D_dino).")
        if self.neoforce.ndim != 3 or self.neoforce.shape[:2] != shape:
            raise ValueError("neoforce must have shape (B,K,D_neoforce).")
        dino_width = self.dino.shape[-1]
        neoforce_width = self.neoforce.shape[-1]
        if dino_width == 0 and neoforce_width == 0:
            raise ValueError(
                "DINO and NeoForce feature dimensions cannot both be zero."
            )
        if not torch.is_floating_point(self.dino) or not torch.is_floating_point(
            self.neoforce
        ):
            raise TypeError("dino and neoforce must be floating point tensors.")
        for name, value in (
            ("observation_flag", self.observation_flag),
            ("visual_valid", self.visual_valid),
            ("tactile_valid", self.tactile_valid),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {tuple(shape)}.")
        if (
            self.visual_valid.dtype != torch.bool
            or self.tactile_valid.dtype != torch.bool
        ):
            raise TypeError("visual_valid and tactile_valid must have dtype bool.")
        if dino_width == 0 and torch.any(self.visual_valid):
            raise ValueError(
                "visual_valid must be all False when the DINO feature width is zero."
            )
        if neoforce_width == 0 and torch.any(self.tactile_valid):
            raise ValueError(
                "tactile_valid must be all False when the NeoForce feature width "
                "is zero."
            )
        if self.observation_flag.dtype == torch.bool:
            pass
        elif (
            self.observation_flag.dtype.is_floating_point
            or self.observation_flag.dtype
            in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        ):
            if torch.any((self.observation_flag != 0) & (self.observation_flag != 1)):
                raise ValueError("observation_flag values must be exactly 0 or 1.")
        else:
            raise TypeError(
                "observation_flag must be bool, integer, or floating point."
            )
        devices = {value.device for value in tensors.values()}
        if len(devices) != 1:
            raise ValueError("All TokenIndex fields must be on the same device.")

    @property
    def observed(self) -> torch.Tensor:
        return self.observation_flag.to(torch.bool)

    @property
    def predicted(self) -> torch.Tensor:
        return ~self.observed

    def validate_presence(self, token_valid_mask: torch.Tensor | None = None) -> None:
        """Require DINO or NeoForce for every real (non-padding) token."""

        present = self.visual_valid | self.tactile_valid
        if token_valid_mask is None:
            token_valid_mask = torch.ones_like(present)
        elif (
            token_valid_mask.shape != present.shape
            or token_valid_mask.dtype != torch.bool
        ):
            raise ValueError(
                "token_valid_mask must be bool and match the TokenIndex (B,K) shape."
            )
        if token_valid_mask.device != present.device:
            raise ValueError("token_valid_mask and TokenIndex must share a device.")
        if torch.any(token_valid_mask & ~present):
            raise ValueError(
                "Every valid token must contain at least one of DINO or NeoForce."
            )


__all__ = [
    "EgoMotionCompensatedMotionDetector",
    "MotionDetectionResult",
    "SparsePatchBatch",
    "SparsePatchGather",
    "SparsePatchScatter",
    "TokenIndex",
    "gather_grid_features",
    "motion_result_to_sidecar",
    "padded_indices_from_mask",
    "patchify_latents",
    "pool_dino_to_grid",
    "unpatchify_latents",
]
