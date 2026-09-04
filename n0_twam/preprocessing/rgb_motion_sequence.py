"""Sequence-level RGB-D preprocessing for sparse N0-TWAM video tokens.

This module deliberately owns no DINOv2 weights and no dataset I/O.  Callers
inject a patch encoder and the motion detector, then provide already aligned
RGB/depth/calibration sequences.  Keeping this layer independent makes the
same temporal and multi-camera logic usable by an offline sidecar builder and
an online observation worker.

The returned semantic fields remain metadata.  They are never mixed into the
WAN content patches or their embeddings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from n0_twam.models.rgb_motion import gather_grid_features, pool_dino_to_grid


FirstFramePolicy = Literal["require_previous", "empty", "all"]

_CANONICAL_FIELDS = (
    "motion_indices",
    "motion_valid_mask",
    "motion_scores",
    "world_time_id",
    "dino_features",
    "neoforce_features",
    "observation_flag",
    "visual_valid",
    "tactile_valid",
)


@dataclass(frozen=True)
class RGBDCameraSequence:
    """Raw frames and camera geometry for one camera.

    ``rgb`` is ``[T,3,H,W]`` or ``[T,H,W,3]``.  ``depth`` is z-depth with
    shape ``[T,H,W]`` or ``[T,1,H,W]``.  Poses and intrinsics may either have
    one entry per raw frame or be a single constant matrix.  ``dino_grid_size``
    is needed only when an injected encoder returns flattened ``[B,N,D]``
    tokens instead of a channels-last patch map.
    """

    rgb: torch.Tensor
    depth: torch.Tensor
    camera_pose: torch.Tensor
    camera_intrinsics: torch.Tensor
    dino_grid_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for name in ("rgb", "depth", "camera_pose", "camera_intrinsics"):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        channels_first = self.rgb.ndim == 4 and self.rgb.shape[1] == 3
        channels_last = self.rgb.ndim == 4 and self.rgb.shape[-1] == 3
        if channels_first == channels_last:
            raise ValueError(
                "rgb must unambiguously have exactly 3 channels and shape "
                f"[T,3,H,W] or [T,H,W,3], got {tuple(self.rgb.shape)}"
            )
        if self.rgb.is_complex() or (
            torch.is_floating_point(self.rgb)
            and not torch.isfinite(self.rgb).all()
        ):
            raise ValueError("rgb must be real-valued and finite")
        if not torch.is_floating_point(self.depth):
            raise TypeError(
                "depth must be calibrated floating-point z-depth; integer "
                "depth units are ambiguous"
            )
        if self.depth.ndim == 4:
            if self.depth.shape[1] != 1:
                raise ValueError(
                    "4-D depth must have shape [T,1,H,W], got "
                    f"{tuple(self.depth.shape)}"
                )
            depth_frames = self.depth.shape[0]
        elif self.depth.ndim == 3:
            depth_frames = self.depth.shape[0]
        else:
            raise ValueError(
                "depth must have shape [T,H,W] or [T,1,H,W], got "
                f"{tuple(self.depth.shape)}"
            )
        if self.rgb.shape[0] < 1 or depth_frames != self.rgb.shape[0]:
            raise ValueError(
                "rgb and depth must have the same non-empty T dimension, got "
                f"{self.rgb.shape[0]} and {depth_frames}"
            )
        rgb_spatial = (
            tuple(self.rgb.shape[-2:])
            if channels_first
            else tuple(self.rgb.shape[1:3])
        )
        if rgb_spatial != tuple(self.depth.shape[-2:]):
            raise ValueError(
                "RGB/depth spatial shapes must match for pixel alignment, got "
                f"{rgb_spatial} and {tuple(self.depth.shape[-2:])}"
            )
        _validate_matrix_series(
            self.camera_pose, self.rgb.shape[0], (4, 4), "camera_pose"
        )
        _validate_matrix_series(
            self.camera_intrinsics,
            self.rgb.shape[0],
            (3, 3),
            "camera_intrinsics",
        )
        for name, matrix in (
            ("camera_pose", self.camera_pose),
            ("camera_intrinsics", self.camera_intrinsics),
        ):
            if not torch.is_floating_point(matrix) or matrix.is_complex():
                raise TypeError(f"{name} must be a real floating-point tensor")
            if not torch.isfinite(matrix).all():
                raise ValueError(f"{name} must contain only finite values")
        intrinsics = (
            self.camera_intrinsics.unsqueeze(0)
            if self.camera_intrinsics.ndim == 2
            else self.camera_intrinsics
        )
        if torch.any(intrinsics[:, 0, 0] <= 0) or torch.any(
            intrinsics[:, 1, 1] <= 0
        ):
            raise ValueError("camera_intrinsics focal lengths fx/fy must be positive")
        if self.dino_grid_size is not None:
            grid = tuple(int(v) for v in self.dino_grid_size)
            if len(grid) != 2 or any(v <= 0 for v in grid):
                raise ValueError(
                    f"dino_grid_size must be two positive integers, got {grid}"
                )

    @property
    def num_frames(self) -> int:
        return int(self.rgb.shape[0])


def _validate_matrix_series(
    value: torch.Tensor,
    num_frames: int,
    matrix_shape: tuple[int, int],
    name: str,
) -> None:
    if tuple(value.shape) == matrix_shape:
        return
    if (
        value.ndim == 3
        and tuple(value.shape[1:]) == matrix_shape
        and value.shape[0]
        in (
            1,
            num_frames,
        )
    ):
        return
    raise ValueError(
        f"{name} must have shape {matrix_shape}, [1,{matrix_shape[0]},"
        f"{matrix_shape[1]}], or [T,{matrix_shape[0]},{matrix_shape[1]}] "
        f"with T={num_frames}; got {tuple(value.shape)}"
    )


def _integer_vector(
    value: Any, *, name: str, expected: int | None = None
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tuple(tensor.shape)}")
    if expected is not None and tensor.numel() != expected:
        raise ValueError(f"{name} must have {expected} entries, got {tensor.numel()}")
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{name} must contain integer values")
    if torch.is_floating_point(tensor):
        if tensor.numel() and (
            not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round())
        ):
            raise ValueError(f"{name} must contain finite integer values")
    return tensor.to(dtype=torch.long, device="cpu")


def _select_matrix_series(
    value: torch.Tensor,
    anchors: torch.Tensor,
    num_raw_frames: int,
    matrix_shape: tuple[int, int],
    name: str,
) -> torch.Tensor:
    _validate_matrix_series(value, num_raw_frames, matrix_shape, name)
    if tuple(value.shape) == matrix_shape:
        return value.unsqueeze(0).expand(anchors.numel(), -1, -1)
    if value.shape[0] == 1:
        return value.expand(anchors.numel(), -1, -1)
    return value.index_select(0, anchors.to(value.device))


def _one_matrix(value: torch.Tensor, shape: tuple[int, int], name: str) -> torch.Tensor:
    if tuple(value.shape) == shape:
        return value.unsqueeze(0)
    if value.ndim == 3 and value.shape[0] == 1 and tuple(value.shape[1:]) == shape:
        return value
    raise ValueError(
        f"previous {name} must have shape {shape} or [1,{shape[0]},{shape[1]}], "
        f"got {tuple(value.shape)}"
    )


def _one_rgb(value: torch.Tensor, example: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) == tuple(example.shape):
        return value.unsqueeze(0)
    if (
        value.ndim == example.ndim + 1
        and value.shape[0] == 1
        and tuple(value.shape[1:]) == tuple(example.shape)
    ):
        return value
    raise ValueError(
        "previous rgb must be one frame with shape matching the current camera; "
        f"expected {tuple(example.shape)} or {(1, *example.shape)}, got "
        f"{tuple(value.shape)}"
    )


def _one_depth(value: torch.Tensor, example: torch.Tensor) -> torch.Tensor:
    # The detector accepts [B,H,W] or [B,1,H,W].  Preserve the sequence's form.
    if (
        example.ndim == 3
        and example.shape[0] == 1
        and tuple(value.shape) == tuple(example.shape[1:])
    ):
        return value.unsqueeze(0).unsqueeze(0)
    if tuple(value.shape) == tuple(example.shape):
        return value.unsqueeze(0)
    if (
        value.ndim == example.ndim + 1
        and value.shape[0] == 1
        and tuple(value.shape[1:]) == tuple(example.shape)
    ):
        return value
    raise ValueError(
        "previous depth must be one frame with shape matching the current camera; "
        f"expected {tuple(example.shape)} or {(1, *example.shape)}, got "
        f"{tuple(value.shape)}"
    )


def _previous_values(
    previous: RGBDCameraSequence | Mapping[str, Any],
    current: RGBDCameraSequence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(previous, RGBDCameraSequence):
        rgb = previous.rgb[-1:]
        depth = previous.depth[-1:]
        pose = _select_matrix_series(
            previous.camera_pose,
            torch.tensor([previous.num_frames - 1]),
            previous.num_frames,
            (4, 4),
            "camera_pose",
        )
        intrinsics = _select_matrix_series(
            previous.camera_intrinsics,
            torch.tensor([previous.num_frames - 1]),
            previous.num_frames,
            (3, 3),
            "camera_intrinsics",
        )
        return rgb, depth, pose, intrinsics
    if not isinstance(previous, Mapping):
        raise TypeError(
            "previous_frames values must be RGBDCameraSequence or a field mapping"
        )
    required = ("rgb", "depth", "camera_pose", "camera_intrinsics")
    missing = [name for name in required if name not in previous]
    if missing:
        raise KeyError(f"previous camera frame is missing fields {missing}")
    rgb = _one_rgb(torch.as_tensor(previous["rgb"]), current.rgb[0])
    depth = _one_depth(torch.as_tensor(previous["depth"]), current.depth[0])
    pose = _one_matrix(torch.as_tensor(previous["camera_pose"]), (4, 4), "camera_pose")
    intrinsics = _one_matrix(
        torch.as_tensor(previous["camera_intrinsics"]),
        (3, 3),
        "camera_intrinsics",
    )
    return rgb, depth, pose, intrinsics


def _encoder_tokens(
    encoder: Any,
    rgb: torch.Tensor,
    *,
    declared_grid: tuple[int, int] | None,
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    output = encoder(rgb)
    output_grid = getattr(output, "grid_size", None)
    if isinstance(output, torch.Tensor):
        tokens = output
    elif hasattr(output, "tokens"):
        tokens = output.tokens
    elif isinstance(output, Mapping) and "tokens" in output:
        tokens = output["tokens"]
        output_grid = output.get("grid_size", output_grid)
    else:
        raise TypeError(
            "dino_encoder must return a Tensor or an object/mapping with `tokens`"
        )
    if not isinstance(tokens, torch.Tensor) or not torch.is_floating_point(tokens):
        raise TypeError("DINO patch tokens must be a floating-point torch.Tensor")
    if tokens.ndim not in (3, 4):
        raise ValueError(
            "DINO tokens must have shape [B,H,W,D] or [B,N,D], got "
            f"{tuple(tokens.shape)}"
        )
    grid = declared_grid if output_grid is None else tuple(int(v) for v in output_grid)
    if tokens.ndim == 4:
        inferred = (int(tokens.shape[1]), int(tokens.shape[2]))
        if grid is not None and tuple(grid) != inferred:
            raise ValueError(
                f"DINO output grid {inferred} disagrees with declared grid {grid}"
            )
        grid = inferred
    else:
        if grid is None:
            raise ValueError("flattened DINO tokens [B,N,D] require dino_grid_size")
        if len(grid) != 2 or grid[0] * grid[1] != tokens.shape[1]:
            raise ValueError(f"DINO grid {grid} does not match N={tokens.shape[1]}")
    return tokens, grid


def _resolve_wan_grid(
    configured: tuple[int, int] | Mapping[str, Sequence[int]], camera_key: str
) -> tuple[int, int]:
    if isinstance(configured, Mapping):
        if camera_key not in configured:
            raise KeyError(f"wan_grid_size has no entry for camera {camera_key!r}")
        value = configured[camera_key]
    else:
        value = configured
    grid = tuple(int(v) for v in value)
    if len(grid) != 2 or any(v <= 0 for v in grid):
        raise ValueError(f"invalid WAN grid for {camera_key!r}: {grid}")
    return grid


class RGBMotionSequencePreprocessor:
    """Convert aligned RGB-D camera sequences into one canonical sidecar.

    The insertion order of ``cameras`` is the semantic camera order unless
    ``camera_keys`` was supplied to the constructor.  Per-camera WAN grids are
    concatenated along width and *then* one global top-k is selected per time
    step; no camera gets an implicit private token budget. ``anchor_indices``
    index the raw ``T`` dimension of every camera sequence. ``world_time_ids``
    are deliberately supplied by the caller and are not derived from those raw
    indices, so the caller is responsible for using the same time unit as the
    cache/server. Under ``require_previous``, each supplied previous frame must
    be the temporal context immediately before the first selected anchor.
    """

    def __init__(
        self,
        dino_encoder: Any,
        detector: Any,
        *,
        max_tokens: int,
        wan_grid_size: tuple[int, int] | Mapping[str, Sequence[int]] = (8, 8),
        first_frame_policy: FirstFramePolicy = "require_previous",
        camera_keys: Sequence[str] | None = None,
        patch_size: Sequence[int] = (1, 2, 2),
    ) -> None:
        if not callable(dino_encoder):
            raise TypeError("dino_encoder must be callable")
        if not callable(detector):
            raise TypeError("detector must be callable")
        if (
            isinstance(max_tokens, bool)
            or int(max_tokens) != max_tokens
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if first_frame_policy not in ("require_previous", "empty", "all"):
            raise ValueError(
                "first_frame_policy must be require_previous, empty, or all"
            )
        patch = tuple(int(v) for v in patch_size)
        if len(patch) != 3 or any(v <= 0 for v in patch):
            raise ValueError(f"patch_size must contain 3 positive values, got {patch}")
        if patch[0] != 1:
            raise ValueError(
                "RGB-motion sequence preprocessing requires patch_size[0] == 1"
            )
        detector_budget = getattr(detector, "max_tokens", None)
        if detector_budget is not None:
            raise ValueError(
                "detector.max_tokens must be None: token limiting is applied only "
                "after all cameras are concatenated"
            )

        self.dino_encoder = dino_encoder
        self.detector = detector
        self.max_tokens = int(max_tokens)
        self.wan_grid_size = wan_grid_size
        self.first_frame_policy = first_frame_policy
        self.camera_keys = None if camera_keys is None else tuple(camera_keys)
        if self.camera_keys is not None and len(set(self.camera_keys)) != len(
            self.camera_keys
        ):
            raise ValueError("camera_keys must not contain duplicates")
        self.patch_size = patch

    def __call__(
        self,
        cameras: Mapping[str, RGBDCameraSequence],
        *,
        anchor_indices: Sequence[int] | torch.Tensor,
        world_time_ids: Sequence[int] | torch.Tensor,
        previous_frames: (
            Mapping[str, RGBDCameraSequence | Mapping[str, Any]] | None
        ) = None,
        observation_flag: int | Sequence[int] | torch.Tensor = 1,
    ) -> dict[str, Any]:
        return self.process(
            cameras,
            anchor_indices=anchor_indices,
            world_time_ids=world_time_ids,
            previous_frames=previous_frames,
            observation_flag=observation_flag,
        )

    @torch.no_grad()
    def process(
        self,
        cameras: Mapping[str, RGBDCameraSequence],
        *,
        anchor_indices: Sequence[int] | torch.Tensor,
        world_time_ids: Sequence[int] | torch.Tensor,
        previous_frames: (
            Mapping[str, RGBDCameraSequence | Mapping[str, Any]] | None
        ) = None,
        observation_flag: int | Sequence[int] | torch.Tensor = 1,
    ) -> dict[str, Any]:
        if not isinstance(cameras, Mapping) or not cameras:
            raise ValueError("cameras must be a non-empty ordered mapping")
        if not all(isinstance(value, RGBDCameraSequence) for value in cameras.values()):
            raise TypeError("every cameras value must be RGBDCameraSequence")

        if self.camera_keys is None:
            camera_keys = tuple(cameras.keys())
        else:
            camera_keys = self.camera_keys
            if set(camera_keys) != set(cameras.keys()) or len(camera_keys) != len(
                cameras
            ):
                raise ValueError(
                    f"camera set mismatch: expected {list(camera_keys)}, got "
                    f"{list(cameras.keys())}"
                )

        anchors = _integer_vector(anchor_indices, name="anchor_indices")
        if anchors.numel() < 1:
            raise ValueError("anchor_indices must contain at least one frame")
        if (anchors < 0).any():
            raise ValueError("anchor_indices must be non-negative")
        if anchors.numel() > 1 and not torch.all(anchors[1:] > anchors[:-1]):
            raise ValueError("anchor_indices must be strictly increasing")
        world_times = _integer_vector(
            world_time_ids, name="world_time_ids", expected=anchors.numel()
        )
        if (world_times < 0).any():
            raise ValueError("world_time_ids must be non-negative")
        if world_times.numel() > 1 and not torch.all(
            world_times[1:] > world_times[:-1]
        ):
            raise ValueError("world_time_ids must be strictly increasing")
        num_frames = int(anchors.numel())

        if self.first_frame_policy == "require_previous":
            if previous_frames is None:
                raise ValueError(
                    "first_frame_policy='require_previous' needs previous_frames"
                )
            missing_previous = [
                key for key in camera_keys if key not in previous_frames
            ]
            if missing_previous:
                raise KeyError(f"previous_frames is missing cameras {missing_previous}")

        camera_masks: list[torch.Tensor] = []
        camera_scores: list[torch.Tensor] = []
        camera_dino: list[torch.Tensor] = []
        camera_grids: dict[str, tuple[int, int]] = {}
        dino_grids: dict[str, tuple[int, int]] = {}
        output_device: torch.device | None = None
        output_feature_width: int | None = None
        common_height: int | None = None

        for camera_key in camera_keys:
            sequence = cameras[camera_key]
            if int(anchors[-1]) >= sequence.num_frames:
                raise IndexError(
                    f"anchor {int(anchors[-1])} is outside camera {camera_key!r} "
                    f"sequence length {sequence.num_frames}"
                )
            anchor_on_rgb = anchors.to(sequence.rgb.device)
            rgb = sequence.rgb.index_select(0, anchor_on_rgb)
            depth = sequence.depth.index_select(0, anchors.to(sequence.depth.device))
            poses = _select_matrix_series(
                sequence.camera_pose,
                anchors,
                sequence.num_frames,
                (4, 4),
                f"{camera_key}.camera_pose",
            )
            intrinsics = _select_matrix_series(
                sequence.camera_intrinsics,
                anchors,
                sequence.num_frames,
                (3, 3),
                f"{camera_key}.camera_intrinsics",
            )
            dino, dino_grid = _encoder_tokens(
                self.dino_encoder,
                rgb,
                declared_grid=sequence.dino_grid_size,
            )
            if dino.shape[0] != num_frames:
                raise ValueError(
                    f"DINO encoder returned B={dino.shape[0]} for {num_frames} anchors"
                )
            assert dino_grid is not None
            dino_grids[camera_key] = tuple(dino_grid)
            grid = _resolve_wan_grid(self.wan_grid_size, camera_key)
            camera_grids[camera_key] = grid
            if common_height is None:
                common_height = grid[0]
            elif grid[0] != common_height:
                raise ValueError(
                    "camera WAN grids must have equal heights for width "
                    f"concatenation, got {common_height} and {grid[0]}"
                )

            pooled_dino = pool_dino_to_grid(
                dino, grid, source_grid_size=dino_grid
            ).float()
            if output_device is None:
                output_device = pooled_dino.device
                output_feature_width = int(pooled_dino.shape[-1])
            elif pooled_dino.device != output_device:
                raise ValueError("all cameras' DINO outputs must share one device")
            elif pooled_dino.shape[-1] != output_feature_width:
                raise ValueError("all cameras' DINO feature widths must match")

            mask = torch.zeros(
                (num_frames, *grid), dtype=torch.bool, device=pooled_dino.device
            )
            score = torch.zeros(
                (num_frames, *grid), dtype=torch.float32, device=pooled_dino.device
            )

            if self.first_frame_policy == "require_previous":
                assert previous_frames is not None
                prev_rgb, prev_depth, prev_pose, prev_intrinsics = _previous_values(
                    previous_frames[camera_key], sequence
                )
                prev_dino, prev_grid = _encoder_tokens(
                    self.dino_encoder,
                    prev_rgb,
                    declared_grid=sequence.dino_grid_size,
                )
                if prev_dino.shape[0] != 1 or prev_grid != dino_grid:
                    raise ValueError(
                        f"previous/current DINO grid mismatch for {camera_key!r}"
                    )
                dino_previous = torch.cat((prev_dino, dino[:-1]), dim=0)
                dino_current = dino
                depth_previous = torch.cat(
                    (prev_depth.to(depth.device), depth[:-1]), dim=0
                )
                pose_previous = torch.cat(
                    (prev_pose.to(poses.device), poses[:-1]), dim=0
                )
                intrinsics_previous = torch.cat(
                    (prev_intrinsics.to(intrinsics.device), intrinsics[:-1]), dim=0
                )
                detect_start = 0
            elif num_frames > 1:
                dino_previous = dino[:-1]
                dino_current = dino[1:]
                depth_previous = depth[:-1]
                pose_previous = poses[:-1]
                intrinsics_previous = intrinsics[:-1]
                detect_start = 1
            else:
                dino_previous = dino[:0]
                dino_current = dino[:0]
                depth_previous = depth[:0]
                pose_previous = poses[:0]
                intrinsics_previous = intrinsics[:0]
                detect_start = 1

            if dino_current.shape[0] > 0:
                depth_current = depth[detect_start:]
                pose_current = poses[detect_start:]
                intrinsics_current = intrinsics[detect_start:]
                # Geometry and DINO must meet on the detector output device.
                detector_device = dino_current.device
                result = self.detector(
                    dino_previous,
                    dino_current,
                    depth_previous.to(detector_device, dtype=torch.float32),
                    depth_current.to(detector_device, dtype=torch.float32),
                    pose_previous.to(detector_device, dtype=torch.float32),
                    pose_current.to(detector_device, dtype=torch.float32),
                    intrinsics_previous.to(detector_device, dtype=torch.float32),
                    dino_grid_size=dino_grid,
                    wan_grid_size=grid,
                    camera_intrinsics_current=intrinsics_current.to(
                        detector_device, dtype=torch.float32
                    ),
                )
                result_mask = result.wan_motion_mask
                result_score = result.wan_motion_score
                expected = (dino_current.shape[0], *grid)
                if (
                    tuple(result_mask.shape) != expected
                    or result_mask.dtype != torch.bool
                ):
                    raise ValueError(
                        "detector wan_motion_mask must be bool with shape "
                        f"{expected}, got {tuple(result_mask.shape)}"
                    )
                if tuple(result_score.shape) != expected or not torch.is_floating_point(
                    result_score
                ):
                    raise ValueError(
                        "detector wan_motion_score must be floating point with "
                        f"shape {expected}, got {tuple(result_score.shape)}"
                    )
                mask[detect_start:] = result_mask.to(mask.device)
                score[detect_start:] = result_score.to(
                    score.device, dtype=torch.float32
                )

            if self.first_frame_policy == "all":
                mask[0] = True
                score[0] = 1.0

            camera_masks.append(mask)
            camera_scores.append(score)
            camera_dino.append(pooled_dino)

        assert output_device is not None and common_height is not None
        global_mask = torch.cat(camera_masks, dim=2)
        global_score = torch.cat(camera_scores, dim=2)
        global_dino = torch.cat(camera_dino, dim=2)
        global_grid = (common_height, int(global_mask.shape[2]))
        spatial_tokens = global_grid[0] * global_grid[1]
        if self.first_frame_policy == "all" and self.max_tokens < spatial_tokens:
            raise ValueError(
                "first_frame_policy='all' requires max_tokens >= the complete "
                f"multi-camera grid ({spatial_tokens}), got {self.max_tokens}"
            )

        indices = torch.full(
            (num_frames, self.max_tokens),
            -1,
            dtype=torch.long,
            device=output_device,
        )
        valid = torch.zeros(
            (num_frames, self.max_tokens), dtype=torch.bool, device=output_device
        )
        flat_mask = global_mask.reshape(num_frames, -1)
        flat_score = global_score.reshape(num_frames, -1)
        for frame_index in range(num_frames):
            candidates = torch.nonzero(flat_mask[frame_index], as_tuple=False).flatten()
            if candidates.numel():
                order = torch.argsort(
                    flat_score[frame_index, candidates],
                    descending=True,
                    stable=True,
                )
                selected = candidates[order[: self.max_tokens]]
                count = int(selected.numel())
                indices[frame_index, :count] = selected
                valid[frame_index, :count] = True

        selected_scores = gather_grid_features(global_score[..., None], indices, valid)[
            ..., 0
        ]
        selected_dino = gather_grid_features(global_dino, indices, valid).float()

        world_time = (
            world_times.to(output_device)[:, None]
            .expand(num_frames, self.max_tokens)
            .clone()
        )
        world_time.masked_fill_(~valid, -1)
        source = (
            self._observation_flags(observation_flag, num_frames, output_device)[
                :, None
            ]
            .expand(num_frames, self.max_tokens)
            .clone()
        )
        source.masked_fill_(~valid, 0)

        canonical: dict[str, Any] = {
            "motion_indices": indices,
            "motion_valid_mask": valid,
            "motion_scores": selected_scores.masked_fill(~valid, 0),
            "world_time_id": world_time,
            "dino_features": selected_dino,
            "neoforce_features": torch.empty(
                (num_frames, self.max_tokens, 0),
                dtype=selected_dino.dtype,
                device=output_device,
            ),
            "observation_flag": source,
            "visual_valid": valid.clone(),
            "tactile_valid": torch.zeros_like(valid),
        }
        assert tuple(canonical) == _CANONICAL_FIELDS

        provenance = {
            "producer": "n0_twam.preprocessing.RGBMotionSequencePreprocessor",
            "camera_keys": list(camera_keys),
            "anchor_indices": anchors.tolist(),
            "world_time_ids": world_times.tolist(),
            "first_frame_policy": self.first_frame_policy,
            "max_tokens": self.max_tokens,
            "patch_size": list(self.patch_size),
            "camera_wan_grid_shapes": {
                key: list(camera_grids[key]) for key in camera_keys
            },
            "camera_dino_grid_shapes": {
                key: list(dino_grids[key]) for key in camera_keys
            },
            "dino_encoder": type(self.dino_encoder).__name__,
            "detector": type(self.detector).__name__,
        }
        # These top-level fields are consumed by the existing dataset loader to
        # reject stale sidecars before their addresses reach the model.
        canonical.update(
            {
                "camera_keys": list(camera_keys),
                "patch_size": self.patch_size,
                "spatial_grid_shape": global_grid,
                "latent_num_frames": num_frames,
                "provenance": provenance,
            }
        )
        return canonical

    @staticmethod
    def _observation_flags(
        value: int | Sequence[int] | torch.Tensor,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            tensor = tensor.expand(num_frames)
        elif tensor.ndim != 1 or tensor.numel() != num_frames:
            raise ValueError(
                "observation_flag must be scalar or have one value per anchor"
            )
        if tensor.numel() and not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError("observation_flag values must be 0 or 1")
        return tensor.to(device=device, dtype=torch.long)


__all__ = ["RGBDCameraSequence", "RGBMotionSequencePreprocessor"]
