#!/usr/bin/env python3
"""Build canonical RGB-motion sidecars from explicitly aligned RGB-D bundles.

The N0-TWAM repository does not define a universal LeRobot schema for depth,
camera poses, or camera intrinsics.  This command therefore never guesses
dataset columns.  A JSON manifest names the already aligned raw bundles, the
per-camera WAN latent files used by training, and the exact raw-frame anchors
corresponding to WAN latent time steps.

All relative paths in the manifest are resolved relative to the manifest
file.  DINOv2 loading is local-only unless ``--allow-dino-download`` is passed
explicitly.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from n0_twam.models.rgb_motion import EgoMotionCompensatedMotionDetector
from n0_twam.preprocessing.dinov2 import FrozenDinoV2PatchEncoder
from n0_twam.preprocessing.rgb_motion_sequence import (
    RGBDCameraSequence,
    RGBMotionSequencePreprocessor,
)


MANIFEST_SCHEMA_VERSION = 1
CANONICAL_FIELDS = (
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
_DETECTOR_FIELDS = {
    "depth_threshold",
    "dino_threshold",
    "depth_weight",
    "dino_weight",
    "dilation_radius",
    "min_depth",
    "pose_convention",
}
_LATENT_TEMPORAL_PROVENANCE_SCHEMA_VERSION = 1
_LATENT_ANCHOR_SEMANTICS = "causal_chunk_end"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build N0-TWAM RGB-motion sidecars from manifest-aligned RGB-D "
            "tensor bundles."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Override manifest.output_root. CLI-relative paths use the current "
            "directory; manifest paths use the manifest directory."
        ),
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Override manifest.dino_model with a local path or cached model id.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--allow-dino-download",
        action="store_true",
        help="Explicitly permit Hugging Face network loading for DINOv2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RGB-motion manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in RGB-motion manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"RGB-motion manifest {path} must contain a JSON object")
    return value


def _load_torch_mapping(path: Path, label: str) -> dict[str, Any]:
    if path.suffix.lower() not in (".pt", ".pth"):
        raise ValueError(f"{label} must be a .pt or .pth file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} {path} must contain a mapping, got {type(value)}")
    return value


def _resolve_manifest_path(value: Any, manifest_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _validate_latent_chunk_path(
    path: Path,
    *,
    expected_chunk_index: int,
    label: str,
) -> None:
    """Require the resolved latent path to live below its manifest chunk.

    Official latent encoders write ``.../chunk-NNN/<camera>/<episode>.pth``.
    The camera directory means that the chunk directory is commonly an
    ancestor rather than the immediate parent of the file.  We inspect the
    nearest numeric ``chunk-*`` ancestor and reject both a different chunk and
    non-standard zero padding.  Paths with no numeric chunk ancestor are
    intentionally rejected instead of guessing one from the episode index.
    """

    expected_name = f"chunk-{expected_chunk_index:03d}"
    for parent in path.parents:
        name = parent.name
        if name.startswith("chunk-") and name.removeprefix("chunk-").isdigit():
            if name != expected_name:
                raise ValueError(
                    f"{label} resolves under {name!r}, but its segment "
                    f"chunk_index={expected_chunk_index} requires the standard "
                    f"ancestor {expected_name!r}: {path}"
                )
            return
    raise ValueError(
        f"{label} must be stored beneath the standard {expected_name!r} "
        f"ancestor matching its segment chunk_index; paths without a "
        f"chunk-NNN directory are not accepted: {path}"
    )


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    try:
        tensor = torch.as_tensor(value)
    except Exception as exc:
        raise TypeError(f"{label} must be an integer") from exc
    if tensor.ndim != 0 or tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{label} must be one integer scalar, not bool or a sequence")
    if torch.is_floating_point(tensor) and (
        not torch.isfinite(tensor) or tensor != tensor.round()
    ):
        raise ValueError(f"{label} must be a finite integer")
    result = int(tensor.item())
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {result}")
    return result


def _integer_vector(
    value: Any, label: str, *, nonnegative: bool = True
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except Exception as exc:
        raise TypeError(f"{label} must be an integer vector") from exc
    if tensor.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got {tuple(tensor.shape)}")
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{label} must contain integers")
    if torch.is_floating_point(tensor) and tensor.numel():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor, tensor.round()):
            raise ValueError(f"{label} must contain finite integers")
    tensor = tensor.to(dtype=torch.long, device="cpu")
    if nonnegative and tensor.numel() and (tensor < 0).any():
        raise ValueError(f"{label} must contain only non-negative values")
    return tensor


def _camera_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest.camera_keys must be a non-empty JSON list")
    if not all(isinstance(item, str) and item for item in value):
        raise TypeError("manifest.camera_keys entries must be non-empty strings")
    keys = tuple(value)
    if len(set(keys)) != len(keys):
        raise ValueError("manifest.camera_keys must not contain duplicates")
    return keys


def _ordered_camera_mapping(
    value: Any, camera_keys: tuple[str, ...], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping keyed by camera name")
    actual = tuple(value.keys())
    if actual != camera_keys:
        raise ValueError(
            f"{label} camera order mismatch: expected {list(camera_keys)}, "
            f"got {list(actual)}"
        )
    return value


def _rgb_spatial_shape(rgb: torch.Tensor, label: str) -> tuple[int, int]:
    if rgb.ndim != 4:
        raise ValueError(f"{label}.rgb must be 4-D, got {tuple(rgb.shape)}")
    channels_first = rgb.shape[1] == 3
    channels_last = rgb.shape[-1] == 3
    if channels_first == channels_last:
        raise ValueError(
            f"{label}.rgb must be unambiguous [T,3,H,W] or [T,H,W,3], "
            f"got {tuple(rgb.shape)}"
        )
    if channels_first:
        return int(rgb.shape[2]), int(rgb.shape[3])
    return int(rgb.shape[1]), int(rgb.shape[2])


def _depth_spatial_shape(depth: torch.Tensor, label: str) -> tuple[int, int]:
    if not torch.is_floating_point(depth):
        raise TypeError(
            f"{label}.depth must be calibrated floating-point z-depth; "
            "the builder will not guess a uint16 scale"
        )
    if depth.ndim == 3:
        return int(depth.shape[1]), int(depth.shape[2])
    if depth.ndim == 4 and depth.shape[1] == 1:
        return int(depth.shape[2]), int(depth.shape[3])
    raise ValueError(
        f"{label}.depth must be [T,H,W] or [T,1,H,W], got {tuple(depth.shape)}"
    )


def _camera_sequence(value: Any, label: str) -> RGBDCameraSequence:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a field mapping")
    required = ("rgb", "depth", "camera_pose", "camera_intrinsics")
    missing = [field for field in required if field not in value]
    if missing:
        raise KeyError(f"{label} is missing required fields {missing}")
    rgb = torch.as_tensor(value["rgb"])
    depth = torch.as_tensor(value["depth"])
    rgb_shape = _rgb_spatial_shape(rgb, label)
    depth_shape = _depth_spatial_shape(depth, label)
    if rgb_shape != depth_shape:
        raise ValueError(
            f"{label} RGB/depth must be pixel-aligned, got {rgb_shape} and "
            f"{depth_shape}"
        )
    grid_value = value.get("dino_grid_size")
    dino_grid = None
    if grid_value is not None:
        dino_grid_tensor = _integer_vector(
            grid_value, f"{label}.dino_grid_size", nonnegative=False
        )
        if dino_grid_tensor.numel() != 2 or (dino_grid_tensor <= 0).any():
            raise ValueError(f"{label}.dino_grid_size must contain two positive values")
        dino_grid = tuple(int(item) for item in dino_grid_tensor.tolist())
    return RGBDCameraSequence(
        rgb=rgb,
        depth=depth,
        camera_pose=torch.as_tensor(value["camera_pose"]),
        camera_intrinsics=torch.as_tensor(value["camera_intrinsics"]),
        dino_grid_size=dino_grid,
    )


def _previous_frames(
    bundle: Mapping[str, Any],
    camera_keys: tuple[str, ...],
    policy: str,
) -> Mapping[str, Mapping[str, torch.Tensor]] | None:
    raw = bundle.get("previous_frames")
    if raw is None:
        if policy == "require_previous":
            raise KeyError(
                "bundle.previous_frames is required when first_frame_policy is "
                "'require_previous'"
            )
        return None
    ordered = _ordered_camera_mapping(raw, camera_keys, "bundle.previous_frames")
    converted: dict[str, Mapping[str, torch.Tensor]] = {}
    for key in camera_keys:
        frame = ordered[key]
        if not isinstance(frame, Mapping):
            raise TypeError(f"bundle.previous_frames[{key!r}] must be a mapping")
        required = ("rgb", "depth", "camera_pose", "camera_intrinsics")
        missing = [field for field in required if field not in frame]
        if missing:
            raise KeyError(
                f"bundle.previous_frames[{key!r}] is missing fields {missing}"
            )
        converted_frame = {field: torch.as_tensor(frame[field]) for field in required}
        if not torch.is_floating_point(converted_frame["depth"]):
            raise TypeError(
                f"bundle.previous_frames[{key!r}].depth must be calibrated "
                "floating-point z-depth; the builder will not guess an integer scale"
            )
        converted[key] = converted_frame
    return converted


def _patch_size(value: Any) -> tuple[int, int, int]:
    patch = _integer_vector(value, "manifest.patch_size", nonnegative=False)
    if patch.numel() != 3 or (patch <= 0).any():
        raise ValueError("manifest.patch_size must contain three positive integers")
    result = tuple(int(item) for item in patch.tolist())
    if result[0] != 1:
        raise ValueError("RGB-motion sidecars require manifest.patch_size[0] == 1")
    return result


def _manifest_wan_grids(
    value: Any, camera_keys: tuple[str, ...]
) -> Mapping[str, tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        ordered = _ordered_camera_mapping(value, camera_keys, "manifest.wan_grid_size")
        raw_grids = ordered
    else:
        raw_grids = {key: value for key in camera_keys}
    result: dict[str, tuple[int, int]] = {}
    for key in camera_keys:
        grid = _integer_vector(
            raw_grids[key], f"manifest.wan_grid_size[{key!r}]", nonnegative=False
        )
        if grid.numel() != 2 or (grid <= 0).any():
            raise ValueError(
                f"manifest.wan_grid_size[{key!r}] must have two positive integers"
            )
        result[key] = tuple(int(item) for item in grid.tolist())
    return result


def _latent_metadata(
    path: Path,
    *,
    label: str,
    expected_filename: str,
    bundle_frame_ids: torch.Tensor,
    expected_anchor_indices: torch.Tensor,
    expected_frames: int,
    patch_size: tuple[int, int, int],
) -> tuple[int, int]:
    if path.name != expected_filename:
        raise ValueError(
            f"{label} filename must match the sidecar segment name "
            f"{expected_filename!r}, got {path.name!r}"
        )
    payload = _load_torch_mapping(path, label)
    required = ("latent_num_frames", "latent_height", "latent_width", "frame_ids")
    missing = [field for field in required if field not in payload]
    if missing:
        raise KeyError(f"{label} {path} is missing metadata fields {missing}")
    latent_frames = _strict_int(
        payload["latent_num_frames"], f"{label}.latent_num_frames", minimum=1
    )
    if latent_frames != expected_frames:
        raise ValueError(
            f"{label} latent frame count {latent_frames} does not match the "
            f"{expected_frames} explicit anchors"
        )
    latent_height = _strict_int(
        payload["latent_height"], f"{label}.latent_height", minimum=1
    )
    latent_width = _strict_int(
        payload["latent_width"], f"{label}.latent_width", minimum=1
    )
    if latent_height % patch_size[1] or latent_width % patch_size[2]:
        raise ValueError(
            f"{label} latent spatial shape {(latent_height, latent_width)} is not "
            f"divisible by patch_size {patch_size[1:]}"
        )
    latent_frame_ids = _integer_vector(payload["frame_ids"], f"{label}.frame_ids")
    if not torch.equal(latent_frame_ids, bundle_frame_ids):
        raise ValueError(
            f"{label}.frame_ids does not match bundle.frame_ids; raw RGB-D and "
            "WAN latent inputs are not aligned"
        )
    if "video_num_frames" in payload:
        video_frames = _strict_int(
            payload["video_num_frames"], f"{label}.video_num_frames", minimum=1
        )
        if video_frames != bundle_frame_ids.numel():
            raise ValueError(
                f"{label}.video_num_frames={video_frames} does not match "
                f"bundle frame count {bundle_frame_ids.numel()}"
            )

    temporal = payload.get("temporal_provenance")
    if not isinstance(temporal, Mapping):
        raise KeyError(
            f"{label} {path} is missing temporal_provenance. Re-encode this "
            "latent with the current encode_lerobot_n0_latents.py --overwrite; "
            "the sidecar builder will not guess WAN temporal anchors from a "
            "manifest alone."
        )
    temporal_required = (
        "schema_version",
        "anchor_semantics",
        "temporal_stride",
        "latent_anchor_indices",
        "latent_anchor_frame_ids",
    )
    temporal_missing = [field for field in temporal_required if field not in temporal]
    if temporal_missing:
        raise KeyError(
            f"{label}.temporal_provenance is missing fields {temporal_missing}"
        )
    schema_version = _strict_int(
        temporal["schema_version"],
        f"{label}.temporal_provenance.schema_version",
        minimum=1,
    )
    if schema_version != _LATENT_TEMPORAL_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"{label}.temporal_provenance.schema_version={schema_version} is "
            f"unsupported; expected {_LATENT_TEMPORAL_PROVENANCE_SCHEMA_VERSION}"
        )
    if temporal["anchor_semantics"] != _LATENT_ANCHOR_SEMANTICS:
        raise ValueError(
            f"{label}.temporal_provenance.anchor_semantics must be "
            f"{_LATENT_ANCHOR_SEMANTICS!r}, got "
            f"{temporal['anchor_semantics']!r}"
        )
    temporal_stride = _strict_int(
        temporal["temporal_stride"],
        f"{label}.temporal_provenance.temporal_stride",
        minimum=1,
    )
    latent_anchor_indices = _integer_vector(
        temporal["latent_anchor_indices"],
        f"{label}.temporal_provenance.latent_anchor_indices",
    )
    latent_anchor_frame_ids = _integer_vector(
        temporal["latent_anchor_frame_ids"],
        f"{label}.temporal_provenance.latent_anchor_frame_ids",
        nonnegative=False,
    )
    expected_schedule = torch.arange(latent_frames, dtype=torch.long) * temporal_stride
    expected_video_frames = 1 + (latent_frames - 1) * temporal_stride
    if bundle_frame_ids.numel() != expected_video_frames:
        raise ValueError(
            f"{label}.temporal_provenance describes stride {temporal_stride} "
            f"and {latent_frames} latent frames, which requires "
            f"{expected_video_frames} input frames, not {bundle_frame_ids.numel()}"
        )
    if not torch.equal(latent_anchor_indices, expected_schedule):
        raise ValueError(
            f"{label}.temporal_provenance.latent_anchor_indices is internally "
            f"inconsistent with causal stride {temporal_stride}: expected "
            f"{expected_schedule.tolist()}, got {latent_anchor_indices.tolist()}"
        )
    if not torch.equal(latent_anchor_indices, expected_anchor_indices):
        raise ValueError(
            f"{label} manifest anchor_indices disagrees with latent temporal "
            f"provenance: manifest={expected_anchor_indices.tolist()}, latent="
            f"{latent_anchor_indices.tolist()}"
        )
    expected_anchor_frame_ids = latent_frame_ids.index_select(
        0, latent_anchor_indices
    )
    if not torch.equal(latent_anchor_frame_ids, expected_anchor_frame_ids):
        raise ValueError(
            f"{label}.temporal_provenance.latent_anchor_frame_ids does not "
            "match frame_ids at latent_anchor_indices"
        )
    return latent_height // patch_size[1], latent_width // patch_size[2]


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _model_source(value: str, manifest_dir: Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    manifest_relative = (manifest_dir / candidate).resolve()
    if manifest_relative.exists():
        return str(manifest_relative)
    # Keep a cached Hugging Face model id such as facebook/dinov2-base intact.
    return value


def _make_dino_encoder(
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    *,
    model_override: str | None,
    device: str,
    dtype: torch.dtype,
    allow_download: bool,
) -> FrozenDinoV2PatchEncoder:
    source = (
        model_override if model_override is not None else manifest.get("dino_model")
    )
    if not isinstance(source, str) or not source:
        raise KeyError(
            "manifest.dino_model (or --dino-model) is required when no DINO "
            "encoder is injected"
        )
    image_size = manifest.get("dino_image_size", [224, 224])
    return FrozenDinoV2PatchEncoder.from_pretrained(
        _model_source(source, manifest_dir),
        local_files_only=not allow_download,
        device=device,
        torch_dtype=dtype,
        image_size=image_size,
        float_input_range=manifest.get("dino_float_input_range", "0_1"),
        output_dtype=torch.float32,
    )


def _make_detector(manifest: Mapping[str, Any]) -> EgoMotionCompensatedMotionDetector:
    config = manifest.get("detector", {})
    if not isinstance(config, Mapping):
        raise TypeError("manifest.detector must be a JSON object")
    unknown = set(config) - _DETECTOR_FIELDS
    if unknown:
        raise ValueError(
            f"manifest.detector has unsupported fields {sorted(unknown)}; "
            "max_tokens is intentionally global and belongs at manifest.max_tokens"
        )
    return EgoMotionCompensatedMotionDetector(max_tokens=None, **dict(config))


def build_rgb_motion_sidecars(
    manifest_path: str | Path,
    *,
    output_root: str | Path | None = None,
    dino_model: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    allow_dino_download: bool = False,
    overwrite: bool = False,
    dino_encoder: Any | None = None,
    detector: Any | None = None,
) -> list[Path]:
    """Validate all inputs and write one canonical sidecar per manifest segment.

    ``dino_encoder`` and ``detector`` are injectable for CPU tests or custom
    runtimes.  In normal CLI use, DINO is loaded locally by default and the
    ego-motion detector is built from ``manifest.detector``.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = _load_json_mapping(manifest_path)
    version = _strict_int(manifest.get("schema_version", -1), "manifest.schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest.schema_version={version}; expected "
            f"{MANIFEST_SCHEMA_VERSION}"
        )
    camera_keys = _camera_keys(manifest.get("camera_keys"))
    max_tokens = _strict_int(
        manifest.get("max_tokens"), "manifest.max_tokens", minimum=1
    )
    patch_size = _patch_size(manifest.get("patch_size", [1, 2, 2]))
    configured_grids = _manifest_wan_grids(manifest.get("wan_grid_size"), camera_keys)

    if output_root is None:
        if "output_root" not in manifest:
            raise KeyError("manifest.output_root or --output-root is required")
        resolved_output_root = _resolve_manifest_path(
            manifest["output_root"], manifest_dir, "manifest.output_root"
        )
    else:
        resolved_output_root = Path(output_root).expanduser().resolve()
    if resolved_output_root.exists() and not resolved_output_root.is_dir():
        raise NotADirectoryError(
            f"RGB-motion output root is not a directory: {resolved_output_root}"
        )

    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("manifest.segments must be a non-empty JSON list")

    # Resolve every destination before model loading, catching duplicate or
    # malformed segment identities without wasting a DINO initialization.
    resolved_segments: list[
        tuple[Mapping[str, Any], Path, str, dict[str, Path]]
    ] = []
    seen_outputs: set[Path] = set()
    for index, segment in enumerate(segments):
        label = f"manifest.segments[{index}]"
        if not isinstance(segment, Mapping):
            raise TypeError(f"{label} must be a JSON object")
        episode = _strict_int(segment.get("episode_index"), f"{label}.episode_index")
        chunk = _strict_int(segment.get("chunk_index"), f"{label}.chunk_index")
        start = _strict_int(segment.get("start_frame"), f"{label}.start_frame")
        end = _strict_int(segment.get("end_frame"), f"{label}.end_frame", minimum=1)
        if end <= start:
            raise ValueError(f"{label}.end_frame must be greater than start_frame")
        filename = f"episode_{episode:06d}_{start}_{end}.pth"
        destination = (resolved_output_root / f"chunk-{chunk:03d}" / filename).resolve()
        if destination in seen_outputs:
            raise ValueError(f"duplicate RGB-motion output path: {destination}")
        seen_outputs.add(destination)

        # Resolve and validate the latent address contract before DINO is
        # initialized.  This is a lexical/provenance check only: payload I/O is
        # deferred until we know the segment actually needs to be built.
        latent_files = _ordered_camera_mapping(
            segment.get("latent_files"), camera_keys, f"{label}.latent_files"
        )
        latent_paths: dict[str, Path] = {}
        for key in camera_keys:
            latent_label = f"{label}.latent_files[{key!r}]"
            latent_path = _resolve_manifest_path(
                latent_files[key], manifest_dir, latent_label
            )
            _validate_latent_chunk_path(
                latent_path,
                expected_chunk_index=chunk,
                label=latent_label,
            )
            latent_paths[key] = latent_path
        if len(set(latent_paths.values())) != len(latent_paths):
            raise ValueError(
                f"{label}.latent_files must name one distinct file per camera"
            )
        resolved_segments.append((segment, destination, label, latent_paths))

    pending_segments = []
    for resolved in resolved_segments:
        destination = resolved[1]
        if destination.exists() and not overwrite:
            print(f"skip existing {destination}")
        else:
            pending_segments.append(resolved)

    # In particular, do not require or initialize a DINO checkpoint for an
    # idempotent invocation whose outputs are already complete.
    if not pending_segments:
        return []

    if dino_encoder is None:
        dino_encoder = _make_dino_encoder(
            manifest,
            manifest_dir,
            model_override=dino_model,
            device=device,
            dtype=dtype,
            allow_download=allow_dino_download,
        )
    if detector is None:
        detector = _make_detector(manifest)
    detector_budget = getattr(detector, "max_tokens", None)
    if detector_budget is not None:
        raise ValueError(
            "detector.max_tokens must be None; max_tokens is applied globally "
            "after camera grids are concatenated"
        )

    written: list[Path] = []
    reference_grids: dict[str, tuple[int, int]] | None = None
    for segment, destination, label, latent_paths in pending_segments:
        bundle_path = _resolve_manifest_path(
            segment.get("bundle"), manifest_dir, f"{label}.bundle"
        )
        bundle = _load_torch_mapping(bundle_path, "RGB-D bundle")
        raw_cameras = _ordered_camera_mapping(
            bundle.get("cameras"), camera_keys, "bundle.cameras"
        )
        cameras = {
            key: _camera_sequence(raw_cameras[key], f"bundle.cameras[{key!r}]")
            for key in camera_keys
        }
        lengths = {key: value.num_frames for key, value in cameras.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"bundle camera sequence lengths must match, got {lengths}"
            )
        raw_frame_count = next(iter(lengths.values()))
        if "frame_ids" not in bundle:
            raise KeyError(
                "bundle.frame_ids is required to prove RGB-D/WAN latent alignment"
            )
        bundle_frame_ids = _integer_vector(bundle["frame_ids"], "bundle.frame_ids")
        if bundle_frame_ids.numel() != raw_frame_count:
            raise ValueError(
                f"bundle.frame_ids has {bundle_frame_ids.numel()} entries but "
                f"camera sequences have {raw_frame_count} frames"
            )
        anchors = _integer_vector(
            segment.get("anchor_indices"), f"{label}.anchor_indices"
        )
        world_times = _integer_vector(
            segment.get("world_time_ids"), f"{label}.world_time_ids"
        )
        if anchors.numel() < 1:
            raise ValueError(f"{label}.anchor_indices must not be empty")
        if anchors.numel() != world_times.numel():
            raise ValueError(
                f"{label}.anchor_indices and world_time_ids must have equal "
                f"length, got {anchors.numel()} and {world_times.numel()}"
            )
        if anchors.numel() > 1 and not torch.all(anchors[1:] > anchors[:-1]):
            raise ValueError(f"{label}.anchor_indices must be strictly increasing")
        if world_times.numel() > 1 and not torch.all(
            world_times[1:] > world_times[:-1]
        ):
            raise ValueError(f"{label}.world_time_ids must be strictly increasing")
        if int(anchors[-1]) >= raw_frame_count:
            raise IndexError(
                f"{label}.anchor_indices ends at {int(anchors[-1])}, outside "
                f"bundle length {raw_frame_count}"
            )

        expected_filename = destination.name
        inferred_grids: dict[str, tuple[int, int]] = {}
        for key in camera_keys:
            latent_path = latent_paths[key]
            if latent_path == bundle_path or latent_path == destination:
                raise ValueError(
                    f"{label}.latent_files[{key!r}] must not alias the bundle "
                    "or output sidecar"
                )
            inferred_grids[key] = _latent_metadata(
                latent_path,
                label=f"latent_files[{key!r}]",
                expected_filename=expected_filename,
                bundle_frame_ids=bundle_frame_ids,
                expected_anchor_indices=anchors,
                expected_frames=int(anchors.numel()),
                patch_size=patch_size,
            )
        heights = {grid[0] for grid in inferred_grids.values()}
        if len(heights) != 1:
            raise ValueError(
                "per-camera WAN grids must share one height for width "
                f"concatenation, got {inferred_grids}"
            )
        if configured_grids is not None and dict(configured_grids) != inferred_grids:
            raise ValueError(
                "manifest.wan_grid_size disagrees with latent metadata: "
                f"manifest={dict(configured_grids)}, inferred={inferred_grids}"
            )
        if reference_grids is None:
            reference_grids = inferred_grids
        elif reference_grids != inferred_grids:
            raise ValueError(
                "WAN grid shapes changed between manifest segments: "
                f"expected {reference_grids}, got {inferred_grids}"
            )

        policy = segment.get(
            "first_frame_policy",
            manifest.get("first_frame_policy", "require_previous"),
        )
        if policy not in ("require_previous", "empty", "all"):
            raise ValueError(
                f"{label}.first_frame_policy must be require_previous, empty, or all"
            )
        previous = _previous_frames(bundle, camera_keys, policy)
        preprocessor = RGBMotionSequencePreprocessor(
            dino_encoder,
            detector,
            max_tokens=max_tokens,
            wan_grid_size=inferred_grids,
            first_frame_policy=policy,
            camera_keys=camera_keys,
            patch_size=patch_size,
        )
        sidecar = preprocessor.process(
            cameras,
            anchor_indices=anchors,
            world_time_ids=world_times,
            previous_frames=previous,
            observation_flag=1,
        )
        missing_fields = [field for field in CANONICAL_FIELDS if field not in sidecar]
        if missing_fields:
            raise RuntimeError(
                f"RGB-motion preprocessor omitted canonical fields {missing_fields}"
            )
        sidecar = _cpu_tree(sidecar)
        provenance = dict(sidecar.get("provenance", {}))
        provenance.update(
            {
                "builder": "script/build_rgb_motion_sidecars.py",
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest": str(manifest_path),
                "bundle": str(bundle_path),
                "latent_files": {key: str(latent_paths[key]) for key in camera_keys},
                "episode_index": _strict_int(
                    segment.get("episode_index"), f"{label}.episode_index"
                ),
                "start_frame": _strict_int(
                    segment.get("start_frame"), f"{label}.start_frame"
                ),
                "end_frame": _strict_int(
                    segment.get("end_frame"), f"{label}.end_frame", minimum=1
                ),
                "bundle_frame_ids": bundle_frame_ids.tolist(),
            }
        )
        sidecar["provenance"] = provenance
        _atomic_torch_save(sidecar, destination)
        written.append(destination)
        print(f"saved {destination}")
    return written


def main(
    argv: Sequence[str] | None = None,
    *,
    dino_encoder: Any | None = None,
    detector: Any | None = None,
) -> list[Path]:
    args = parse_args(argv)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    return build_rgb_motion_sidecars(
        args.manifest,
        output_root=args.output_root,
        dino_model=args.dino_model,
        device=args.device,
        dtype=dtype,
        allow_dino_download=args.allow_dino_download,
        overwrite=args.overwrite,
        dino_encoder=dino_encoder,
        detector=detector,
    )


if __name__ == "__main__":
    main()
