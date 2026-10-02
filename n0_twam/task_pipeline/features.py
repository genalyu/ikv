"""Bounded-memory RGB-native sidecars using the online motion implementation."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .config import paths, fingerprint


def selected_rgb(path, frame_ids):
    import av
    import cv2

    wanted = iter(frame_ids)
    target = next(wanted, None)
    with av.open(str(path)) as video:
        for i, frame in enumerate(video.decode(video=0)):
            if target == i:
                # Match the official RGB latent encoder's full-field resize.
                yield cv2.resize(
                    frame.to_ndarray(format="rgb24"),
                    (256, 256),
                    interpolation=cv2.INTER_AREA,
                )
                target = next(wanted, None)
                if target is None:
                    return
    raise ValueError(f"Video {path} ended before source frame {target}")


class RecordingEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.outputs = []

    def __call__(self, rgb):
        result = self.encoder(rgb)
        self.outputs.append(result.tokens.detach().cpu())
        return result


def build_payloads(
    streams, frame_ids, anchors, encoder, threshold=0.02, patch_size=(1, 2, 2)
):
    from n0_twam.preprocessing.rgb_frame_difference import (
        RGBFrameDifferencePreprocessor,
    )
    from n0_twam.models.rgb_motion import pool_dino_to_grid

    if (
        anchors != list(range(0, len(frame_ids), 4))
        or anchors[-1] != len(frame_ids) - 1
    ):
        raise ValueError("Expected official causal WAN anchors and full-frame coverage")
    keys = list(streams)
    recorder = RecordingEncoder(encoder)
    producer = RGBFrameDifferencePreprocessor(
        camera_keys=keys,
        height=256,
        width=256,
        patch_size=patch_size,
        threshold=threshold,
        dino_encoder=recorder,
    )
    previous = None
    start = 0
    parts = []
    dense = []
    gh = 256 // (16 * patch_size[1])
    gw = 256 // (16 * patch_size[2])
    for ordinal, end in enumerate(anchors):
        cameras = {}
        for key, stream in streams.items():
            frames = [next(stream) for _ in range(end - start + 1)]
            cameras[key] = SimpleNamespace(
                rgb=torch.from_numpy(np.stack(frames)), num_frames=len(frames)
            )
        recorder.outputs = []
        payload = producer(
            cameras,
            anchor_indices=[end - start],
            world_time_ids=[ordinal],
            previous_frames=previous,
        )
        dense.append(
            torch.cat(
                [pool_dino_to_grid(x, (gh, gw)) for x in recorder.outputs], dim=2
            ).reshape(1, gh * gw * len(keys), -1)
        )
        parts.append(payload)
        previous = {k: {"rgb": v.rgb[-1]} for k, v in cameras.items()}
        start = end + 1
    result = dict(parts[0])
    for key, value in parts[0].items():
        if isinstance(value, torch.Tensor):
            result[key] = torch.cat([p[key] for p in parts], dim=0)
    result.update(
        latent_num_frames=len(anchors),
        frame_ids=list(frame_ids),
        camera_wan_grid_shapes={key: [gh, gw] for key in keys},
    )
    result["provenance"].update(
        anchor_indices=anchors,
        producer="RGBFrameDifferencePreprocessor",
        first_frame_policy="all",
        input_mode="rgb",
    )
    dense_payload = dict(
        camera_keys=keys,
        patch_size=list(patch_size),
        spatial_grid_shape=[gh, gw * len(keys)],
        frame_ids=list(frame_ids),
        dino_features=torch.cat(dense),
        neoforce_features=torch.empty(len(anchors), gh * gw * len(keys), 0),
    )
    return result, dense_payload



def _existing_sidecars_match(motion_path, dense_path, frame_ids, anchors, camera_keys, threshold):
    """Only skip atomically completed sidecars aligned to this source episode."""
    if not motion_path.is_file() or not dense_path.is_file():
        return False
    try:
        motion = torch.load(motion_path, map_location="cpu", weights_only=True)
        dense = torch.load(dense_path, map_location="cpu", weights_only=True)
        return (
            motion["frame_ids"] == frame_ids
            and dense["frame_ids"] == frame_ids
            and motion["camera_keys"] == camera_keys
            and dense["camera_keys"] == camera_keys
            and motion["latent_num_frames"] == len(anchors)
            and motion["motion_indices"].shape[0] == len(anchors)
            and dense["dino_features"].shape[0] == len(anchors)
            and motion["provenance"]["anchor_indices"] == anchors
            and motion["provenance"]["input_mode"] == "rgb"
            and motion["provenance"]["threshold"] == threshold
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def build_features(task, device="cuda"):
    from n0_twam.preprocessing.dinov2 import FrozenDinoV2PatchEncoder

    root = paths(task)["dataset"]
    keys = list(task["robot"]["cameras"].values())
    model = task["runtime"].get("dino_model")
    if not model or not Path(model).is_dir():
        raise FileNotFoundError(
            "Set runtime.dino_model to local DINOv2 weights; no implicit downloads"
        )
    progress_path = root / "features.progress.json"
    progress = dict(task_fingerprint=fingerprint(task), builder_version=1)
    if progress_path.exists():
        if json.loads(progress_path.read_text()) != progress:
            raise ValueError(
                "Feature sidecars belong to a different task configuration; "
                "use a new task root or remove the stale sidecars and progress file"
            )
    else:
        if any((root / name).glob("chunk-*/*.pth") for name in ("rgb_motion", "ikv_index")):
            raise ValueError(
                "Existing feature sidecars lack a matching progress manifest; "
                "remove or migrate them before rebuilding"
            )
        progress_path.write_text(json.dumps(progress, indent=2))
    encoder = FrozenDinoV2PatchEncoder.from_pretrained(model, device=device)
    latent_files = sorted((root / "latents").glob(f"chunk-*/{keys[0]}/episode_*.pth"))
    if not latent_files:
        raise FileNotFoundError("Encode RGB latents first")
    shard_count = int(os.environ.get("IKV_FEATURE_SHARD_COUNT", "1"))
    shard_id = int(os.environ.get("IKV_FEATURE_SHARD_ID", "0"))
    if shard_count < 1 or not 0 <= shard_id < shard_count:
        raise ValueError("Invalid feature shard configuration")
    for episode_index, lp in enumerate(latent_files):
        if episode_index % shard_count != shard_id:
            continue
        chunk = lp.parent.parent.name
        reference = torch.load(lp, map_location="cpu", weights_only=True)
        ids = reference["frame_ids"]
        prov = reference["temporal_provenance"]
        if prov["anchor_semantics"] != "causal_chunk_end":
            raise ValueError("Unknown temporal semantics")
        anchors = prov["latent_anchor_indices"]
        for key in keys:
            other = torch.load(
                lp.parent.parent / key / lp.name, map_location="cpu", weights_only=True
            )
            for field in (
                "frame_ids",
                "latent_num_frames",
                "latent_height",
                "latent_width",
            ):
                if other[field] != reference[field]:
                    raise ValueError(f"Unaligned cameras: {field}")
        motion_dest = root / "rgb_motion" / chunk / lp.name
        dense_dest = root / "ikv_index" / chunk / lp.name
        threshold = float(task.get("features", {}).get("motion_threshold", 0.02))
        if _existing_sidecars_match(
            motion_dest, dense_dest, ids, anchors, keys, threshold
        ):
            print(f"Skipped existing features {lp.name}", flush=True)
            continue
        streams = {}
        for key in keys:
            stem = "_".join(lp.stem.split("_")[:2])
            streams[key] = selected_rgb(
                root / "videos" / chunk / key / (stem + ".mp4"), ids
            )
        motion, dense = build_payloads(
            streams,
            ids,
            anchors,
            encoder,
            threshold,
        )
        for dest, payload in ((motion_dest, motion), (dense_dest, dense)):
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
        print(f"Indexed {lp.name}", flush=True)
    if shard_count > 1:
        return
    (root / "features.json").write_text(
        json.dumps(
            dict(
                task_fingerprint=fingerprint(task),
                episodes=len(latent_files),
                dino_model=model,
                neoforce="unavailable; zero width",
                threshold=task.get("features", {}).get("motion_threshold", 0.02),
            ),
            indent=2,
        )
    )
