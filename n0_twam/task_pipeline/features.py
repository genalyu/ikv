"""Bounded-memory RGB-native sidecars using the online motion implementation."""

import json
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


def build_features(task, device="cuda"):
    from n0_twam.preprocessing.dinov2 import FrozenDinoV2PatchEncoder

    root = paths(task)["dataset"]
    keys = list(task["robot"]["cameras"].values())
    model = task["runtime"].get("dino_model")
    if not model or not Path(model).is_dir():
        raise FileNotFoundError(
            "Set runtime.dino_model to local DINOv2 weights; no implicit downloads"
        )
    encoder = FrozenDinoV2PatchEncoder.from_pretrained(model, device=device)
    latent_files = sorted((root / "latents").glob(f"chunk-*/{keys[0]}/episode_*.pth"))
    if not latent_files:
        raise FileNotFoundError("Encode RGB latents first")
    for lp in latent_files:
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
            float(task.get("features", {}).get("motion_threshold", 0.02)),
        )
        for name, payload in (("rgb_motion", motion), ("ikv_index", dense)):
            dest = root / name / chunk / lp.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
        print(f"Indexed {lp.name}", flush=True)
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
