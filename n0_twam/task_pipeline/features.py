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
        self.relevance = []

    def __call__(self, rgb):
        result = self.encoder(rgb)
        self.outputs.append(result.tokens.detach().cpu())
        if hasattr(result, "task_relevance"):
            self.relevance.append(result.task_relevance.detach().cpu())
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
    relevance = []
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
        recorder.relevance = []
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
        if recorder.relevance:
            relevance.append(torch.cat([
                pool_dino_to_grid(x[..., None], (gh, gw)) for x in recorder.relevance
            ], dim=2).reshape(1, gh * gw * len(keys)))
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
    if relevance:
        dense_payload["task_relevance"] = torch.cat(relevance)
        dense_payload["semantic_provenance"] = encoder.provenance()
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


def estimate_depth_contact(depth, threshold_mm=0.1, minimum_fraction=0.005):
    """Estimate contact from indentation relative to each pixel's temporal 95th percentile."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3 or not np.isfinite(depth).all():
        raise ValueError("Expected finite tactile depth [time,height,width]")
    reference = np.quantile(depth, 0.95, axis=0)
    fraction = ((reference[None] - depth) > threshold_mm).mean(axis=(1, 2))
    return fraction >= minimum_fraction


def build_tactile_payload(task, root, chunk, name, ids, anchors, encoder):
    """Native tactile RGB patches plus measured-contact spans in world-time units."""
    import h5py
    from .neosim import iter_neosim
    from .data import aligned_timeline
    from n0_twam.models.rgb_motion import pool_dino_to_grid

    marker = json.loads((root/"conversion_episodes"/("_".join(Path(name).stem.split("_")[:2])+".json")).read_text())
    provenance = root / "meta/source_provenance.jsonl"
    sid = marker.get("source_id")
    if provenance.is_file():
        episode_index = int(Path(name).stem.split("_")[1])
        matches = [row["source_id"] for row in map(json.loads, provenance.read_text().splitlines())
                   if row["episode_index"] == episode_index]
        if len(matches) != 1:
            raise ValueError("Expected exactly one audited source mapping for tactile episode")
        if sid is not None and str(sid) != str(matches[0]):
            raise ValueError("Conversion marker and audited tactile source mapping disagree")
        sid = str(matches[0])
    if sid is None:
        raise ValueError("Missing audited tactile source mapping")
    source = Path(task["source"])
    # Reuse the explicit single-episode source view made by the converter worker.
    view = root/"tactile_index_sources"/sid
    for kind, ext in (("hdf5",".hdf5"),("metadata",".json")):
        (view/kind).mkdir(parents=True,exist_ok=True)
        dest = view/kind/(sid+ext)
        if not dest.exists():
            dest.symlink_to(source/kind/(sid+ext))
    episode = next(iter_neosim(dict(task,source=str(view))))
    _, _, aligned, _ = aligned_timeline(episode,task["robot"])
    keys = list(task["robot"]["tactile"].values())
    anchor_ids = [int(ids[i]) for i in anchors]
    features, contacts, contact_sources = [], [], []
    with h5py.File(source/"hdf5"/(sid+".hdf5"),"r") as raw:
        for raw_key,key in task["robot"]["tactile"].items():
            sensor = raw_key.rsplit("/",1)[0]
            if sensor+"/contact_force" in raw:
                force = np.asarray(raw[sensor+"/contact_force"])
                if not np.isfinite(force).all():
                    raise ValueError("nonfinite measured contact force")
                contact = np.any(force.reshape(len(force),-1)!=0,axis=1)[aligned[key]]
                contact_sources.append(dict(sensor=key, method="measured_contact_force"))
            else:
                contact = estimate_depth_contact(raw[sensor+"/depth"])[aligned[key]]
                contact_sources.append(dict(sensor=key, method="estimated_depth_indentation",
                    reference="per_pixel_temporal_95th_percentile", threshold_mm=0.1,
                    minimum_area_fraction=0.005, limitation="relative deformation estimate; sustained contact may be underestimated"))
            contacts.append(contact)
            frames = selected_rgb(root/"videos"/chunk/key/("_".join(Path(name).stem.split("_")[:2])+".mp4"),anchor_ids)
            values = []
            for rgb in frames:
                tokens = encoder(torch.from_numpy(rgb)[None]).tokens
                values.append(pool_dino_to_grid(tokens,(4,4)).reshape(16,-1).detach().cpu())
            if len(values)!=len(anchors):
                raise ValueError("tactile semantics do not cover WAN anchors")
            features.append(torch.stack(values))
    contact = np.stack(contacts,axis=1)
    def duration(mask):
        count=np.zeros(mask.shape[1:],dtype=np.float32)
        result=[]
        for row in mask:
            count=np.where(row,count+1,0);result.append(count.copy())
        # The converted action clock is 30Hz, each WAN world step spans 12 frames.
        return np.asarray(result,dtype=np.float32)/12.
    return dict(contact_provenance=contact_sources,tactile_camera_keys=keys,tactile_grid_shape=[4,4],
        tactile_semantic_backend="dinov3-native-rgb",
        tactile_dino_features=torch.stack(features,dim=1),
        frame_contact_duration=torch.from_numpy(duration(contact.any(1))[anchor_ids]),
        tactile_contact_duration=torch.from_numpy(duration(contact)[anchor_ids]))


def build_features(task, device="cuda"):
    from n0_twam.preprocessing.dinov3 import FrozenDinoV3PatchEncoder

    root = paths(task)["dataset"]
    keys = list(task["robot"]["cameras"].values())
    model = task["runtime"].get("dino_model")
    semantic_config = dict(task["runtime"].get("semantic_encoder", {}) or {})
    semantic_enabled = semantic_config.get("backend", "dinov3") != "dinov3"
    tactile_semantics = task.get("features",{}).get("tactile_semantic_backend") == "dinov3"
    if tactile_semantics and semantic_enabled:
        raise ValueError("tactile DINOv3 requires the raw DINOv3 visual backend")
    if not semantic_enabled and (not model or not Path(model).is_dir()):
        raise FileNotFoundError(
            "Set runtime.dino_model to local DINOv3 weights; no implicit downloads"
        )
    progress_path = root / "features.progress.json"
    progress = dict(task_fingerprint=fingerprint(task), builder_version=4 if tactile_semantics else (2 if semantic_enabled else 1))
    if progress_path.exists():
        if json.loads(progress_path.read_text()) != progress:
            raise ValueError(
                "Feature sidecars belong to a different task configuration; "
                "use a new task root or remove the stale sidecars and progress file"
            )
    else:
        if any(any((root / name).glob("chunk-*/*.pth")) for name in ("rgb_motion", "ikv_index")):
            raise ValueError(
                "Existing feature sidecars lack a matching progress manifest; "
                "remove or migrate them before rebuilding"
            )
        progress_path.write_text(json.dumps(progress, indent=2))
    if semantic_enabled:
        from n0_twam.preprocessing.semantic_patch import load_patch_encoder
        encoder = load_patch_encoder(semantic_config, device=device, prompt=task["prompt"])
    else:
        encoder = FrozenDinoV3PatchEncoder.from_pretrained(
            model, device=device, torch_dtype=torch.bfloat16, image_size=tuple(task["runtime"].get("dino_image_size", [224,224])))
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
        semantic_matches = True
        if semantic_enabled and dense_dest.exists():
            old = torch.load(dense_dest, map_location="cpu", weights_only=True)
            semantic_matches = (old.get("semantic_provenance") == encoder.provenance()
                                and "task_relevance" in old)
        if not semantic_enabled and dense_dest.exists():
            old = torch.load(dense_dest,map_location="cpu",weights_only=True)
            semantic_matches = old.get("dino_provenance") == encoder.provenance()
        if tactile_semantics and dense_dest.exists():
            old = torch.load(dense_dest,map_location="cpu",weights_only=True)
            semantic_matches = semantic_matches and old.get("tactile_semantic_backend") == "dinov3-native-rgb"
        if semantic_matches and _existing_sidecars_match(
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
        if not semantic_enabled:
            dense["dino_provenance"] = encoder.provenance()
        if tactile_semantics:
            dense.update(build_tactile_payload(task,root,chunk,lp.name,ids,anchors,encoder))
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
                dino_provenance=encoder.provenance() if not semantic_enabled else None,
                semantic_provenance=encoder.provenance() if semantic_enabled else None,
                neoforce="unavailable; zero width",
                tactile_semantic_backend="dinov3-native-rgb" if tactile_semantics else None,
                threshold=task.get("features", {}).get("motion_threshold", 0.02),
            ),
            indent=2,
        )
    )
