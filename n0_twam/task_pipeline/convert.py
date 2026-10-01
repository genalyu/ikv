"""Episode-preserving conversion to official LeRobot v2.1 storage."""

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import paths, fingerprint, conversion_identity
from .data import episodes, video_reader, aligned_timeline, quality_exclusion_reason


def stats(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return {
        k: np.asarray(v).tolist()
        for k, v in dict(
            min=x.min(0),
            max=x.max(0),
            mean=x.mean(0),
            std=x.std(0),
            count=np.array([len(x)]),
        ).items()
    }


def image_stats_rgb(frames):
    """Exact RGB statistics without allocating full-resolution float64 images."""
    import cv2

    count = 0
    frames_seen = 0
    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    minimum = np.full(3, 255, dtype=np.uint8)
    maximum = np.zeros(3, dtype=np.uint8)
    for rgb in frames:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB video frames")
        pixels = rgb.reshape(-1, 3)
        n = len(pixels)
        mean, std = cv2.meanStdDev(rgb)
        mean = mean[:, 0]
        std = std[:, 0]
        sums += mean * n
        sums_sq += (std * std + mean * mean) * n
        minimum = np.minimum(minimum, pixels.min(axis=0))
        maximum = np.maximum(maximum, pixels.max(axis=0))
        count += n
        frames_seen += 1
    if not count:
        raise ValueError("Empty converted video")
    mean = sums / count / 255
    variance = np.maximum(sums_sq / count / (255 * 255) - mean * mean, 0)
    return {
        k: np.asarray(v).reshape(3, 1, 1).tolist()
        for k, v in {
            "min": minimum.astype(np.float64) / 255,
            "max": maximum.astype(np.float64) / 255,
            "mean": mean,
            "std": np.sqrt(variance),
        }.items()
    } | {"count": [frames_seen]}


def image_stats_video(path):
    import av

    with av.open(str(path)) as video:
        return image_stats_rgb(
            frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)
        )


def encode_aligned_video(source, destination, ids, fps):
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".partial.mp4")

    def frames():
        if isinstance(source, dict) and source.get("hdf5"):
            from .neosim import image_frames

            yield from image_frames(source)
        else:
            if isinstance(source, bytes):
                with av.open(io.BytesIO(source)) as container:
                    for frame in container.decode(video=0):
                        yield frame.to_ndarray(format="rgb24")
            else:
                with video_reader(source) as f:
                    with av.open(f) as container:
                        for frame in container.decode(video=0):
                            yield frame.to_ndarray(format="rgb24")

    with av.open(str(temp), "w") as out:
        stream = None
        target = 0
        for index, rgb in enumerate(frames()):
            if stream is None:
                stream = out.add_stream("libx264", rate=Fraction(str(fps)))
                stream.width, stream.height = rgb.shape[1], rgb.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast"}
            while target < len(ids) and ids[target] == index:
                f = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                f.pts = target
                f.time_base = Fraction(1, 1) / Fraction(str(fps))
                for packet in stream.encode(f):
                    out.mux(packet)
                target += 1
            if target == len(ids):
                break
        if target != len(ids):
            raise ValueError(
                f"Video shorter than timestamp table: wrote {target}/{len(ids)}"
            )
        for packet in stream.encode():
            out.mux(packet)
    temp.replace(destination)
    return [stream.height, stream.width, 3]


def convert(task):
    if (
        task["format"] == "neosim_hdf5"
        and task.get("action_labels") != "next_observation"
    ):
        raise ValueError(
            "NeoSim has no commanded actions; explicitly choose action_labels=next_observation or supply command data"
        )
    p = paths(task)
    root = p["dataset"]
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / "conversion.json"
    sources = task["source"] if isinstance(task["source"], list) else [task["source"]]
    for src in sources:
        if Path(src).is_dir() and root.resolve().is_relative_to(Path(src).resolve()):
            raise ValueError("Conversion output must be outside the source tree")
    identity = conversion_identity(task)
    spec = fingerprint(identity)
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if old["fingerprint"] != spec:
            raise ValueError(
                "Output belongs to a different conversion; choose another work root"
            )
        print(f"Conversion already complete: {root}")
        return
    # A partial conversion is restartable only for the identical inputs/configuration.
    pending = root / "conversion_pending.json"
    if pending.exists() and json.loads(pending.read_text())["fingerprint"] != spec:
        raise ValueError("Partial conversion configuration differs")
    pending.write_text(json.dumps({"fingerprint": spec}))
    fps = 30
    robot = task["robot"]
    all_eps = []
    all_stats = []
    provenance = []
    shapes = {}
    total = 0
    excluded = []
    source_count = 0
    width = robot.get("arms", 1) * 10
    for ep in episodes(task):
        source_count += 1
        if len(ep.timestamps) < 2:
            raise ValueError("Episode too short")
        if not np.isfinite(ep.action).all():
            raise ValueError("Missing finite supervised action labels")
        quality_reason = quality_exclusion_reason(ep, task)
        if quality_reason:
            excluded.append(dict(source_id=ep.source_id, reason=quality_reason))
            print(f"EXCLUDED {ep.source_id}: {quality_reason}", flush=True)
            continue
        try:
            ts, ai, camera_indices, time_audit = aligned_timeline(ep, robot)
        except ValueError as error:
            if task.get("on_invalid_alignment", "error") != "exclude":
                raise
            excluded.append(dict(source_id=ep.source_id, reason=str(error)))
            print(f"EXCLUDED {ep.source_id}: {error}", flush=True)
            continue
        index = len(all_eps)
        a = np.zeros((len(ts), 20), np.float32)
        s = a.copy()
        a[:, :width] = ep.action[ai]
        s[:, :width] = ep.state[ai]
        chunk = f"chunk-{index // 1000:03d}"
        stem = f"episode_{index:06d}"
        marker = root / "conversion_episodes" / (stem + ".json")
        marker_data = json.loads(marker.read_text()) if marker.exists() else None
        encoded = False
        alignment = {}
        if task["format"] == "collector_v06":
            # The tar handle is shared; read sources serially, then encode cameras in parallel.
            jobs = {}
            with ThreadPoolExecutor(max_workers=min(4, len(ep.videos))) as pool:
                for key, source in ep.videos.items():
                    ids = camera_indices[key]
                    alignment[key] = time_audit["cameras"][key]
                    video = root / "videos" / chunk / key / (stem + ".mp4")
                    if marker_data is not None and video.exists():
                        shapes[key] = marker_data["shapes"][key]
                    else:
                        encoded = True
                        with video_reader(source) as source_file:
                            payload = (
                                Path(source_file).read_bytes()
                                if isinstance(source_file, str)
                                else source_file.read()
                            )
                        jobs[key] = pool.submit(
                            encode_aligned_video, payload, video, ids, fps
                        )
                for key, job in jobs.items():
                    shapes[key] = job.result()
        else:
            for key, source in ep.videos.items():
                ids = camera_indices[key]
                alignment[key] = time_audit["cameras"][key]
                video = root / "videos" / chunk / key / (stem + ".mp4")
                if marker_data is not None and video.exists():
                    shapes[key] = marker_data["shapes"][key]
                else:
                    encoded = True
                    shapes[key] = encode_aligned_video(source, video, ids, fps)
        data = dict(
            action=list(a),
            **{"observation.state": list(s)},
            timestamp=np.arange(len(ts), dtype=np.float32) / fps,
            frame_index=np.arange(len(ts), dtype=np.int64),
            episode_index=np.full(len(ts), index, dtype=np.int64),
            index=np.arange(total, total + len(ts), dtype=np.int64),
            task_index=np.zeros(len(ts), dtype=np.int64),
        )
        df = pd.DataFrame(data)
        target = root / "data" / chunk / (stem + ".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, index=False)
        all_eps.append(
            dict(
                episode_index=index,
                tasks=[task["prompt"]],
                length=len(ts),
                action_config=[dict(start_frame=0, end_frame=len(ts))],
            )
        )
        st = {k: stats(np.stack(df[k])) for k in df.columns}
        # Cache exact converted-video statistics at the episode boundary so a
        # preempted conversion never decodes completed episodes twice.
        if marker_data is not None and not encoded and "video_stats" in marker_data:
            st.update(marker_data["video_stats"])
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(ep.videos))) as pool:
                stat_jobs = {
                    key: pool.submit(
                        image_stats_video,
                        root / "videos" / chunk / key / (stem + ".mp4"),
                    )
                    for key in ep.videos
                }
                for key, job in stat_jobs.items():
                    st[key] = job.result()
        all_stats.append(dict(episode_index=index, stats=st))
        provenance.append(
            dict(
                episode_index=index,
                source_id=ep.source_id,
                source_metadata=ep.metadata,
                alignment=alignment,
                time_alignment=time_audit,
            )
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker_tmp = marker.with_suffix(".tmp")
        marker_tmp.write_text(json.dumps({
            "shapes": shapes,
            "video_stats": {key: st[key] for key in ep.videos},
        }))
        marker_tmp.replace(marker)
        total += len(ts)
        print(f"Converted {index}: {ep.source_id} ({len(ts)} frames)", flush=True)
    if not all_eps:
        raise ValueError("Empty input")
    features = {}
    for key in ("action", "observation.state"):
        features[key] = {
            "dtype": "float32",
            "shape": [20],
            "names": [f"channel_{i}" for i in range(20)],
        }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {
            "dtype": "float32" if key == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }
    for key, shape in shapes.items():
        features[key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    info = dict(
        codebase_version="v2.1",
        robot_type=robot["name"],
        fps=fps,
        total_episodes=len(all_eps),
        total_frames=total,
        source_total_episodes=source_count,
        excluded_episodes=len(excluded),
        total_tasks=1,
        total_videos=len(all_eps) * len(shapes),
        chunks_size=1000,
        total_chunks=(len(all_eps) + 999) // 1000,
        splits={"train": f"0:{len(all_eps)}"},
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        features=features,
    )
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    for name, items in (
        ("episodes", all_eps),
        ("episodes_stats", all_stats),
        ("tasks", [{"task_index": 0, "task": task["prompt"]}]),
        ("source_provenance", provenance),
    ):
        (meta / (name + ".jsonl")).write_text(
            "".join(
                json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in items
            )
        )
    (meta / "excluded_episodes.json").write_text(json.dumps(excluded, indent=2))
    stamp.write_text(
        json.dumps(
            dict(
                fingerprint=spec,
                episodes=len(all_eps),
                frames=total,
                source=task["source"],
                identity=identity,
            ),
            indent=2,
        )
    )
    pending.unlink()
