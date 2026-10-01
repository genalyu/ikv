"""Explicit raw-data adapters. No inferred action semantics or RGB/BGR swaps."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd


@dataclass
class Episode:
    source_id: str
    timestamps: np.ndarray
    action: np.ndarray
    state: np.ndarray
    metadata: dict
    videos: dict = field(default_factory=dict)
    camera_times: dict = field(default_factory=dict)


class Collector:
    def __init__(self, path):
        self.path = Path(path)
        self.tar = tarfile.open(path) if self.path.is_file() else None
        if self.tar:
            self.members = {m.name: m for m in self.tar.getmembers() if m.isfile()}
            self.episodes = sorted(
                n[:-14] for n in self.members if n.endswith("/metadata.json")
            )
        else:
            self.episodes = sorted(
                str(p.parent.relative_to(self.path))
                for p in self.path.rglob("metadata.json")
            )
        if not self.episodes:
            raise ValueError("No collector episodes found")

    @contextmanager
    def open(self, name):
        f = (
            self.tar.extractfile(self.members[name])
            if self.tar
            else (self.path / name).open("rb")
        )
        try:
            yield f
        finally:
            f.close()

    def read(self, name):
        with self.open(name) as f:
            return f.read()

    def close(self):
        if self.tar:
            self.tar.close()


def validate_pose(x, label, arms=1):
    if x.ndim != 2 or x.shape[1] != arms * 10 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"{label}: expected finite [T,{arms * 10}] with T>=2")
    for offset in range(0, arms * 10, 10):
        r1, r2 = x[:, offset + 3 : offset + 6], x[:, offset + 6 : offset + 9]
        error = max(
            np.abs(np.linalg.norm(r1, axis=1) - 1).max(),
            np.abs(np.linalg.norm(r2, axis=1) - 1).max(),
            np.abs((r1 * r2).sum(1)).max(),
        )
        if error > 0.05:
            raise ValueError(f"{label}: malformed column rot6d (error={error:.4g})")
        if x[:, offset + 9].min() < -1e-4 or x[:, offset + 9].max() > 1.0001:
            raise ValueError(f"{label}: gripper outside explicit [0,1] open ratio")


def causal_indices(source_times, target_times, max_age=0.1):
    source_times, target_times = np.asarray(source_times), np.asarray(target_times)
    if (
        len(source_times) == 0
        or not np.isfinite(source_times).all()
        or (np.diff(source_times) < 0).any()
    ):
        raise ValueError(
            "Source timestamps must be finite and nondecreasing (equal timestamps select last)"
        )
    ids = np.searchsorted(source_times, target_times + 1e-9, side="right") - 1
    if (ids < 0).any():
        raise ValueError(
            "No past observation available at start; refusing future-frame fill"
        )
    age = target_times - source_times[ids]
    if age.max() > max_age + 1e-9:
        raise ValueError(f"Alignment age {age.max():.6f}s exceeds {max_age}s")
    return ids, age


def normalize_grip(x, representation, stroke):
    x = np.asarray(x, dtype=np.float64).copy()
    for i in range(9, x.shape[1], 10):
        if representation == "width_m":
            if stroke <= 0:
                raise ValueError("Positive gripper stroke required")
            x[:, i] /= stroke
        elif representation != "open_ratio":
            raise ValueError(
                "Explicit gripper representation must be width_m or open_ratio"
            )
    return x


def _iter_collector_source(task, path, source_id_prefix=""):
    robot = task["robot"]
    if robot.get("arms", 1) != 1:
        raise ValueError(
            "collector_v06 adapter currently supports the verified single-arm schema"
        )
    source = Collector(path)
    cameras = {**robot["cameras"], **robot["tactile"]}
    try:
        for ep in source.episodes:
            meta = json.loads(source.read(ep + "/metadata.json"))
            if meta.get("version") != "v0.6":
                raise ValueError("Collector adapter only supports verified v0.6 schema")
            if abs(float(meta["fps_config"]) - float(robot["fps"])) > 1e-6:
                raise ValueError("Robot fps differs from collector metadata")
            act = pd.read_csv(
                io.BytesIO(source.read(ep + "/actions.eef_pose/data.csv"))
            )
            obs = pd.read_csv(
                io.BytesIO(source.read(ep + "/observation.state.eef_pose/data.csv"))
            )
            acols = (
                ["tcp.x", "tcp.y", "tcp.z"]
                + [f"tcp.r{i}" for i in range(1, 7)]
                + ["gripper.pos"]
            )
            scols = ["x", "y", "z"] + [f"r{i}" for i in range(1, 7)] + ["gripper"]
            ts = act["timestamp_ms"].to_numpy(float) / 1000
            state_ids, age = causal_indices(
                obs["timestamp_ms"].to_numpy(float) / 1000,
                ts,
                robot.get("max_alignment_age_s", 0.1),
            )
            a = normalize_grip(
                act[acols].to_numpy(float),
                robot["action_gripper"],
                robot["gripper_stroke_m"],
            )
            s = normalize_grip(
                obs[scols].to_numpy(float)[state_ids],
                robot["state_gripper"],
                robot["gripper_stroke_m"],
            )
            validate_pose(a, "action")
            validate_pose(s, "state")
            times, videos = {}, {}
            for raw, dest in cameras.items():
                df = pd.read_csv(
                    io.BytesIO(source.read(ep + "/" + raw + "/timestamps.csv"))
                )
                times[dest] = df["timestamp_ms"].to_numpy(float) / 1000
                videos[dest] = (source, ep + "/" + raw + "/video.mp4")
                meta.setdefault("camera_timestamp_audit", {})[dest] = {
                    "frames": len(df),
                    "non_new_frames": int((df["is_new"] == 0).sum()),
                    "non_increasing": int((np.diff(times[dest]) <= 0).sum()),
                    "max_gap_s": float(np.diff(times[dest]).max()),
                }
            meta["state_alignment_max_s"] = float(age.max())
            meta["duplicate_action_timestamps"] = int((np.diff(ts) == 0).sum())
            meta["duplicate_timestamp_policy"] = (
                "causal last at equal time; zero-dt excluded from speed"
            )
            yield Episode(source_id_prefix + ep, ts, a, s, meta, videos, times)
    finally:
        source.close()


def quality_exclusion_reason(ep, task):
    allowed = task.get("allowed_quality_labels")
    if allowed is None:
        return None
    labels = ep.metadata.get("quality", {}).get("labels", [])
    if sorted(labels) != sorted(allowed):
        return f"Quality labels {labels!r} do not match allowed {allowed!r}"
    return None


def iter_collector(task):
    sources = task["source"] if isinstance(task["source"], list) else [task["source"]]
    for i, path in enumerate(sources):
        prefix = f"{i}:{Path(path).name}:" if len(sources) > 1 else ""
        yield from _iter_collector_source(task, path, prefix)


def iter_lerobot(task):
    root, robot = Path(task["source"]), task["robot"]
    info = json.loads((root / "meta/info.json").read_text())
    if not str(info["codebase_version"]).startswith("v2"):
        raise ValueError("LeRobot adapter currently requires v2.x episode files")
    fps = float(info["fps"])
    for p in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        df = pd.read_parquet(p)
        width = 10 * robot.get("arms", 1)
        a = np.stack(df["action"])[:, :width]
        s = np.stack(df["observation.state"])[:, :width]
        a = normalize_grip(a, robot["action_gripper"], robot.get("gripper_stroke_m", 1))
        s = normalize_grip(s, robot["state_gripper"], robot.get("gripper_stroke_m", 1))
        validate_pose(a, "action", width // 10)
        validate_pose(s, "state", width // 10)
        times = df["timestamp"].to_numpy(float)
        videos, ct = {}, {}
        for raw, dest in {**robot["cameras"], **robot["tactile"]}.items():
            videos[dest] = root / "videos" / p.parent.name / raw / (p.stem + ".mp4")
            ct[dest] = np.arange(len(df)) / fps
        yield Episode(
            p.stem,
            times,
            a,
            s,
            {"source_info": info, "robot": robot["name"]},
            videos,
            ct,
        )


def episodes(task):
    if task["format"] == "collector_v06":
        yield from iter_collector(task)
    elif task["format"] == "lerobot_v21":
        yield from iter_lerobot(task)
    else:
        from .neosim import iter_neosim

        yield from iter_neosim(task)


@contextmanager
def video_reader(value):
    if isinstance(value, tuple) and isinstance(value[0], Collector):
        with value[0].open(value[1]) as f:
            # tar member streams are seekable, but libav may seek beyond the member.
            yield io.BytesIO(f.read())
    else:
        yield str(value)


def aligned_timeline(ep, robot):
    tolerance = float(robot.get("max_alignment_age_s", 0.1))
    start = max([ep.timestamps[0]] + [t[0] for t in ep.camera_times.values()])
    offset = float(start - ep.timestamps[0])
    if offset > tolerance:
        raise ValueError(f"Initial sensor offset {offset:.6f}s exceeds {tolerance}s")
    end = ep.timestamps[-1] + 1 / float(robot["fps"])
    times = start + np.arange(int(np.ceil((end - start) * 30 - 1e-7))) / 30
    action_ids, action_age = causal_indices(ep.timestamps, times, tolerance)
    camera_indices, audit = {}, {}
    for key, timestamps in ep.camera_times.items():
        ids, age = causal_indices(timestamps, times, tolerance)
        camera_indices[key] = ids
        audit[key] = dict(
            max_age_s=float(age.max()),
            mean_age_s=float(age.mean()),
            held_frames=int((np.diff(ids) == 0).sum()),
        )
    return (
        times,
        action_ids,
        camera_indices,
        dict(
            common_start_offset_s=offset,
            action_alignment_max_s=float(action_age.max()),
            cameras=audit,
        ),
    )
