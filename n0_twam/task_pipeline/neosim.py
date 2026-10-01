"""Verified NeoSim HDF5 observations; action-label policy is always explicit."""

import json
from pathlib import Path
import numpy as np
from .data import Episode, validate_pose


def quaternion_to_rot6d(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or (norm < 1e-8).any():
        raise ValueError("Invalid wxyz quaternion")
    w, x, y, z = (q / norm).T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
            2 * (x * y - w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z + w * x),
        ],
        axis=1,
    )


def image_frames(source):
    import cv2
    import h5py

    with h5py.File(source["hdf5"], "r") as h:
        for value in h[source["key"]]:
            # NeoSim DataManager.img_to_stream passes native RGB directly to
            # cv2.imencode. Its stream_to_img uses imdecode without channel swapping.
            # Preserve that verified round-trip; PIL would invert the stored convention.
            frame = cv2.imdecode(
                np.frombuffer(bytes(value), np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError("Invalid NeoSim JPEG")
            yield frame


def iter_neosim(task):
    import h5py

    root = Path(task["source"])
    robot = task["robot"]
    if robot.get("quaternion_order") != "wxyz":
        raise ValueError("NeoSim Pose.totensor convention is explicitly wxyz")
    if robot.get("arms", 1) != 1:
        raise ValueError(
            "Verified NeoSim adapter supports single-arm embodiment/ee; use a mapped LeRobot dataset for dual arm"
        )
    policy = task.get("action_labels")
    for path in sorted((root / "hdf5").glob("*.hdf5"), key=lambda p: int(p.stem)):
        meta_path = root / "metadata" / (path.stem + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        with h5py.File(path, "r") as h:
            steps = np.asarray(h["step"], dtype=float)
            if not np.isclose(
                np.median(np.diff(steps)) / robot["simulation_hz"], 1 / robot["fps"]
            ):
                raise ValueError(
                    "Recorded simulation frequency differs from explicit robot fps"
                )
            ee = np.asarray(h["embodiment/ee"], dtype=float)
            joints = np.asarray(h["embodiment/joint"], dtype=float)
            if (np.diff(steps) <= 0).any():
                raise ValueError("Non-monotonic simulation steps")
            grip = joints[:, robot["gripper_joint_indices"]].sum(axis=1) / float(
                robot["gripper_stroke_m"]
            )
            if grip.min() < -0.002 or grip.max() > 1.002:
                raise ValueError("Gripper joints exceed configured physical stroke")
            # Allow only numerical simulator limit overshoot, recorded in provenance.
            meta["gripper_clamped_samples"] = int(((grip < 0) | (grip > 1)).sum())
            state = np.column_stack(
                [ee[:, :3], quaternion_to_rot6d(ee[:, 3:7]), np.clip(grip, 0, 1)]
            )
            validate_pose(state, "NeoSim state")
            times = (steps - steps[0]) / float(robot["simulation_hz"])
            action = np.full_like(state, np.nan)
            if policy == "next_observation":
                action = state[1:].copy()
                state = state[:-1]
                times = times[:-1]
            elif policy not in (None, "unavailable"):
                raise ValueError("Unknown NeoSim action_labels policy")
            meta.update(
                action_labels=policy or "unavailable",
                position_unit="m",
                source_first_step=int(steps[0]),
                source_last_step=int(steps[-1]),
                simulation_hz=robot["simulation_hz"],
                recorded_frames=len(steps),
                source_duration_s=float(
                    (steps[-1] - steps[0] + np.median(np.diff(steps)))
                    / robot["simulation_hz"]
                ),
            )
            cameras = {**robot["cameras"], **robot["tactile"]}
            videos = {}
            ct = {}
            for raw, dest in cameras.items():
                if len(h[raw]) != len(steps):
                    raise ValueError(f"Camera length mismatch: {raw}")
                videos[dest] = {"hdf5": str(path), "key": raw}
                ct[dest] = (steps - steps[0]) / robot["simulation_hz"]
        yield Episode(path.stem, times, action, state, meta, videos, ct)
