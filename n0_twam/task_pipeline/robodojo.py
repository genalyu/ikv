"""RoboDojo simulation adapter using the release EEF20 coordinate contract."""
from pathlib import Path
import sys
import io
import numpy as np
from .data import Episode, validate_pose

def image_frames(source):
    import h5py
    from PIL import Image
    with h5py.File(source["hdf5"], "r") as h:
        for value in h[source["key"]]:
            with Image.open(io.BytesIO(bytes(value))) as image:
                yield np.asarray(image.convert("RGB"))

def iter_robodojo(task):
    import h5py
    sys.path.insert(0, task["runtime"]["robodojo_reader_root"])
    from openwam.dataloader.robodojo import read_calibrated_eef20, validate_robodojo_episode
    from openwam.dataloader.robodojo_contract import resolve_robodojo_calibration
    if task.get("action_labels") != "next_observation":
        raise ValueError("RoboDojo EEF supervision must explicitly select next_observation")
    calibration = resolve_robodojo_calibration()
    paths = sorted(Path(task["source"]).glob("episode_*.hdf5"))
    if len(paths) != task["expected_episodes"]:
        raise ValueError("RoboDojo source episode count mismatch")
    for path in paths:
        meta = validate_robodojo_episode(path, variant="sim")
        if meta["instruction"] != task["prompt"]:
            raise ValueError(f"Per-episode instruction differs: {path}")
        with h5py.File(path, "r") as h:
            state = read_calibrated_eef20(h, calibration)
        validate_pose(state, str(path), arms=2)
        times = np.arange(len(state)-1, dtype=np.float64) / 25.0
        videos = {dst: {"hdf5": str(path), "key": src, "format": "robodojo"}
                  for src, dst in task["robot"]["cameras"].items()}
        yield Episode(str(path), times, state[1:], state[:-1],
            {"action_labels": "next_observation", "source_fps": 25,
             "coordinate_frame": "per_arm_robot_base", "calibration": calibration},
            videos, {key: times for key in videos})
