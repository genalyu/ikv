"""RoboDojo simulation adapter using the release EEF20 coordinate contract."""
from pathlib import Path
import sys
import numpy as np
from .data import Episode, validate_pose

def image_frames(source):
    """Use the simulator's marker-aware decoder for legacy and standard JPEGs."""
    import h5py
    decoder_root = source.get("xpolicylab_root")
    if not decoder_root:
        raise ValueError("RoboDojo conversion requires runtime.xpolicylab_root; refusing ambiguous JPEG colors")
    package = Path(decoder_root).resolve()
    if not (package / "utils/process_data.py").is_file():
        raise FileNotFoundError(package / "utils/process_data.py")
    sys.path.insert(0, str(package.parent))
    from XPolicyLab.utils.process_data import decode_image_bit
    import XPolicyLab.utils.process_data as codec
    if Path(codec.__file__).resolve() != package / "utils/process_data.py":
        raise RuntimeError("A different XPolicyLab image decoder is already imported")
    with h5py.File(source["hdf5"], "r") as h:
        for value in h[source["key"]]:
            rgb = np.asarray(decode_image_bit(value))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("Canonical RoboDojo decoder must return HWC uint8 RGB")
            yield rgb

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
        videos = {dst: {"hdf5": str(path), "key": src, "format": "robodojo",
                        "xpolicylab_root": task["runtime"].get("xpolicylab_root")}
                  for src, dst in task["robot"]["cameras"].items()}
        yield Episode(str(path), times, state[1:], state[:-1],
            {"action_labels": "next_observation", "source_fps": 25,
             "coordinate_frame": "per_arm_robot_base", "calibration": calibration},
            videos, {key: times for key in videos})
