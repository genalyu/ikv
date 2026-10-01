"""Opt-in observational NeoSim IK failure capture; no solver fallback."""
import functools
import json
from pathlib import Path


def _values(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def install_ik_failure_capture(robot_class, output_path):
    """Wrap solve_ik once; preserve its return value, exceptions and call count."""
    original = robot_class.solve_ik
    if getattr(original, "_ik_failure_capture", False):
        raise RuntimeError("IK failure capture already installed")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    @functools.wraps(original)
    def wrapped(self, target_pose, *args, **kwargs):
        # Snapshot before the solve so failed inputs can be reproduced.
        current_joints = _values(self.get_qpos())
        record = {
            "robot": str(self.name),
            "tick": int(self.task.take_action_cnt),
            "target_p": _values(target_pose.p),
            "target_q_wxyz": _values(target_pose.q),
            "current_joints": current_joints,
            "root_p": _values(self.root_pose.p),
            "root_q_wxyz": _values(self.root_pose.q),
        }
        result = original(self, target_pose, *args, **kwargs)
        if result["status"] != "Success":
            record["status"] = str(result["status"])
            # Fail loudly on recording errors rather than claim a complete audit.
            with destination.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    wrapped._ik_failure_capture = True
    robot_class.solve_ik = wrapped
    return original
