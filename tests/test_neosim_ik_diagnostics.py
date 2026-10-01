import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SPEC = importlib.util.spec_from_file_location(
    "neosim_ik_diagnostics", Path(__file__).parents[1] / "example_client/neosim_ik_diagnostics.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

def robot_class(result):
    class Robot:
        name = "test"
        task = SimpleNamespace(take_action_cnt=645)
        root_pose = SimpleNamespace(p=[0, 0, 0], q=[1, 0, 0, 0])
        calls = 0
        def get_qpos(self):
            return [[0.1, 0.2]]
        def solve_ik(self, pose):
            self.calls += 1
            if isinstance(result, Exception):
                raise result
            return result
    return Robot

@pytest.mark.parametrize("status", ["Success", "Fail"])
def test_result_and_call_count_unchanged(tmp_path, status):
    result = {"status": status, "position": None}
    cls = robot_class(result)
    output = tmp_path / "failures.jsonl"
    MODULE.install_ik_failure_capture(cls, output)
    robot = cls()
    pose = SimpleNamespace(p=[1, 2, 3], q=[1, 0, 0, 0])
    assert robot.solve_ik(pose) is result
    assert robot.calls == 1
    assert pose.p == [1, 2, 3]
    if status == "Fail":
        row = json.loads(output.read_text())
        assert row["target_p"] == pose.p
        assert row["current_joints"] == [[0.1, 0.2]]
        assert row["tick"] == 645
    else:
        assert not output.exists()
    with pytest.raises(RuntimeError, match="already installed"):
        MODULE.install_ik_failure_capture(cls, output)

def test_solver_exception_propagates(tmp_path):
    cls = robot_class(ValueError("solver failed"))
    MODULE.install_ik_failure_capture(cls, tmp_path / "failures.jsonl")
    robot = cls()
    with pytest.raises(ValueError, match="solver failed"):
        robot.solve_ik(SimpleNamespace(p=[0, 0, 0], q=[1, 0, 0, 0]))
    assert robot.calls == 1
