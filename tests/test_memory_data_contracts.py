import importlib.util
from pathlib import Path
import numpy as np
import pytest
from n0_twam.task_pipeline.normalization import fit_eef20_bounds, ROTATION_IDS


def test_native_rotation_does_not_amplify_valid_rare_pose():
    a = np.zeros((100, 20)); a[:, ROTATION_IDS] = 1.
    a[-1, 3] = -.285
    bounds = fit_eef20_bounds(a)
    lo, hi = np.array(bounds["q01"]), np.array(bounds["q99"])
    z = 2 * (a - lo) / (hi - lo + 1e-6) - 1
    assert z[-1, 3] == pytest.approx(-.285, abs=1e-6)
    recovered = (z + 1) / 2 * (hi - lo + 1e-6) + lo
    np.testing.assert_allclose(recovered, a, atol=1e-12)
    assert np.abs(z).max() <= 1.000001


def test_norm_rejects_nonphysical_rotation():
    a = np.zeros((1, 20)); a[0, 4] = 3.
    with pytest.raises(ValueError, match="Rotation"):
        fit_eef20_bounds(a)


def test_dataset_episode_view_cannot_train_on_validation(tmp_path):
    import json
    from types import SimpleNamespace
    from n0_twam.dataset.lerobot_latent_dataset import LatentLeRobotDataset
    d = LatentLeRobotDataset.__new__(LatentLeRobotDataset)
    d.root = tmp_path; d.repo_id = str(tmp_path)
    d.meta = SimpleNamespace(episodes={i: dict(episode_index=i,tasks=["task"],
        action_config=[dict(start_frame=0,end_frame=13)]) for i in range(4)})
    d._load_valid_seg_cache = lambda: None
    d._save_valid_seg_cache = lambda x: None
    d._check_meta = lambda *args: True
    d._meta_filter_counts = {}
    (tmp_path / "episode_selection.json").write_text(json.dumps(dict(episode_indices=[0,2])))
    d.parse_meta()
    assert [m["episode_index"] for m in d.new_metas] == [0,2]
    (tmp_path / "episode_selection.json").write_text(json.dumps(dict(episode_indices=[8])))
    with pytest.raises(ValueError, match="missing"):
        d.parse_meta()


def test_rgb_configuration_does_not_silently_replace_requested_scorer(tmp_path):
    from test_task_pipeline import task
    from n0_twam.task_pipeline.config import training_config
    t = task(tmp_path); t["rgb_only"] = True; t["robot"]["tactile"] = {}
    t["features"] = {"score_backend":"qwen_sys2"}
    c = training_config(t, "ikv", 4, require_ready=False)
    assert c.kv_retention["score_backend"] == "qwen_sys2"
