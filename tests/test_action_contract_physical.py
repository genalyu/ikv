"""Physical pose scaling and active-channel supervision on the real preprocessing path."""
from types import SimpleNamespace

import numpy as np
import pytest

from n0_twam.dataset.lerobot_latent_dataset import LatentLeRobotDataset
from n0_twam.task_pipeline.normalization import absolute_rotation_bounds


@pytest.mark.parametrize("arms,raw_width", [(1, 10), (1, 20), (2, 20)])
def test_padded_actions_and_cold_seed_use_only_active_columns(arms, raw_width):
    used = list(range(10 * arms))
    inverse = used + [len(used)] * (20 - len(used))
    repo = SimpleNamespace(used_action_channel_ids=used, inverse_used_action_channel_ids=inverse,
                           q01=-np.ones((1, 20)), q99=np.ones((1, 20)))
    action = np.full((25, raw_width), 0.75)
    seed = np.full(raw_width, -0.25)
    if arms == 1:
        action[:, 10:] = 123.
        seed[10:] = 456.
    values, mask = LatentLeRobotDataset._action_post_process(repo, 0, len(action),
        np.arange(0, 25, 3), action, initial_state=seed)
    assert mask[:len(used)].all() and not mask[len(used):].any()
    assert (values[len(used):] == 0).all()
    np.testing.assert_allclose(values[:len(used), 0], -.25, atol=2e-6)
    np.testing.assert_allclose(values[:len(used), 1:], .75, atol=2e-6)


@pytest.mark.parametrize("arms", [1, 2])
def test_real_pose_outside_percentile_span_stays_reversible(arms):
    old = {"q01": [-1.] * 20, "q99": [-0.9990143] * 20}
    stats = absolute_rotation_bounds(old, arms)
    changed = [j for j,(a,b) in enumerate(zip(old["q99"], stats["q99"])) if a != b]
    assert changed == [10 * arm + j for arm in range(arms) for j in range(3, 9)]
    assert stats["q01"][7] == -1. and stats["q99"][7] == 1.
    assert np.isclose(2 * (-.97804248-stats["q01"][7]) /
                      (stats["q99"][7]-stats["q01"][7]) - 1, -.97804248)


def test_refuse_checkpoint_with_legacy_action_supervision(tmp_path):
    import json
    from n0_twam.task_pipeline.checkpoint import load_training_state
    cfg = SimpleNamespace(world_size=8, rank=0, task_fingerprint="same", task_mode="ikv",
                          action_contract_fingerprint="physical-rotation-active-mask")
    (tmp_path / "complete.json").write_text(json.dumps(
        dict(world_size=8, task_fingerprint="same", mode="ikv")))
    with pytest.raises(ValueError, match="action_contract_fingerprint mismatch"):
        load_training_state(SimpleNamespace(config=cfg), tmp_path)
