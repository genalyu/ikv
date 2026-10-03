"""Behavior tests for opt-in multimodal retention, with no model downloads."""
import torch
import pytest
from n0_twam.models.global_kv_retention import RetentionConfig
from n0_twam.models.multimodal_kv_retention import make_retention_policy, ContentHistory
from test_global_kv_retention import rows, append, tiny_model
from test_ikv_training import case
from n0_twam.models.ikv_training import run_ikv_training


def policy(**kwargs):
    defaults = dict(version=2, video_capacity=2, action_capacity=2, tactile_capacity=2,
                    persistence_weight=0, visual_weight=0, contact_weight=0)
    defaults.update(kwargs)
    cfg = RetentionConfig(**defaults)
    return make_retention_policy(sum((cfg.video_capacity, cfg.action_capacity, cfg.tactile_capacity)), "cpu", cfg)


def test_no_position_or_action_vectors_in_index():
    p = policy()
    assert "grid_position" not in p.data and "action" not in p.data
    assert "action_repetition" not in p.data


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_query_usage_changes_victims_in_every_modality(kind):
    p = policy(time_weight=0)
    mask = torch.zeros(6, dtype=torch.bool)
    slots, _ = append(p, mask, rows([0, 1], kind=kind))
    p.data["query_mass"][slots] = torch.tensor([.9, .1])
    p.data["query_exposure"][slots] = 1
    _, victims = p.plan(mask, 1, rows([2], kind=kind))
    assert victims.tolist() == [int(slots[1])]
    assert len(victims) == 1  # cannot borrow another modality's free budget


def test_action_evidence_precedes_query_and_is_dynamic():
    p = policy(time_weight=0)
    mask = torch.zeros(6, dtype=torch.bool)
    visual, _ = append(p, mask, rows([0]))
    actions, _ = append(p, mask, rows([.25, 1.25], kind=1))
    p.data["query_mass"][actions] = torch.tensor([0., 1.])
    p.data["query_exposure"][actions] = 1
    _, victims = p.plan(mask, 1, rows([2.25], kind=1))
    assert victims.tolist() == [int(actions[1])]
    mask[visual] = False
    _, victims = p.plan(mask, 1, rows([2.25], kind=1))
    assert victims.tolist() == [int(actions[0])]


def test_touch_without_query_is_fifo():
    p = policy(tactile_query_weight=0)
    mask = torch.zeros(6, dtype=torch.bool)
    slots, _ = append(p, mask, rows([0, 1], kind=2))
    p.data["query_mass"][slots[0]] = 100
    _, victims = p.plan(mask, 1, rows([2], kind=2))
    assert victims.tolist() == [int(slots[0])]


def test_content_support_once_per_time_not_elapsed_gap_or_patch_count():
    h = ContentHistory(RetentionConfig(version=2, video_capacity=2, action_capacity=2,
                                      tactile_capacity=2), "cpu")
    h.observe(torch.tensor([[1., 0.]] * 20), torch.zeros(20), torch.ones(20))
    assert h.duration.tolist() == [1.]
    h.observe(torch.tensor([[1., 0.]]), torch.tensor([0.]), torch.ones(1))
    assert h.duration.tolist() == [1.]
    h.observe(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([100., 100.]), torch.ones(2))
    assert h.duration.tolist() == [2., 1.]
    assert h.persistence(torch.eye(2))[0] > h.persistence(torch.eye(2))[1]


def test_class_recency_uses_last_observed_time_within_each_dino_class():
    p = policy(video_capacity=3, time_weight=0, class_recency_weight=1,
               class_recency_scale=2, query_weight=0)
    mask = torch.zeros(7, dtype=torch.bool)
    features = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
    slots, _ = append(p, mask, rows([1, 1, 10], dino=features))
    p.observe_dense(features, torch.tensor([1., 1., 10.]), torch.ones(3))
    scores = p.scores(slots)
    torch.testing.assert_close(scores[[0, 2]], torch.ones(2))
    torch.testing.assert_close(scores[1], torch.exp(torch.tensor(-4.5)))
    assert p.components(slots)["visual"].eq(0).all()
    assert p.components(slots)["persistence"].eq(0).all()
    _, victims = p.plan(mask, 1, rows([11], dino=features[2:]))
    assert victims.tolist() == [int(slots[1])]


def test_class_recency_does_not_count_duplicate_patches_or_predictions():
    p = policy(video_capacity=2, time_weight=0, class_recency_weight=1,
               class_recency_scale=2, query_weight=0)
    mask = torch.zeros(6, dtype=torch.bool)
    slots, _ = append(p, mask, rows([1, 1], dino=torch.tensor([[1., 0.], [0., 1.]])))
    p.observe_dense(torch.tensor([[1., 0.]] * 20 + [[0., 1.]]),
                    torch.tensor([1.] * 21), torch.ones(21))
    p.observe_dense(torch.tensor([[0., 1.]]), torch.tensor([5.]), torch.ones(1))
    torch.testing.assert_close(p.components(slots)["class_recency"],
                               torch.tensor([1., torch.exp(torch.tensor(-2.)).item()]))
    assert p.history.duration.tolist() == [1., 2.]
    predicted = rows([6], dino=torch.tensor([[0., 1.]]))
    predicted["observation_flag"][:] = False
    predicted_slot, _ = append(p, mask, predicted)
    assert p.components(predicted_slot)["class_recency"].item() == 0.


def test_query_softmax_is_over_all_modalities_and_layer_average():
    p = policy(query_samples=2)
    mask = torch.zeros(6, dtype=torch.bool)
    incoming = rows([0, 0, 0])
    incoming["kind"] = torch.tensor([0, 1, 2])
    slots, _ = append(p, mask, incoming)
    q = torch.zeros(1, 2, 1, 2)
    k = torch.zeros(1, 3, 1, 2)
    measurements = [p.measure_usage(q, k, slots) for _ in range(3)]
    p.add_usage(measurements)
    torch.testing.assert_close(p.components(slots)["query"], torch.full((3,), 1/3))
    torch.testing.assert_close(p.data["query_exposure"][slots], torch.full((3,), 2.))


def test_snapshot_rolls_back_content_history():
    p = policy()
    snap = p.snapshot()
    p.observe_dense(torch.ones(1, 2), torch.zeros(1), torch.ones(1))
    p.restore(snap)
    assert p.history.clock is None and len(p.history.features) == 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_v2_real_mot_training_gradient_and_causality(checkpoint):
    model = tiny_model(False).mot
    model.gradient_checkpointing = checkpoint
    h, text, ts, temb, rope, mem = case()
    mem["config"] = dict(capacity=8, retention=dict(version=2,
        video_capacity=4, action_capacity=2, tactile_capacity=2, persistence_weight=0))
    out = run_ikv_training(model, h, text, ts, temb, rope, mem)
    out[:, 9].square().sum().backward()
    assert torch.isfinite(out).all() and h.grad[:, 4:6].abs().sum() > 0
    changed = h.detach().clone()
    future = mem["layout"]["phase"] >= 2
    changed[:, future] += 100
    other = run_ikv_training(model, changed, text, ts, temb, rope, mem)
    torch.testing.assert_close(out[:, ~future], other[:, ~future])



@pytest.mark.parametrize("motion", [False, True])
def test_full_model_v2_training_with_dense_history_and_frame_contacts(motion):
    from test_ikv_training import training_input, original_loss
    model = tiny_model(False).to(torch.bfloat16).train()
    model.use_rgb_motion_tokens = motion
    model.rgb_motion_require_index = False
    data = training_input(motion=motion)
    data["latent_dict"]["dense_dino_features"] = torch.ones(1, 3, 4, 2)
    data["latent_dict"]["frame_neoforce_features"] = torch.tensor([[[1., 0.], [0., 0.], [0., 1.]]])
    data["ikv_training"] = dict(capacity=12, retention=dict(
        version=2, video_capacity=4, action_capacity=4, tactile_capacity=4))
    output = model(data, train_mode=True)
    loss = original_loss().compute_loss(data, output)["total_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@torch.no_grad()
@pytest.mark.parametrize("motion", [False, True])
def test_v2_serving_dense_history_survives_sparse_filter_and_rollback(motion):
    from unittest.mock import patch
    from test_global_kv_retention import model_input, cache_snapshot, assert_cache_equal
    model = tiny_model(False)
    model.use_rgb_motion_tokens = motion
    model.rgb_motion_require_index = False
    model.configure_global_retention("test", version=2, video_capacity=8,
                                     action_capacity=8, tactile_capacity=8)
    def data(t):
        value = model_input(t)
        value["kv_index"] = {"dino": torch.tensor([[1.,0.], [0.,1.], [1.,0.], [0.,1.]])}
        value["tactile_kv_index"] = {"neoforce": torch.tensor([[1.,2.]] * 4)}
        if motion:
            value.update(rgb_motion_patch_indices=torch.tensor([[[0, -1]]]),
                         rgb_motion_valid_mask=torch.tensor([[[True, False]]]))
        return value
    model(data(0), update_cache=2, cache_name="test")
    p = model.mot.retention_policies["test"]
    assert p.history.duration.tolist() == [1., 1.]  # unselected content is counted
    state = model.get_global_retention("test")
    assert "grid_position" not in state
    assert state["neoforce"][state["kind"] == 0].eq(torch.tensor([1.,2.])).all()
    before = cache_snapshot(model)
    duration = p.history.duration.clone()
    with patch.object(model.proj_out, "forward", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError, match="fail"):
            model(data(1), update_cache=2, cache_name="test")
    assert_cache_equal(before, cache_snapshot(model))
    torch.testing.assert_close(p.history.duration, duration)
    model(data(1), update_cache=0, cache_name="test")
    torch.testing.assert_close(p.history.duration, duration)


def test_v2_server_broadcasts_frame_contacts_without_visual_rows():
    from test_contact_pair_server import inputs
    server, forward, index, video, tactile, packet = inputs()
    server.job_config.kv_retention = {"version": 2}
    packet.pop("visual_rows")
    server._attach_observed_contact_pairs({"contact_index": packet}, forward, index, video, tactile)
    neo = forward["latent_res_lst"]["kv_index"]["neoforce"]
    torch.testing.assert_close(neo, torch.tensor([[3.,4.],[3.,4.],[5.,6.],[5.,6.]]))
    assert torch.equal(forward["latent_res_lst"]["tactile_kv_index"]["neoforce"], packet["neoforce"])



def test_v2_evicts_low_score_then_oldest_on_tie(monkeypatch):
    p = policy(video_capacity=3, query_weight=0,
               action_query_weight=0, tactile_query_weight=0)
    mask = torch.zeros(7, dtype=torch.bool)
    slots, _ = append(p, mask, rows([0, 1, 2], kind=0))
    score_by_slot = {int(slot): score for slot, score in zip(slots, (0.9, 0.1, 0.1))}
    monkeypatch.setattr(p, "scores", lambda used, **kwargs: torch.tensor(
        [score_by_slot[int(slot)] for slot in used]))
    _, victims = p.plan(mask, 1, rows([3], kind=0))
    assert victims.tolist() == [int(slots[1])]


def test_history_host_decisions_preserve_ties_seen_slots_and_durations():
    c = RetentionConfig(version=2, video_capacity=2, action_capacity=2,
                        tactile_capacity=2, content_capacity=2, content_threshold=0.9)
    h = ContentHistory(c, "cpu")
    h.observe(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
              torch.zeros(3), torch.tensor([1., 3., 2.]))
    torch.testing.assert_close(h.duration, torch.tensor([3., 2.]))
    before = h.snapshot()
    # Equal oldest timestamps choose the first slot; seen slots cannot be evicted
    # again in the same frame. A third new prototype therefore cannot be admitted.
    h.observe(torch.tensor([[-1., 0.], [0., -1.], [1., 0.]]),
              torch.ones(3), torch.tensor([4., 5., 6.]))
    torch.testing.assert_close(h.features, torch.tensor([[-1., 0.], [0., -1.]]))
    torch.testing.assert_close(h.duration, torch.tensor([4., 5.]))
    torch.testing.assert_close(h.last_time, torch.ones(2))
    h.restore(before)
    torch.testing.assert_close(h.features, torch.eye(2))
    torch.testing.assert_close(h.duration, torch.tensor([3., 2.]))
    assert h.clock == 0.


def test_history_ignores_zero_content_but_advances_clock():
    c = RetentionConfig(version=2, video_capacity=2, action_capacity=2,
                        tactile_capacity=2, content_capacity=2)
    h = ContentHistory(c, "cpu")
    h.observe(torch.zeros(3, 2), torch.tensor([2., 0., 1.]), torch.ones(3))
    assert h.features.numel() == 0
    assert h.clock == 2.
