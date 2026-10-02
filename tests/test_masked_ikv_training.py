"""One-pass IKV uses only retention-selected historical condition tokens."""
from copy import deepcopy

import pytest
import torch

from n0_twam.models.ikv_training import (
    build_ikv_support_plan, run_ikv_masked_training, run_ikv_training,
)
from test_global_kv_retention import tiny_model
from test_ikv_training import case


def prepared_case(capacity, version):
    hidden, text, timestep, temb, rope, memory = case()
    memory["splits"] = [4, 4, 2, 2, 2, 2, 0]
    memory["config"]["capacity"] = capacity
    memory["config"]["retention"].update(
        query_weight=0.0, action_query_weight=0.0, tactile_query_weight=0.0)
    if version == 2:
        memory["config"]["retention"].update(
            version=2, video_capacity=2, action_capacity=1,
            tactile_capacity=1, persistence_weight=0.0)
    return hidden, text, timestep, temb, rope, memory


@pytest.mark.parametrize("capacity,version", [(32, 1), (4, 1), (4, 2)])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_one_pass_matches_recurrent_output_and_gradient(capacity, version, checkpoint):
    left = tiny_model(False).mot
    right = deepcopy(left)
    left.gradient_checkpointing = right.gradient_checkpointing = checkpoint
    hidden, text, timestep, temb, rope, memory = prepared_case(capacity, version)
    a = hidden.detach().clone().requires_grad_()
    b = hidden.detach().clone().requires_grad_()
    expected = run_ikv_training(left, a, text, timestep, temb, rope, memory)
    actual = run_ikv_masked_training(right, b, text, timestep, temb, rope, memory)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(b.grad, a.grad, rtol=3e-4, atol=3e-6)
    for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, rtol=3e-4, atol=3e-6, msg=lambda m: name + ": " + m)


def test_evicted_recent_and_retained_old_tokens():
    *_, memory = prepared_case(4, 1)
    plan = build_ikv_support_plan(memory, torch.device("cpu"))
    clean = memory["layout"]["clean"]
    phase = memory["layout"]["phase"]
    stage = int(phase.max())
    historical = clean & (phase < stage)
    selected = plan["noisy"][stage] & historical
    assert selected.sum() <= memory["config"]["capacity"]
    assert (~selected & historical).any(), "some recent history must be evicted"


def test_query_usage_weights_must_be_zero():
    *_, memory = prepared_case(4, 1)
    memory["config"]["retention"]["query_weight"] = 1.0
    with pytest.raises(ValueError, match="zero query-usage"):
        build_ikv_support_plan(memory, torch.device("cpu"))


def test_serving_skips_query_usage_when_all_weights_are_zero(monkeypatch):
    from test_global_kv_retention import model_input
    model = tiny_model(False)
    model.configure_global_retention(
        "test", query_weight=0.0, action_query_weight=0.0,
        tactile_query_weight=0.0)
    policy = model.mot.retention_policies["test"]
    def unexpected(*args, **kwargs):
        raise AssertionError("query usage should not be measured")
    monkeypatch.setattr(policy, "measure_usage", unexpected)
    with torch.no_grad():
        model(model_input(), update_cache=2, cache_name="test")
    assert policy.data["query_mass"].eq(0).all()
    assert policy.data["query_exposure"].eq(0).all()


def test_future_truth_and_same_phase_condition_do_not_leak():
    model = tiny_model(False).mot
    hidden, text, timestep, temb, rope, memory = prepared_case(4, 1)
    memory["rows"]["dino"] = torch.randn(len(memory["layout"]["seq"]), 3)
    changed = deepcopy(memory)
    phase = memory["layout"]["phase"]
    clean = memory["layout"]["clean"]
    future = phase >= 2
    changed["rows"]["dino"][future & clean] *= -100
    hidden_changed = hidden.detach().clone()
    hidden_changed[:, future & clean] += 100
    before = build_ikv_support_plan(memory, torch.device("cpu"))
    after = build_ikv_support_plan(changed, torch.device("cpu"))
    torch.testing.assert_close(before["noisy"][2], after["noisy"][2])
    first = run_ikv_masked_training(model, hidden, text, timestep, temb, rope, memory)
    second = run_ikv_masked_training(
        model, hidden_changed, text, timestep, temb, rope, changed)
    torch.testing.assert_close(first[:, phase < 2], second[:, phase < 2])
    torch.testing.assert_close(first[:, (phase == 2) & ~clean],
                               second[:, (phase == 2) & ~clean])


def test_random_v2_capacity_respects_modality_minima_and_is_reproducible():
    *_, memory = prepared_case(10, 2)
    memory["config"]["retention"].update(
        video_capacity=4, action_capacity=3, tactile_capacity=3)
    memory["config"]["sample_capacity"] = True
    values = []
    for seed in range(20):
        torch.manual_seed(seed)
        from n0_twam.models.global_kv_retention import RetentionConfig
        from n0_twam.models.ikv_training import sample_ikv_capacity
        k, cfg = sample_ikv_capacity(
            memory, torch.device("cpu"), RetentionConfig(**memory["config"]["retention"]))
        budgets = memory["config"]["sampled_modality_capacities"]
        assert 5 <= k <= 10 and sum(budgets) == k
        assert all(low <= got <= high for low, got, high in zip(
            (2, 1, 1), budgets, (4, 3, 3)))
        assert (cfg.video_capacity, cfg.action_capacity, cfg.tactile_capacity) == tuple(budgets)
        torch.manual_seed(seed)
        repeated, _ = sample_ikv_capacity(
            memory, torch.device("cpu"), RetentionConfig(**memory["config"]["retention"]))
        assert repeated == k
        values.append(k)
    assert len(set(values)) > 1


def test_random_budget_one_pass_matches_fixed_budget_recurrent():
    left = tiny_model(False).mot
    right = deepcopy(left)
    hidden, text, timestep, temb, rope, memory = prepared_case(10, 2)
    memory["config"]["retention"].update(
        video_capacity=4, action_capacity=3, tactile_capacity=3)
    memory["config"]["sample_capacity"] = True
    torch.manual_seed(4)
    actual = run_ikv_masked_training(right, hidden, text, timestep, temb, rope, memory)
    assert right.last_sampled_ikv_capacity == memory["config"]["sampled_capacity"]
    fixed = deepcopy(memory)
    fixed["config"]["capacity"] = memory["config"]["sampled_capacity"]
    fixed["config"]["retention"].update(zip(
        ("video_capacity", "action_capacity", "tactile_capacity"),
        memory["config"]["sampled_modality_capacities"]))
    expected = run_ikv_training(left, hidden, text, timestep, temb, rope, fixed)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)



def test_sampled_k_does_not_depend_on_future_layout():
    from n0_twam.models.global_kv_retention import RetentionConfig
    from n0_twam.models.ikv_training import sample_ikv_capacity
    *_, memory = prepared_case(10, 2)
    memory["config"]["retention"].update(
        video_capacity=4, action_capacity=3, tactile_capacity=3)
    memory["config"]["sample_capacity"] = True
    future = deepcopy(memory)
    phase = future["layout"]["phase"]
    future["layout"]["kind"][phase >= 2] = 2
    cfg = RetentionConfig(**memory["config"]["retention"])
    torch.manual_seed(37)
    k1, r1 = sample_ikv_capacity(memory, torch.device("cpu"), cfg)
    torch.manual_seed(37)
    k2, r2 = sample_ikv_capacity(future, torch.device("cpu"), cfg)
    assert k1 == k2 and r1 == r2


@pytest.mark.parametrize("q_len,kv_len", [(512, 512), (320, 448)])
def test_refined_mask_preserves_visibility_and_coarse_full_block_order(q_len, kv_len):
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
    from n0_twam.models.ikv_training import _preserve_parent_full_blocks

    def visible(b, h, q, k):
        return (q < q_len) & (k < kv_len) & ((q // 160 == k // 160) | (k < 64))

    coarse = create_block_mask(visible, 1, 1, q_len, kv_len, device="cpu", BLOCK_SIZE=128)
    fine = create_block_mask(visible, 1, 1, q_len, kv_len, device="cpu", BLOCK_SIZE=64)
    refined = _preserve_parent_full_blocks(fine)
    assert torch.equal(refined.to_dense(), fine.to_dense())
    assert refined.mask_mod is fine.mask_mod

    def full_blocks(mask):
        return BlockMask.from_kv_blocks(
            mask.full_kv_num_blocks, mask.full_kv_indices,
            BLOCK_SIZE=mask.BLOCK_SIZE, seq_lengths=mask.seq_lengths,
        ).to_dense()

    expected = full_blocks(coarse).repeat_interleave(2, -2).repeat_interleave(2, -1)
    actual = full_blocks(refined)
    assert torch.equal(actual, expected[..., :actual.shape[-2], :actual.shape[-1]])
