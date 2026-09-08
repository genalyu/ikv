"""CPU policy + real two-layer MoT tests; no checkpoints or DINO downloads."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from n0_twam.models.global_kv_retention import GlobalKVRetention, RetentionConfig, token_rows
from n0_twam.models.mot import WanMoTTransformer3DModel
from n0_twam.preprocessing.kv_index import observed_index, prediction_index, concat_indices, encode_dense_dino


def rows(times, *, kind=0, observed=True, dino=None, neo=None, duration=1, actions=None):
    n = len(times)
    grid = torch.zeros(1, 4, n)
    grid[0, 0] = torch.tensor(times)
    index = {'observation_flag': int(observed), 'duration': duration}
    if dino is not None:
        index['dino'] = dino
    if neo is not None:
        index['neoforce'] = neo
    result = token_rows({'grid_id': grid, 'index': index, 'actions': actions},
                        batch_size=1, length=n, main_count=n, action_mode=kind == 1,
                        update_cache=2 if observed else 1, device='cpu')
    result['kind'][:] = kind
    return result


def append(policy, mask, incoming):
    slots, victims = policy.plan(mask, len(incoming['kind']), incoming)
    policy.commit(slots, incoming, mask.clone())
    mask[slots] = True
    return slots, victims


def assert_snapshot_equal(a, b):
    assert a[:3] == b[:3]
    torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)
    for name in a[4]:
        torch.testing.assert_close(a[4][name], b[4][name], rtol=0, atol=0)


def test_global_topk_across_experts_and_random_exact_deficit():
    policy = GlobalKVRetention(10, 'cpu', RetentionConfig(
        top_k=3, contact_weight=0, visual_weight=0, time_weight=0,
        query_weight=1, repetition_weight=0, seed=19))
    mask = torch.zeros(10, dtype=torch.bool)
    incoming = rows(range(10))
    incoming['kind'] = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    append(policy, mask, incoming)
    policy.data['query_mass'][:] = torch.arange(10.)
    policy.data['query_exposure'][:] = 1
    newer = rows([20, 21, 22, 23], observed=False)
    before = policy.snapshot()
    slots, victims = policy.plan(mask, 4, newer)
    assert len(victims) == 4
    assert set(victims.tolist()).isdisjoint({7, 8, 9})
    # No RNG/state mutation at planning time; failed/temporary calls repeat.
    assert torch.equal(victims, policy.plan(mask, 4, newer)[1])
    assert_snapshot_equal(before, policy.snapshot())
    append(policy, mask, newer)
    assert mask.sum() == 10
    assert policy.data['token_uid'][7:10].tolist() == [7, 8, 9]
    assert policy.t0 == 9


def test_t0_is_latest_real_video_and_time_is_two_sided():
    policy = GlobalKVRetention(8, 'cpu', RetentionConfig(time_scale=2))
    mask = torch.zeros(8, dtype=torch.bool)
    append(policy, mask, rows([10]))
    slots, _ = append(policy, mask, rows([8, 12], observed=False))
    time = policy.components(slots)['time']
    torch.testing.assert_close(time, torch.full((2,), torch.exp(torch.tensor(-1.))))
    append(policy, mask, rows([20.5], kind=1, observed=True))
    assert policy.t0 == 10
    append(policy, mask, rows([9], observed=True))
    assert policy.t0 == 10
    append(policy, mask, rows([11], observed=True))
    assert policy.t0 == 11


def test_zero_sentinel_and_contact_interval_not_layer_or_step_count():
    policy = GlobalKVRetention(6, 'cpu')
    mask = torch.zeros(6, dtype=torch.bool)
    incoming = rows([0, 1, 2], neo=torch.tensor([[0., 0.], [1., -1.], [0., .01]]),
                    duration=[3., 2., .5], dino=torch.zeros(3, 2))
    slots, _ = append(policy, mask, incoming)
    assert policy.data['contact_duration'][slots].tolist() == [0, 2, .5]
    assert policy.data['contact_time'][slots].tolist() == [-torch.inf, 1, 2]
    assert policy.components(slots)['visual'].eq(0).all()
    assert 'visual_valid' not in policy.data and 'tactile_valid' not in policy.data
    snapshot = policy.snapshot()
    for _ in range(10):
        policy.plan(mask, 3, incoming)
    assert_snapshot_equal(snapshot, policy.snapshot())


def test_visual_relevance_and_action_redundancy_with_zero_action():
    policy = GlobalKVRetention(8, 'cpu')
    mask = torch.zeros(8, dtype=torch.bool)
    slots, _ = append(policy, mask, rows([0, 0], dino=torch.tensor([[1., 0.], [0., 1.]])))
    append(policy, mask, rows([1], dino=torch.tensor([[1., 0.]])))
    assert policy.components(slots)['visual'].tolist() == [1, 0]
    action_slots, _ = append(policy, mask, rows(
        [1, 2, 3], kind=1, actions=torch.tensor([[[[[0.], [0.], [1.]]]]])))
    assert policy.data['action_repetition'][action_slots].tolist() == pytest.approx([0, 1, 0], abs=1e-6)


def test_query_usage_tracks_attention_not_just_key_presence():
    policy = GlobalKVRetention(2, 'cpu', RetentionConfig(query_samples=2))
    q = torch.tensor([[[[5., 0.]], [[5., 0.]]]])
    k = torch.tensor([[[[5., 0.]], [[-5., 0.]]]])
    measurement = policy.measure_usage(q, k, torch.arange(2))
    policy.add_usage([measurement, measurement])  # two layers count once
    assert policy.data['query_exposure'].tolist() == [2, 2]
    assert policy.data['query_mass'][0] > 1.99
    assert policy.data['query_mass'][1] < .01
    before = policy.components(torch.arange(2))['query']
    policy.add_usage([measurement])
    torch.testing.assert_close(policy.components(torch.arange(2))['query'], before)


def test_capacity_guard_and_topk_reserves_current_input():
    policy = GlobalKVRetention(3, 'cpu', RetentionConfig(top_k=100))
    mask = torch.ones(3, dtype=torch.bool)
    assert len(policy.plan(mask, 3, rows([1, 2, 3]))[1]) == 3
    with pytest.raises(ValueError, match='exceeds'):
        policy.plan(mask, 4, rows([1, 2, 3, 4]))


def tiny_model(global_policy=True):
    model = WanMoTTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=1, attention_head_dim=4,
        in_channels=2, out_channels=2, action_dim=3, text_dim=8, freq_dim=4,
        ffn_dim=8, num_layers=2, rope_max_seq_len=32, attn_mode='torch',
        use_local_tactile=False, use_rgb_motion_tokens=False).eval()
    model.create_empty_cache('test', 2, 16, 8, torch.device('cpu'), torch.float32, 1)
    for attention in model.mot.shared_attn:
        attention.attn_caches['test']['k'].zero_()
        attention.attn_caches['test']['v'].zero_()
    if global_policy:
        model.configure_global_retention('test', top_k=3, seed=17)
    return model


def model_input(time=0, action=False):
    count = 2 if action else 4
    grid = torch.zeros(1, 4, count)
    grid[:, 0] = time
    return {
        'noisy_latents': torch.randn(1, 3, 1, 2, 1) if action else torch.randn(1, 2, 1, 4, 4),
        'timesteps': torch.zeros(1, 1), 'grid_id': grid,
        'text_emb': torch.randn(1, 3, 8),
        'tactile_global_latent': torch.randn(1, 1, 48, 1, 4, 4),
        'tactile_sensor_ids': torch.zeros(1, 1, dtype=torch.long),
    }


def cache_snapshot(model):
    return [{name: c[name].clone() for name in ('k', 'v', 'mask', 'id', 'is_pred')}
            for c in (a.attn_caches['test'] for a in model.mot.shared_attn)]


def assert_cache_equal(a, b):
    for first, second in zip(a, b):
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


@torch.no_grad()
def test_dense_index_does_not_change_embeddings_kv_or_output_without_eviction():
    indexed = tiny_model()
    baseline = tiny_model(False)
    baseline.load_state_dict(indexed.state_dict(), strict=True)
    data = model_input()
    expected = baseline(deepcopy(data), update_cache=2, cache_name='test')
    data['kv_index'] = {'dino': torch.randn(4, 5), 'neoforce': torch.randn(4, 2)}
    actual = indexed(deepcopy(data), update_cache=2, cache_name='test')
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert_cache_equal(cache_snapshot(indexed), cache_snapshot(baseline))
    stats = indexed.get_global_retention('test')
    assert stats['t0'] == 0 and len(stats['token_uid']) == 8  # all 4 RGB + 4 tactile
    assert stats['dino'].shape == (8, 5)
    assert stats['dino'][4:].eq(0).all()
    assert indexed.mot.shared_attn[0].attn_caches['test']['semantic'] is None


@torch.no_grad()
def test_all_experts_share_one_plan_and_temporary_forward_is_inert():
    model = tiny_model()
    for time, action in [(0, False), (1, True), (2, False), (3, False)]:
        model(model_input(time, action), update_cache=2, cache_name='test', action_mode=action)
    caches = cache_snapshot(model)
    assert caches[0]['mask'].sum() == 24
    for other in caches[1:]:
        assert torch.equal(caches[0]['mask'], other['mask'])
        assert torch.equal(caches[0]['id'], other['id'])
    policy = model.mot.retention_policies['test']
    before = policy.snapshot()
    model(model_input(9), update_cache=0, cache_name='test')
    assert_cache_equal(caches, cache_snapshot(model))
    assert_snapshot_equal(before, policy.snapshot())
    assert policy.t0 == 3


@torch.no_grad()
@pytest.mark.parametrize('failure', ['second_layer', 'output', 'metadata_width'])
def test_failed_forward_restores_layers_statistics_and_random_state(failure):
    model = tiny_model()
    for t in range(3):
        data = model_input(t)
        data['kv_index'] = {'dino': torch.randn(4, 2)}
        model(data, update_cache=2, cache_name='test')
    policy = model.mot.retention_policies['test']
    before = policy.snapshot()
    caches = cache_snapshot(model)
    data = model_input(4)
    if failure == 'metadata_width':
        data['kv_index'] = {'dino': torch.randn(4, 3)}
        with pytest.raises(ValueError, match='width changed'):
            model(data, update_cache=2, cache_name='test')
    else:
        target = model.mot.shared_attn[1] if failure == 'second_layer' else model.proj_out
        with patch.object(target, 'forward', side_effect=RuntimeError('injected')):
            with pytest.raises(RuntimeError, match='injected'):
                model(data, update_cache=2, cache_name='test')
    assert_cache_equal(caches, cache_snapshot(model))
    assert_snapshot_equal(before, policy.snapshot())
    assert not model.mot._active_cache_transactions
    model(model_input(4), update_cache=2, cache_name='test')
    assert policy.t0 == 4


def test_feature_shapes_cfg_agreement_and_zero_fill():
    grid = torch.zeros(2, 4, 3)
    context = {'grid_id': grid, 'index': {'dino': torch.ones(2, 3, 2)}}
    kwargs = dict(batch_size=2, length=3, main_count=3, action_mode=False, update_cache=2, device='cpu')
    assert token_rows(context, **kwargs)['dino'].shape == (3, 2)
    context['index']['dino'][1, 0, 0] = 4
    with pytest.raises(ValueError, match='CFG'):
        token_rows(context, **kwargs)
    context['index'] = {}
    grid[1, 0, 0] = 1
    with pytest.raises(ValueError, match='CFG'):
        token_rows(context, **kwargs)


def test_dense_camera_and_time_order_with_no_motion_filter():
    from types import SimpleNamespace
    videos = torch.zeros(2, 3, 4, 2, 2)
    for camera in range(2):
        for t in range(4):
            videos[camera, :, t] = 10 * camera + t
    encoder = lambda rgb: SimpleNamespace(tokens=rgb[:, :1].permute(0, 2, 3, 1))
    features = encode_dense_dino(videos, torch.tensor([1, 3]), (2, 2), encoder)
    assert features[:, 0].tolist() == [1, 1, 11, 11, 1, 1, 11, 11,
                                      3, 3, 13, 13, 3, 3, 13, 13]
    seed = observed_index({'dino': features[:8]}, 8, 'cpu')
    predicted = prediction_index(24, 'cpu', seed)
    assert predicted['observation_flag'].sum() == 8
    assert predicted['dino'][8:].eq(0).all()
    assert predicted['neoforce'].numel() == 0
    merged = concat_indices(seed, observed_index({}, 8, 'cpu'))
    assert merged['dino'].shape == (16, 1)
    assert merged['dino'][8:].eq(0).all()


@pytest.mark.parametrize('payload', [
    {'visual_valid': [False]}, {'dino': [[float('nan')]]},
    {'observation_flag': 0}, {'duration': -1}, {'neoforce': [[1], [2]]},
])
def test_observed_boundary_rejects_invalid_payload(payload):
    with pytest.raises(ValueError):
        observed_index(payload, 1, 'cpu')


@torch.no_grad()
def test_output_dino_labels_existing_video_kv_without_touching_real_seed_or_clock():
    model = tiny_model()
    cursor = model.global_cache_cursor('test')
    data = model_input(5)
    data['kv_index'] = {'observation_flag': [1, 0, 0, 0],
                        'dino': torch.tensor([[1., 2.], [0., 0.], [0., 0.], [0., 0.]])}
    model(data, update_cache=1, cache_name='test')
    handle = model.video_index_handle('test', cursor)
    policy = model.mot.retention_policies['test']
    before = policy.snapshot()
    caches = cache_snapshot(model)
    assert model.annotate_video_dino('test', handle, torch.ones(4, 2) * 3) == 3
    assert_cache_equal(caches, cache_snapshot(model))
    assert policy.t0 == 5 and policy.next_uid == before[1] and policy.revision == before[2]
    torch.testing.assert_close(policy.reference_dino, before[3])
    assert policy.data['dino'][handle['slots']].tolist() == [[1, 2], [3, 3], [3, 3], [3, 3]]
    for name in policy.data:
        if name != 'dino':
            torch.testing.assert_close(policy.data[name], before[4][name])


def test_backfill_rejects_stale_and_reset_handles_without_mutation():
    policy = GlobalKVRetention(4, 'cpu', RetentionConfig(top_k=0))
    mask = torch.zeros(4, dtype=torch.bool)
    append(policy, mask, rows([0, 1, 2, 3], observed=False))
    handle = policy.video_handle(mask, 0)
    append(policy, mask, rows([4, 5, 6, 7], observed=False))
    before = policy.snapshot()
    with pytest.raises(ValueError, match='stale'):
        policy.annotate_video_dino(mask, handle, torch.ones(4, 2))
    assert_snapshot_equal(before, policy.snapshot())
    with pytest.raises(ValueError, match='generation'):
        GlobalKVRetention(4, 'cpu').annotate_video_dino(mask, handle, torch.ones(4, 2))
    new_handle = policy.video_handle(mask, 4)
    # Eviction scatters physical slots, but handle order follows input UID.
    assert new_handle['world_time_id'].tolist() == [4, 5, 6, 7]
    assert new_handle['token_uid'].tolist() == [4, 5, 6, 7]


@torch.no_grad()
def test_backfill_is_rolled_back_with_later_action_failure():
    model = tiny_model()
    policy = model.mot.retention_policies['test']
    before = policy.snapshot()
    caches = cache_snapshot(model)
    with pytest.raises(RuntimeError, match='action failure'):
        with model.cache_transaction('test'):
            cursor = model.global_cache_cursor('test')
            model(model_input(2), update_cache=1, cache_name='test')
            handle = model.video_index_handle('test', cursor)
            model.annotate_video_dino('test', handle, torch.ones(4, 3))
            with patch.object(model.mot.shared_attn[1], 'forward', side_effect=RuntimeError('action failure')):
                model(model_input(2, True), update_cache=1, cache_name='test', action_mode=True)
    assert_snapshot_equal(before, policy.snapshot())
    assert_cache_equal(caches, cache_snapshot(model))


@pytest.mark.parametrize('features', [torch.ones(3, 2), torch.full((4, 2), torch.nan), torch.ones(4, 3)])
def test_backfill_validates_entire_packet_before_mutation(features):
    policy = GlobalKVRetention(4, 'cpu')
    mask = torch.zeros(4, dtype=torch.bool)
    append(policy, mask, rows([0, 1, 2, 3], observed=False, dino=torch.ones(4, 2)))
    before = policy.snapshot()
    with pytest.raises(ValueError):
        policy.annotate_video_dino(mask, policy.video_handle(mask, 0), features)
    assert_snapshot_equal(before, policy.snapshot())
