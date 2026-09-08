"""Real serving helpers with deterministic VAE/DINO stubs (no downloads)."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from test_rgb_motion_server_helpers import _server, TWAMServer
from n0_twam.models.global_kv_retention import GlobalKVRetention, token_rows
from n0_twam.utils.utils import get_mesh_id


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(latents_mean=[.1], latents_std=[.5], scale_factor_temporal=4)
        self._feat_map = ['original decoder state']
        self._enc_feat_map = ['original encoder state']
        self.inputs = []
        self.fail = False

    def decode(self, latents, return_dict=False):
        self.inputs.append(latents.clone())
        self._feat_map = ['changed decoder state']
        self._enc_feat_map = ['changed encoder state']
        if self.fail:
            raise RuntimeError('decode failed')
        frames = []
        for t in range(latents.shape[2]):
            rgb = F.interpolate(latents[:, :1, t], size=(64, 64)).repeat(1, 3, 1, 1)
            frames.append(rgb[:, :, None].repeat(1, 1, 1 if t == 0 else 4, 1, 1))
        return (torch.cat(frames, dim=2),)


def dense_server():
    server = _server(enabled=False, camera_keys=('left', 'right'))
    server.job_config.kv_cache_policy = 'global'
    server.job_config.kv_index_predicted_dino = True
    server.height = server.width = 64
    server.vae = Decoder()
    server.streaming_vae.vae = server.vae
    server.cache_name = 'test'
    server.dino_inputs = []

    def encoder(rgb):
        server.dino_inputs.append(rgb.clone())
        return SimpleNamespace(tokens=rgb[:, :1].permute(0, 2, 3, 1))
    server._kv_dino_encoder = encoder
    return server


def latent_chunk():
    latents = torch.zeros(1, 1, 2, 4, 8)
    latents[:, :, 0, :, :4] = .2
    latents[:, :, 1, :, :4] = .4
    latents[:, :, 0, :, 4:] = .6
    latents[:, :, 1, :, 4:] = .8
    return latents


@pytest.mark.parametrize('frame_start', [0, 10])
def test_decoder_camera_split_normalization_and_causal_endpoints(frame_start):
    server = dense_server()
    server._last_observed_video_latent = torch.full((1, 1, 1, 4, 8), -.8)
    encoder_state = server.streaming_vae.feat_cache
    decoder_state = server.vae._feat_map
    raw_encoder_state = server.vae._enc_feat_map
    videos, anchors = server._decode_prediction_for_index(latent_chunk(), frame_start)
    assert videos.shape == (2, 3, 5 if frame_start == 0 else 9, 64, 64)
    assert anchors.tolist() == ([0, 4] if frame_start == 0 else [4, 8])
    torch.testing.assert_close(videos[:, 0, anchors, 0, 0], torch.tensor([[.6, .65], [.7, .75]]))
    assert server.streaming_vae.feat_cache is encoder_state
    assert server.vae._feat_map is decoder_state
    assert server.vae._enc_feat_map is raw_encoder_state
    assert server.vae.inputs[0].shape[0] == 2  # cameras are batches, not adjacent decoder pixels
    if frame_start:
        torch.testing.assert_close(server.vae.inputs[0][:, :, 0], torch.full((2, 1, 4, 4), -.3))


def test_decoder_failure_preserves_state_and_missing_prefix_is_rejected():
    server = dense_server()
    with pytest.raises(ValueError, match='most recent real latent'):
        server._decode_prediction_for_index(latent_chunk(), 3)
    before = server.vae._feat_map
    server.vae.fail = True
    with pytest.raises(RuntimeError, match='decode failed'):
        server._decode_prediction_for_index(latent_chunk(), 0)
    assert server.vae._feat_map is before


@pytest.mark.parametrize('frame_start', [0, 10])
def test_backfill_matches_time_camera_patch_order_and_keeps_seed(frame_start):
    server = dense_server()
    server._last_observed_video_latent = torch.full((1, 1, 1, 4, 8), -.8)
    grid = get_mesh_id(2, 2, 4, 0, 1, frame_start)[None]
    flags = torch.zeros(16, dtype=torch.bool)
    if frame_start == 0:
        flags[:8] = True
    index = {'observation_flag': flags, 'dino': torch.full((16, 1), .9)}
    rows = token_rows({'grid_id': grid, 'index': index}, batch_size=1, length=16,
                      main_count=16, action_mode=False, update_cache=1, device='cpu')
    policy = GlobalKVRetention(16, 'cpu')
    mask = torch.zeros(16, dtype=torch.bool)
    policy.commit(torch.arange(16), rows, mask)
    mask[:] = True
    handle = policy.video_handle(mask, 0)
    server.transformer = SimpleNamespace(annotate_video_dino=lambda name, handle, features:
                                         policy.annotate_video_dino(mask, handle, features))
    count = server._backfill_predicted_video_index(latent_chunk(), frame_start, handle)
    assert count == (8 if frame_start == 0 else 16)
    dino = policy.data['dino'].reshape(2, 2, 4)
    expected = torch.tensor([[[.6, .6, .7, .7]] * 2, [[.65, .65, .75, .75]] * 2])
    if frame_start == 0:
        expected[0] = .9
    torch.testing.assert_close(dino, expected)
    assert policy.t0 == (0 if frame_start == 0 else None)
    assert len(server.vae.inputs) == 1 and len(server.dino_inputs) == 2
    # A permutation with the right token count must not silently label wrong patches.
    handle['grid_position'] = handle['grid_position'].flip(0)
    with pytest.raises(ValueError, match='grid'):
        server._backfill_predicted_video_index(latent_chunk(), frame_start, handle)


def test_observation_index_uses_dense_rgb_without_depth_or_motion():
    server = dense_server()
    videos = torch.zeros(2, 3, 1, 64, 64)
    videos[0] = .2
    videos[1] = .6
    index = server._prepare_dense_observed_index({}, videos)
    torch.testing.assert_close(index['dino'][:, 0], torch.tensor([.2, .2, .6, .6] * 2))
    assert index['observation_flag'].all() and index['neoforce'].shape == (8, 0)
    assert len(server.dino_inputs) == 2
    # Precomputed DINO bypasses the encoder, while zero NeoForce stays absent.
    server._get_kv_dino_encoder = lambda: pytest.fail('must not load DINO')
    precomputed = server._prepare_dense_observed_index(
        {'kv_index': {'dino': torch.ones(8, 3), 'neoforce': torch.zeros(8, 2)}}, videos)
    assert precomputed['dino'].shape == (8, 3)


def test_disabled_prediction_index_does_not_need_decoder_or_dino():
    server = dense_server()
    assert server._predicted_dino_enabled()
    server.job_config.kv_index_predicted_dino = False
    assert not server._predicted_dino_enabled()
    server.job_config.kv_index_predicted_dino = True
    server.job_config.kv_cache_policy = 'fifo'
    assert not server._predicted_dino_enabled()


@torch.no_grad()
def test_real_grounding_keeps_predicted_video_and_action_kv_and_dino(monkeypatch):
    from test_global_kv_retention import tiny_model, model_input
    from n0_twam.preprocessing.kv_index import observed_index

    server = dense_server()
    server.frame_st_id = 5
    server.exp_save_root = '/tmp'
    server.job_config.action_dim = 3
    server.action_mask = torch.ones(3, dtype=torch.bool)
    server.prompt_embeds = torch.randn(1, 3, 8)
    server.use_cfg = False
    server._last_gen_tactile = None
    server._last_gen_tactile_fsid = None
    model = tiny_model()
    # Leave enough room so this test isolates grounding from capacity eviction.
    model.create_empty_cache('test', 4, 16, 8, torch.device('cpu'), torch.float32, 1)
    model.configure_global_retention('test')
    server.transformer = model
    cursor = model.global_cache_cursor('test')
    model(model_input(5), update_cache=1, cache_name='test')
    model.annotate_video_dino('test', model.video_index_handle('test', cursor),
                              torch.arange(1., 9.).reshape(4, 2))
    model(model_input(5, True), update_cache=1, cache_name='test', action_mode=True)
    before = model.get_global_retention('test')
    old_slots = before['slot_indices']
    old_layers = [{name: attention.attn_caches['test'][name][:, old_slots].clone()
                   for name in ('k', 'v')} for attention in model.mot.shared_attn]
    observed = model_input(5)

    def encode_obs(_):
        server._current_observed_kv_index = observed_index({'dino': torch.ones(4, 2)}, 4, 'cpu')
        return observed['noisy_latents']

    server._encode_obs = encode_obs
    server.preprocess_action = lambda *args, **kwargs: model_input(5, True)['noisy_latents']
    server._encode_tactile_obs = lambda _: {
        'tactile_global_latent': observed['tactile_global_latent'],
        'tactile_sensor_ids': observed['tactile_sensor_ids'],
    }
    monkeypatch.setitem(TWAMServer._infer_impl.__globals__, 'get_mesh_id', get_mesh_id)
    server._compute_kv_cache({'obs': [], 'state': torch.zeros(3, 1, 2)})
    policy = model.mot.retention_policies['test']
    for attention, old in zip(model.mot.shared_attn, old_layers):
        cache = attention.attn_caches['test']
        assert cache['mask'].sum() == 28  # 14 predicted + 14 newly observed
        assert cache['mask'][old_slots].all() and cache['is_pred'][old_slots].all()
        for name in ('k', 'v'):
            torch.testing.assert_close(cache[name][:, old_slots], old[name], rtol=0, atol=0)
        assert not hasattr(attention, 'clear_pred_cache')
    for name in ('token_uid', 'world_time_id', 'observation_flag', 'dino', 'neoforce'):
        torch.testing.assert_close(policy.data[name][old_slots], before[name], rtol=0, atol=0)
    assert not hasattr(model, 'clear_pred_cache')
    assert policy.t0 == 5


@pytest.mark.parametrize('video_exec_step', [-1, 2])
@pytest.mark.parametrize('fail_dino', [False, True])
def test_real_generation_loop_labels_once_between_experts_and_rolls_back(monkeypatch, video_exec_step, fail_dino):
    from contextlib import contextmanager
    from n0_twam.preprocessing.kv_index import observed_index
    from n0_twam.utils.utils import data_seq_to_patch

    server = dense_server()
    server.job_config.frame_chunk_size = 2
    server.job_config.tactile_keys = []
    server.job_config.server_tactile_denoise = False
    server.job_config.action_dim = 1
    server.job_config.action_delta_mode = 'pi05_delta'
    server.job_config.num_inference_steps = 2
    server.job_config.action_num_inference_steps = 2
    server.job_config.video_exec_step = video_exec_step
    server.job_config.guidance_scale = server.job_config.action_guidance_scale = 1.
    server.vae.config.latents_mean = [.1] * 48
    server.vae.config.latents_std = [.5] * 48
    server.latent_height, server.latent_width = 4, 8
    server.action_per_frame = 1
    server.use_cfg = False
    server.action_mask = torch.ones(1, dtype=torch.bool)
    server.prompt_embeds = torch.ones(1, 2, 4)
    server.init_latent = None
    server.exp_save_root = '/tmp'
    server.tactile_global_vae = SimpleNamespace(clear_cache=lambda: None, feat_cache=[])
    server.tactile_local_vae = SimpleNamespace(clear_cache=lambda: None, feat_cache=[])
    events = []

    class Scheduler:
        def set_timesteps(self, _):
            self.timesteps = torch.tensor([2., 1.])

        def step(self, prediction, t, sample, return_dict=False):
            return sample

    class Transformer:
        def __init__(self):
            self.policy = GlobalKVRetention(64, 'cpu')
            self.mask = torch.zeros(64, dtype=torch.bool)

        @contextmanager
        def cache_transaction(self, name):
            snapshot, mask = self.policy.snapshot(), self.mask.clone()
            try:
                yield
            except BaseException:
                self.policy.restore(snapshot)
                self.mask = mask
                raise

        def __call__(self, data, *, update_cache, cache_name, action_mode=False):
            events.append(('action' if action_mode else 'video', update_cache))
            count = data['grid_id'].shape[-1]
            if update_cache:
                packet = token_rows({'grid_id': data['grid_id'], 'index': data.get('kv_index'),
                                     'actions': data['noisy_latents'] if action_mode else None},
                                    batch_size=1, length=count, main_count=count,
                                    action_mode=action_mode, update_cache=update_cache, device='cpu')
                slots, _ = self.policy.plan(self.mask, count, packet)
                self.policy.commit(slots, packet, self.mask.clone())
                self.mask[slots] = True
            return torch.zeros(1, count if action_mode else count * 4, 1 if action_mode else 48)

        def global_cache_cursor(self, name):
            return self.policy.next_uid

        def video_index_handle(self, name, cursor):
            return self.policy.video_handle(self.mask, cursor)

        def annotate_video_dino(self, name, handle, features):
            events.append(('label', 0))
            return self.policy.annotate_video_dino(self.mask, handle, features)

    def encode_obs(obs):
        server._current_observed_kv_index = observed_index({'dino': torch.ones(8, 1)}, 8, 'cpu')
        return torch.zeros(1, 48, 1, 4, 8)

    server._encode_obs = encode_obs
    server.scheduler = Scheduler()
    server.action_scheduler = Scheduler()
    server.transformer = Transformer()
    server.postprocess_action = lambda value, **kwargs: value
    monkeypatch.setitem(TWAMServer._infer_impl.__globals__, 'get_mesh_id', get_mesh_id)
    monkeypatch.setitem(TWAMServer._infer_impl.__globals__, 'data_seq_to_patch', data_seq_to_patch)
    if fail_dino:
        def fail(rgb):
            raise RuntimeError('DINO failed')
        server._kv_dino_encoder = fail
        with pytest.raises(RuntimeError, match='DINO failed'):
            server._infer({}, frame_st_id=0)
        assert not server.transformer.mask.any()
        assert server.transformer.policy.next_uid == 0
        assert server.transformer.policy.t0 is None
        assert server.init_latent is None and server._init_kv_index is None
        assert all(event[0] == 'video' for event in events)
    else:
        server._infer({}, frame_st_id=0)
        video_rounds = 3 if video_exec_step == -1 else 2
        assert events == [('video', 0)] * (video_rounds - 1) + [('video', 1), ('label', 0),
                                                             ('action', 0), ('action', 0), ('action', 1)]
        assert len(server.vae.inputs) == 1  # no decode/encode feedback per denoising step
        assert server.transformer.policy.t0 == 0
