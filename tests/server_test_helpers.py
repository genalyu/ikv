from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import pytest
import torch

def _load_server_class():
    module_path = Path(__file__).parents[1] / 'n0_twam' / 'n0_twam_server.py'
    configs = types.ModuleType('configs')
    configs.TWAM_CONFIGS = {}
    distributed = types.ModuleType('distributed')
    distributed.__path__ = []
    distributed_fsdp = types.ModuleType('distributed.fsdp')
    distributed_fsdp.shard_model = lambda *args, **kwargs: None
    distributed_util = types.ModuleType('distributed.util')
    distributed_util._configure_model = lambda model, **kwargs: model
    distributed_util.init_distributed = lambda *args, **kwargs: None
    models = types.ModuleType('models')
    models.__path__ = []
    models_utils = types.ModuleType('models.utils')
    for name in ('WanVAEStreamingWrapper', 'load_text_encoder', 'load_tokenizer', 'load_transformer', 'load_vae'):
        setattr(models_utils, name, object)
    utils = types.ModuleType('utils')
    utils.FlowMatchScheduler = object
    utils.data_seq_to_patch = lambda *args, **kwargs: None
    utils.get_mesh_id = lambda *args, **kwargs: torch.empty(4, 0)
    utils.init_logger = lambda *args, **kwargs: None
    utils.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    utils.run_async_server_mode = lambda *args, **kwargs: None
    utils.save_async = lambda *args, **kwargs: None
    diffusers = types.ModuleType('diffusers')
    diffusers.__path__ = []
    diffusers_video = types.ModuleType('diffusers.video_processor')
    diffusers_video.VideoProcessor = object
    diffusers_utils = types.ModuleType('diffusers.utils')
    diffusers_utils.export_to_video = lambda *args, **kwargs: None
    diffusers_pipelines = types.ModuleType('diffusers.pipelines')
    diffusers_pipelines.__path__ = []
    diffusers_wan = types.ModuleType('diffusers.pipelines.wan')
    diffusers_wan.__path__ = []
    diffusers_pipeline_wan = types.ModuleType('diffusers.pipelines.wan.pipeline_wan')
    diffusers_pipeline_wan.prompt_clean = lambda value: value
    stubs = {'configs': configs, 'distributed': distributed, 'distributed.fsdp': distributed_fsdp, 'distributed.util': distributed_util, 'models': models, 'models.utils': models_utils, 'utils': utils, 'diffusers': diffusers, 'diffusers.video_processor': diffusers_video, 'diffusers.utils': diffusers_utils, 'diffusers.pipelines': diffusers_pipelines, 'diffusers.pipelines.wan': diffusers_wan, 'diffusers.pipelines.wan.pipeline_wan': diffusers_pipeline_wan}
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location('_server_under_test', module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.TWAM_Server
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
TWAMServer = _load_server_class()

class _TestStreamingVAE:

    def __init__(self, *, temporal_stride=4, warm=False):
        self.vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=temporal_stride))
        self.feat_cache = [torch.tensor([1.0]) if warm else None]

    def clear_cache(self):
        self.feat_cache = [None]

def _server(*, enabled: bool=True, patch_size=(1, 2, 2), online: bool=False, camera_keys=('camera',), first_frame_policy='require_previous'):
    server = TWAMServer.__new__(TWAMServer)
    server.device = torch.device('cpu')
    server.dtype = torch.float32
    server.job_config = SimpleNamespace(patch_size=patch_size, obs_cam_keys=list(camera_keys), height=64, width=64)
    server.streaming_vae = _TestStreamingVAE()
    server.vae = server.streaming_vae.vae
    return server

def _consistency_server(tmp_path, *, enabled, live_cameras, metadata, patch_size=(1, 2, 2)):
    bundle = tmp_path / 'bundle'
    (bundle / 'transformer').mkdir(parents=True)
    (bundle / 'train_meta.json').write_text(json.dumps(metadata))
    server = TWAMServer.__new__(TWAMServer)
    server.job_config = SimpleNamespace(wan22_pretrained_model_name_or_path=str(bundle), norm_stat={}, obs_cam_keys=live_cameras, patch_size=patch_size)
    return server

class _CacheTransformer:

    def __init__(self):
        self.clear_calls = []
        self.forward_calls = []

    def clear_pred_cache(self, name):
        self.clear_calls.append(name)

    def __call__(self, input_dict, **kwargs):
        self.forward_calls.append((input_dict, kwargs))
        return torch.empty(0)

class _FailingActionCacheTransformer(_CacheTransformer):
    """Tiny transactional cache double for the server's two-pass grounding."""

    def __init__(self):
        super().__init__()
        self.committed = ['old-prediction']
        self.transaction_events = []

    def clear_pred_cache(self, name):
        super().clear_pred_cache(name)
        self.committed.clear()

    def cache_transaction(self, name):
        owner = self

        class _Transaction:

            def __enter__(self):
                self.snapshot = list(owner.committed)
                owner.transaction_events.append(('begin', name))
                return self

            def __exit__(self, exc_type, _exc, _traceback):
                if exc_type is None:
                    owner.transaction_events.append(('commit', name))
                else:
                    owner.committed[:] = self.snapshot
                    owner.transaction_events.append(('rollback', name))
                return False
        return _Transaction()

    def __call__(self, input_dict, **kwargs):
        super().__call__(input_dict, **kwargs)
        branch = input_dict['branch']
        self.committed.append(branch)
        if kwargs['action_mode']:
            raise RuntimeError('forced action grounding failure')
        return torch.empty(0)

def _kv_lifecycle_server(*, sparse: bool):
    server = _server(enabled=sparse)
    server.cache_name = 'pos'
    server.frame_st_id = 0
    server.exp_save_root = '/tmp'
    server.init_latent = torch.full((1, 1, 1, 2, 2), -5.0)
    server.transformer = _CacheTransformer()
    server._last_gen_tactile = None
    server._last_gen_tactile_fsid = None
    server._encode_obs = lambda obs: torch.stack([torch.full((1, 2, 2), 10.0), torch.full((1, 2, 2), 20.0)], dim=1).unsqueeze(0)
    server.preprocess_action = lambda *args, **kwargs: torch.arange(3, dtype=torch.float32).reshape(1, 1, 3, 1, 1).expand(1, 2, -1, -1, -1).clone()
    server._encode_tactile_obs = lambda obs: None
    server._repeat_input_for_cfg = lambda value: value
    captured = {}

    def _prepare(latent, action, **kwargs):
        captured.update(latent=latent, action=action, kwargs=kwargs)
        return {'latent_res_lst': {'branch': 'video'}, 'action_res_lst': {'branch': 'action'}}
    server._prepare_latent_input = _prepare
    return (server, captured)
