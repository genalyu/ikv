from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch
from einops import rearrange
import pytest
import torch
import torch.nn.functional as F
from n0_twam.models.model import WanTransformer3DModel
from n0_twam.utils.utils import data_seq_to_patch

def _load_trainer_class():
    """Load Trainer without importing optional LeRobot training dependencies."""
    module_path = Path(__file__).parents[1] / 'n0_twam' / 'train.py'
    configs = types.ModuleType('configs')
    configs.TWAM_CONFIGS = {}
    distributed = types.ModuleType('distributed')
    distributed.__path__ = []
    distributed_fsdp = types.ModuleType('distributed.fsdp')
    distributed_fsdp.shard_model = lambda *args, **kwargs: None
    distributed_fsdp.apply_ac = lambda *args, **kwargs: None
    distributed_util = types.ModuleType('distributed.util')
    distributed_util._configure_model = lambda model, **kwargs: model
    distributed_util.init_distributed = lambda *args, **kwargs: None
    distributed_util.dist_mean = lambda value: value
    distributed_util.dist_max = lambda value: value
    distributed_util.dist_mean_and_max = lambda value: (value.clone(), value.clone())
    models = types.ModuleType('models')
    models.__path__ = []
    models_utils = types.ModuleType('models.utils')
    models_utils.load_transformer = lambda *args, **kwargs: None
    models_utils.load_mot_transformer = lambda *args, **kwargs: None
    models_utils.load_mot_checkpoint = lambda *args, **kwargs: None
    utils = types.ModuleType('utils')
    utils.init_logger = lambda *args, **kwargs: None
    utils.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    utils.get_mesh_id = lambda *args, **kwargs: None
    utils.sample_timestep_id = lambda *args, **kwargs: None
    utils.data_seq_to_patch = data_seq_to_patch
    utils.warmup_constant_lambda = lambda *args, **kwargs: 1.0
    utils.FlowMatchScheduler = object
    dataset = types.ModuleType('dataset')
    dataset.__path__ = []
    bucket_sampler = types.ModuleType('dataset.bucket_sampler')
    bucket_sampler.WithinUpdateDistributedSampler = object
    dataset.MultiLatentLeRobotDataset = object
    dataset.BucketedDistributedBatchSampler = object
    wandb = types.ModuleType('wandb')
    wandb.login = lambda *args, **kwargs: None
    wandb.init = lambda *args, **kwargs: None
    stubs = {'configs': configs, 'distributed': distributed, 'distributed.fsdp': distributed_fsdp, 'distributed.util': distributed_util, 'models': models, 'models.utils': models_utils, 'utils': utils, 'dataset': dataset, 'dataset.bucket_sampler': bucket_sampler, 'wandb': wandb}
    spec = importlib.util.spec_from_file_location('_trainer_under_test', module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.Trainer
Trainer = _load_trainer_class()

class _StubScheduler:

    @staticmethod
    def training_weight(timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() + 0.5
