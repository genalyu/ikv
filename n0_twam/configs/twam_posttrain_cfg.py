# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Post-training config (registered as "posttrain") — the validated recipe.

Fine-tunes the released N0-TWAM checkpoint on ONE task pool: MoT narrow experts,
LocalTactile ON in "current" mode (flip-on over the local-off release ckpt),
horizon 12 x 12 actions/frame. _ACTION_MODE selects "absee" (absolute EE,
default) or "delta" (delta EE); each mode has its own norm file and they must never
overwrite each other. All tactile-emptying drop probs stay 0 (the MoT tactile
expert cannot take an empty token sequence).

Pool layout (script/build_task_pool.py):
  <POOL>/{train,val}/<repo>  +  norm_stat_absee.json / norm_stat.json

Edit the EDIT-ME block, then `NGPU=8 bash run_posttrain.sh`.
"""
import json
from pathlib import Path
from easydict import EasyDict
from .twam_base_cfg import twam_base_cfg
_POOL = Path('/path/to/pools/my_task')
_BASE_MODEL = '/path/to/base-model'
_RELEASED_CKPT = '/path/to/n0-twam-checkpoint'
_EMPTY_EMB = '/path/to/base-model/empty_emb.pt'
_SAVE_ROOT = str(_POOL / 'runs' / 'posttrain')
_PROMPT = 'your task instruction, verbatim from meta/tasks.jsonl'
_CAM_KEYS = ['observation.images.top', 'observation.images.wrist_l']
_TACTILE_KEYS = ['observation.images.tactile_a', 'observation.images.tactile_b']
_USED_ACTION_CHANNELS = list(range(10))
_ACTION_MODE = 'absee'
_USE_LOCAL_TACTILE = True
_TACTILE_GLOBAL_ZERO = False
_USE_IKV = False
_IKV_VIDEO_CAPACITY = 2816
_IKV_ACTION_CAPACITY = 512
_IKV_TACTILE_CAPACITY = 768
_IKV_CAPACITY = _IKV_VIDEO_CAPACITY + _IKV_ACTION_CAPACITY + _IKV_TACTILE_CAPACITY
_IKV_INDEX_ROOT = 'ikv_index'
assert _ACTION_MODE in ('absee', 'delta'), _ACTION_MODE
cfg = EasyDict(twam_base_cfg.copy())
cfg.__name__ = f'Config: N0-TWAM post-train ({_ACTION_MODE}, MoT)'
cfg.use_ikv_training = _USE_IKV
cfg.ikv_train_execution = 'masked' if _USE_IKV else 'recurrent'
cfg.ikv_train_sample_capacity = bool(_USE_IKV)
cfg.ikv_train_capacity = _IKV_CAPACITY
cfg.ikv_train_min_capacity = (_IKV_CAPACITY + 1) // 2 if _USE_IKV else 0
cfg.ikv_index_root_name = _IKV_INDEX_ROOT
cfg.kv_retention = dict(cfg.kv_retention)
cfg.kv_retention.update(version=2, video_capacity=_IKV_VIDEO_CAPACITY, action_capacity=_IKV_ACTION_CAPACITY, tactile_capacity=_IKV_TACTILE_CAPACITY, query_weight=0.0, action_query_weight=0.0, tactile_query_weight=0.0, visual_weight=0.0, persistence_weight=0.0, persistence_scale=8.0, class_recency_weight=1.0, class_recency_scale=8.0, contact_scale=8.0, content_threshold=0.75, content_capacity=2048)
cfg.kv_semantic_encoder = {}
cfg.kv_semantic_provenance = None
cfg.kv_retention.setdefault('task_weight', 0.0)
cfg.kv_cache_policy = 'global' if _USE_IKV else 'fifo'
cfg.dataset_path = str(_POOL / 'train')
cfg.val_dataset_path = str(_POOL / 'val')
cfg.val_interval = 9999
cfg.obs_cam_keys = list(_CAM_KEYS)
cfg.tactile_keys = list(_TACTILE_KEYS)
cfg.per_repo_obs_cam_keys = {}
cfg.per_repo_tactile_keys = {}
cfg.tactile_optional = False
cfg.synthetic_tactile_data = False
cfg.tactile_sensor_id_map = {k: i for i, k in enumerate(_TACTILE_KEYS)}
cfg.max_tactile_streams = 4
cfg.use_local_tactile = bool(_USE_LOCAL_TACTILE)
cfg.local_tactile_mode = 'current'
cfg.tactile_global_zero = bool(_TACTILE_GLOBAL_ZERO)
cfg.action_dim = 20
cfg.action_delta_mode = 'none' if _ACTION_MODE == 'absee' else 'pi05_delta'
cfg.pi05_action_horizon = 12
cfg.action_per_frame = 12
cfg.pi05_source_action_is_delta = False
cfg.pi05_state_column = 'observation.state'
cfg.pi05_delta_channel_ids = list(range(0, 9)) + list(range(10, 19))
cfg.pi05_rot6d_relative_delta = False
cfg.pi05_condition_first_frame_zero = True
cfg.pi05_condition_first_frame_loss = False
cfg.pi05_invalid_horizon_mask = True
cfg.used_action_channel_ids = list(_USED_ACTION_CHANNELS)
cfg.per_repo_used_action_channel_ids = {}
_inverse = [len(cfg.used_action_channel_ids)] * cfg.action_dim
for _i, _j in enumerate(cfg.used_action_channel_ids):
    _inverse[_j] = _i
cfg.inverse_used_action_channel_ids = _inverse
cfg.action_norm_method = 'q01q99'
_NORM_NAME = 'norm_stat_absee' if _ACTION_MODE == 'absee' else 'norm_stat'
_NORM = _POOL / f'{_NORM_NAME}.json'
if _NORM.is_file():
    cfg.norm_stat = json.loads(_NORM.read_text())
    cfg.norm_stat_path = str(_NORM)
else:
    cfg.norm_stat = {'q01': [-1.0] * 20, 'q99': [1.0] * 20}
    cfg.norm_stat_path = str(_NORM) + ' (MISSING)'
_PER_ROBOT = _POOL / f'{_NORM_NAME}_per_robot.json'
cfg.per_repo_norm_stat = json.loads(_PER_ROBOT.read_text()) if _PER_ROBOT.is_file() else {}
cfg.wan22_pretrained_model_name_or_path = _BASE_MODEL
cfg.empty_emb_path = _EMPTY_EMB
cfg.resume_from = _RELEASED_CKPT
cfg.save_root = _SAVE_ROOT
cfg.eval_prompt = _PROMPT
cfg.use_mot = True
cfg.mot_cross_attn_experts = ('video', 'action')
cfg.mot_warmstart_experts = ('video', 'action', 'tactile')
cfg.mot_expert_hidden_dim = {'action': 1024, 'tactile': 1024}
cfg.mot_expert_ffn_dim = {'action': 4096, 'tactile': 4096}
cfg.use_contact_gate = False
cfg.tactile_cfg_prob = 0.0
cfg.cfg_prob = 0.0
cfg.noisy_cond_prob_tactile = 0.0
cfg.enable_wandb = False
cfg.load_worker = 4
cfg.num_init_worker = 1
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 4
cfg.num_steps = 2000
cfg.save_interval = 500
cfg.gc_interval = 50
cfg.learning_rate = 0.0001
cfg.lr_schedule = 'cosine'
cfg.lr_min_ratio = 0.1
cfg.warmup_steps = 20
assert cfg.local_tactile_mode == 'current'
assert cfg.action_delta_mode in ('none', 'pi05_delta')
assert cfg.pi05_action_horizon == 12 and cfg.action_per_frame == 12
assert cfg.tactile_cfg_prob == 0.0 and cfg.cfg_prob == 0.0
assert cfg.noisy_cond_prob_tactile == 0.0
assert len(cfg.norm_stat['q01']) == 20 and len(cfg.norm_stat['q99']) == 20
assert cfg.action_norm_method == 'q01q99'
assert cfg.use_local_tactile or not cfg.tactile_global_zero, 'local-off + global-zero removes the whole tactile pathway'
twam_posttrain_cfg = cfg
