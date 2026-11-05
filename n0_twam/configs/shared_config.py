# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import torch
from easydict import EasyDict
twam_shared_cfg = EasyDict()
twam_shared_cfg.host = '0.0.0.0'
twam_shared_cfg.port = 29536
twam_shared_cfg.param_dtype = torch.bfloat16
twam_shared_cfg.save_root = './train_out'
twam_shared_cfg.patch_size = (1, 2, 2)
twam_shared_cfg.kv_cache_policy = 'global'
twam_shared_cfg.kv_retention = dict(top_k=128, time_scale=8.0, contact_weight=1.0, visual_weight=1.0, time_weight=1.0, query_weight=1.0, repetition_weight=1.0, action_scale=0.1, query_samples=16, seed=0)
twam_shared_cfg.kv_index_dino_online = True
twam_shared_cfg.kv_index_dino_model_name_or_path = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
twam_shared_cfg.kv_index_dino_device = 'cpu'
twam_shared_cfg.kv_index_dino_image_size = (224, 224)
twam_shared_cfg.kv_index_predicted_dino = True
twam_shared_cfg.enable_offload = True
twam_shared_cfg.tactile_keys = []
twam_shared_cfg.max_tactile_streams = 4
twam_shared_cfg.tactile_height = 64
twam_shared_cfg.tactile_width = 64
twam_shared_cfg.synthetic_tactile_data = False
twam_shared_cfg.tactile_latent_root_name = 'latents_tactile'
twam_shared_cfg.tactile_sensor_id_map = {'observation.images.tactile_a': 0, 'observation.images.tactile_b': 1}
twam_shared_cfg.tactile_latent_height = 8
twam_shared_cfg.tactile_latent_width = 8
twam_shared_cfg.tactile_cfg_prob = 0.1
twam_shared_cfg.use_ikv_training = False
twam_shared_cfg.ikv_train_capacity = 4096
twam_shared_cfg.ikv_index_root_name = None
