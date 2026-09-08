# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import torch
from easydict import EasyDict

twam_shared_cfg = EasyDict()

twam_shared_cfg.host = '0.0.0.0'
twam_shared_cfg.port = 29536

twam_shared_cfg.param_dtype = torch.bfloat16
twam_shared_cfg.save_root = './train_out'

twam_shared_cfg.patch_size = (1, 2, 2)

# Full RGB + independent index metadata + one retention budget for all experts.
# Set "fifo" to restore the legacy serving cache (also required by old sparse
# RGB-motion experiments). No new model weights or training losses are added.
twam_shared_cfg.kv_cache_policy = 'global'
twam_shared_cfg.kv_retention = dict(
    top_k=128, time_scale=8.0, contact_weight=1.0, visual_weight=1.0,
    time_weight=1.0, query_weight=1.0, repetition_weight=1.0,
    action_scale=0.1, query_samples=16, seed=0,
)
# Already-cached/local DINO only. Alternatively provide obs['kv_index']['dino'].
# To run without DINO, disable BOTH online observation and predicted DINO.
twam_shared_cfg.kv_index_dino_online = True
twam_shared_cfg.kv_index_dino_model_name_or_path = 'facebook/dinov2-base'
twam_shared_cfg.kv_index_dino_device = 'cpu'
twam_shared_cfg.kv_index_dino_image_size = (224, 224)
# After the FINAL video forward only: decode for DINO labels, never feed the
# predicted RGB/latents into another world-model forward. Adds VAE+DINO cost.
twam_shared_cfg.kv_index_predicted_dino = True

# Optional sparse RGB-motion sidecars.  Kept off by default so released
# checkpoints and existing latent datasets retain their exact input contract.
# When enabled, each latent segment must have a matching file under
#   <dataset>/<rgb_motion_root_name>/chunk-XXX/episode_XXXXXX_START_END.pth
twam_shared_cfg.use_rgb_motion_tokens = False
twam_shared_cfg.rgb_motion_root_name = 'rgb_motion'
# Fixed per-frame K keeps default DataLoader collation valid across segments.
# Set to 0 only when every sidecar already uses the same padded K (or batch=1).
twam_shared_cfg.rgb_motion_max_tokens = 32
twam_shared_cfg.rgb_motion_require_index = True

# Separate opt-in for the online RGB-D -> RGB-motion producer.  It requires
# ``use_rgb_motion_tokens=True``; keeping it false lets clients use precomputed
# ``obs['rgb_motion']`` payloads without loading a second vision backbone.
twam_shared_cfg.rgb_motion_online_preprocess = False
# DINOv2-B/14 may be either a local directory or an already-cached HF model id.
# The server always calls from_pretrained(..., local_files_only=True): it never
# downloads weights implicitly while serving.
twam_shared_cfg.rgb_motion_dino_model_name_or_path = 'facebook/dinov2-base'
twam_shared_cfg.rgb_motion_dino_device = 'cpu'
twam_shared_cfg.rgb_motion_dino_image_size = (224, 224)
twam_shared_cfg.rgb_motion_dino_float_input_range = '0_1'
twam_shared_cfg.rgb_motion_first_frame_policy = 'require_previous'
twam_shared_cfg.rgb_motion_depth_threshold = 0.02
twam_shared_cfg.rgb_motion_dino_threshold = 0.2
twam_shared_cfg.rgb_motion_depth_weight = 1.0
twam_shared_cfg.rgb_motion_dino_weight = 1.0
twam_shared_cfg.rgb_motion_dilation_radius = 1
twam_shared_cfg.rgb_motion_min_depth = 1e-6

twam_shared_cfg.enable_offload = True

twam_shared_cfg.tactile_keys = []
twam_shared_cfg.max_tactile_streams = 4
twam_shared_cfg.tactile_height = 64
twam_shared_cfg.tactile_width = 64
twam_shared_cfg.synthetic_tactile_data = False

# ───── New (latent-tactile pipeline) ─────
# Shared by the global and local tactile pathways
twam_shared_cfg.tactile_latent_root_name = 'latents_tactile'   # dir under dataset root
twam_shared_cfg.tactile_sensor_id_map = {                       # tactile_key → sensor_id
    # single-arm default — override per-dataset cfg if more sensors
    'observation.images.tactile_a': 0,
    'observation.images.tactile_b': 1,
    # Reserved IDs for future bimanual setups:
    # 'observation.images.tactile_ll': 0, 'observation.images.tactile_lr': 1,
    # 'observation.images.tactile_rl': 2, 'observation.images.tactile_rr': 3,
}
twam_shared_cfg.tactile_latent_height = 8      # 128 / 16 (Wan VAE 16x spatial compress)
twam_shared_cfg.tactile_latent_width = 8
twam_shared_cfg.tactile_cfg_prob = 0.1          # prob to drop tactile (CFG dropout)
