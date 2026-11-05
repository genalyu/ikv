# Copyright 2025-2026 NeoteAI Team. All rights reserved.
import argparse
from contextlib import nullcontext
import os
import sys
import time
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import TWAM_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from models.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_transformer, load_vae
from utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger, logger, run_async_server_mode, save_async

class TWAM_Server:

    def __init__(self, job_config):
        from n0_twam.input_contract import validate_dense_rgb_config
        validate_dense_rgb_config(job_config)
        self.cache_name = 'pos'
        self.frame_st_id = 0
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f'cuda:{job_config.local_rank}')
        self.enable_offload = getattr(job_config, 'enable_offload', True)
        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(shift=self.job_config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)
        self.vae = load_vae(os.path.join(job_config.wan22_pretrained_model_name_or_path, 'vae'), torch_dtype=self.dtype, torch_device='cpu' if self.enable_offload else self.device)
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_global_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_local_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_first_frames = None
        self.tactile_prev_frames = None
        self._last_gen_tactile = None
        self._last_gen_tactile_fsid = None
        self._last_observed_video_latent = None
        self.tokenizer = load_tokenizer(os.path.join(job_config.wan22_pretrained_model_name_or_path, 'tokenizer'))
        self.text_encoder = load_text_encoder(os.path.join(job_config.wan22_pretrained_model_name_or_path, 'text_encoder'), torch_dtype=self.dtype, torch_device='cpu' if self.enable_offload else self.device)
        _tpath = os.path.join(job_config.wan22_pretrained_model_name_or_path, 'transformer')
        _is_mot = False
        try:
            import json as _json
            _is_mot = _json.load(open(os.path.join(_tpath, 'config.json'))).get('is_mot', False)
        except Exception:
            _is_mot = False
        if _is_mot:
            from models.utils import load_mot_checkpoint
            self.transformer = load_mot_checkpoint(_tpath, torch_dtype=self.dtype, torch_device=self.device, attn_mode='torch', config_overrides={})
        else:
            self.transformer = load_transformer(_tpath, torch_dtype=self.dtype, torch_device=self.device, max_tactile_streams=job_config.max_tactile_streams, target_action_dim=int(getattr(job_config, 'action_dim', 30)), attn_mode=getattr(job_config, 'attn_mode', 'flashattn'))
        logger.info('loaded transformer: %s (is_mot=%s) from %s', type(self.transformer).__name__, _is_mot, _tpath)
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer, shard_fn=shard_fn, param_dtype=self.dtype, device=self.device, eval_mode=True)
        self._check_train_serve_consistency()

    def _check_train_serve_consistency(self):
        """Refuse placeholder norm stats, and cross-check the checkpoint's
        train_meta.json (when present) against the live serve config.

        Multi-task checkpoints (``serve_task`` set, see multitask_server): the
        snapshot's norm_stat is the pool-level fallback envelope no task
        actually trained with, so the live norm is checked against the pool's
        per-task table instead, and ``used_action_channel_ids`` — the training
        union vs the served task's subset — is checked for containment."""
        import json
        from pathlib import Path
        serve_task = getattr(self.job_config, 'serve_task', None)
        ns = getattr(self.job_config, 'norm_stat', None) or {}
        q01 = [float(v) for v in ns.get('q01', [])]
        q99 = [float(v) for v in ns.get('q99', [])]
        if q01 and q01 == [-1.0] * len(q01) and (q99 == [1.0] * len(q99)):
            raise RuntimeError(f'norm_stat is the [-1, 1] placeholder — the stats file was missing when the config was imported (norm_stat_path={getattr(self.job_config, 'norm_stat_path', None)!r}). Compute the norm stats and point the config at them before serving.')
        problems = []
        if serve_task:
            per_path = Path(getattr(self.job_config, 'multitask_norm_path', ''))
            if not per_path.is_file():
                raise RuntimeError(f'[consistency] serve_task={serve_task!r} but the per-task norm table is missing: {per_path}')
            per = json.loads(per_path.read_text()).get(serve_task)
            if per is None:
                raise RuntimeError(f'[consistency] task {serve_task!r} not in {per_path}')
            for key in ('q01', 'q99'):
                want = np.asarray(per.get(key, []), dtype=np.float64)
                live = np.asarray(ns.get(key, []), dtype=np.float64)
                if want.shape != live.shape or not np.allclose(want, live, atol=1e-06):
                    problems.append(f'norm_stat.{key} differs from the per-task table ({per_path.name}[{serve_task}])')
        meta_path = (Path(self.job_config.wan22_pretrained_model_name_or_path) / 'transformer').resolve().parent / 'train_meta.json'
        if not meta_path.is_file():
            if problems:
                raise RuntimeError('[consistency] serve config does not match the per-task norm table:\n  ' + '\n  '.join(problems))
            logger.info('[consistency] no train_meta.json next to the checkpoint (%s) — skipping the training-snapshot cross-check', meta_path)
            if serve_task:
                logger.info('[consistency] multi-task per-task norm check passed (task=%s)', serve_task)
            return
        meta = json.loads(meta_path.read_text())
        if not serve_task:
            meta_ns = meta.get('norm_stat') or {}
            for key in ('q01', 'q99'):
                trained = np.asarray(meta_ns.get(key, []), dtype=np.float64)
                live = np.asarray(ns.get(key, []), dtype=np.float64)
                if trained.shape != live.shape or not np.allclose(trained, live, atol=1e-06):
                    problems.append(f'norm_stat.{key} differs from training')
        for key in ('action_norm_method', 'action_delta_mode', 'action_dim', 'action_per_frame', 'pi05_action_horizon', 'used_action_channel_ids', 'use_local_tactile', 'local_tactile_mode', 'tactile_global_zero', 'patch_size'):
            if key not in meta:
                continue
            live = getattr(self.job_config, key, None)
            if isinstance(meta[key], bool):
                live = bool(live)
            elif isinstance(meta[key], list):
                live = [int(v) for v in live or []]
                meta[key] = [int(v) for v in meta[key]]
            if serve_task and key == 'used_action_channel_ids':
                if not set(live) <= set(meta[key]):
                    problems.append(f'{key}: serve {live!r} is not a subset of train {meta[key]!r}')
                continue
            if meta[key] != live:
                problems.append(f'{key}: train={meta[key]!r} serve={live!r}')
        if bool(meta.get('use_ikv_training', False)):
            from n0_twam.models.global_kv_retention import RetentionConfig
            if not bool(getattr(self.job_config, 'use_ikv_training', False)):
                problems.append('use_ikv_training must match the checkpoint')
            if getattr(self.job_config, 'kv_cache_policy', 'fifo') != 'global':
                problems.append('IKV-trained checkpoint requires global retention')
            if int(getattr(self.job_config, 'ikv_train_capacity', 0)) != meta['ikv_train_capacity']:
                problems.append('ikv_train_capacity differs from training')
            if RetentionConfig(**dict(getattr(self.job_config, 'kv_retention', {}))) != RetentionConfig(**meta['kv_retention']):
                problems.append('kv_retention differs from training')
        trained_semantic = meta.get('kv_semantic_provenance')
        if trained_semantic is not None:
            if trained_semantic != getattr(self.job_config, 'kv_semantic_provenance', None):
                problems.append('semantic backend/assets/preprocessing differ from training')
        if problems:
            raise RuntimeError(f'[consistency] serve config does not match the training snapshot {meta_path}:\n  ' + '\n  '.join(problems))
        logger.info('[consistency] train_meta.json cross-check passed (%s%s)', meta_path, f'; multi-task norm checked per-task ({serve_task})' if serve_task else '')

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(self, prompt, negative_prompt=None, do_classifier_free_guidance=True, num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None, max_sequence_length=226, device=None, dtype=None):
        device = device or self.device
        dtype = dtype or self.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(prompt=prompt, num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype)
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.')
            elif batch_size != len(negative_prompt):
                raise ValueError(f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.')
            negative_prompt_embeds = self._get_t5_prompt_embeds(prompt=negative_prompt, num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype)
        return (prompt_embeds, negative_prompt_embeds)

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def _uses_pi05_delta_actions(self):
        action_delta_mode = str(getattr(self.job_config, 'action_delta_mode', '')).lower()
        return action_delta_mode in {'pi05_delta', 'openpi_delta', 'pi0.5_delta'}

    def _pi05_delta_channel_ids(self):
        action_dim = int(self.job_config.action_dim)
        delta_ids = getattr(self.job_config, 'pi05_delta_channel_ids',
                            list(range(0, 9)) + list(range(10, 19)))
        return [int(v) for v in delta_ids if 0 <= int(v) < action_dim]

    def _pi05_required_state_dim(self):
        delta_ids = self._pi05_delta_channel_ids()
        return max(delta_ids) + 1 if delta_ids else 1

    def _pad_state_vector(self, state_vec, field_name):
        state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        required_dim = self._pi05_required_state_dim()
        action_dim = int(self.job_config.action_dim)
        if state_vec.shape[0] < required_dim:
            raise ValueError(
                f"{field_name} must contain at least {required_dim} dims for "
                f"pi05_delta channels {self._pi05_delta_channel_ids()}, got "
                f"shape {state_vec.shape}."
            )
        if state_vec.shape[0] < action_dim:
            state_vec = np.pad(state_vec, (0, action_dim - state_vec.shape[0]))
        return state_vec[:action_dim].astype(np.float32, copy=False)

    def _extract_current_state_vector(self, state, field_name='current_state'):
        if state is None:
            raise ValueError(
                f"pi05_delta requires obs['{field_name}'] to convert between "
                "model deltas and absolute action targets."
            )
        state_arr = np.asarray(state, dtype=np.float32)
        action_dim = int(self.job_config.action_dim)
        required_dim = self._pi05_required_state_dim()
        if state_arr.ndim == 1:
            state_vec = state_arr
        elif state_arr.ndim == 2:
            if required_dim <= state_arr.shape[0] <= action_dim:
                state_vec = state_arr[:, -1]
            elif required_dim <= state_arr.shape[1] <= action_dim:
                state_vec = state_arr[-1, :]
            elif state_arr.shape[0] >= action_dim:
                state_vec = state_arr[:action_dim, -1]
            elif state_arr.shape[1] >= action_dim:
                state_vec = state_arr[-1, :action_dim]
            else:
                raise ValueError(
                    f"Unsupported {field_name} shape for pi05_delta: "
                    f"{state_arr.shape}"
                )
        elif state_arr.ndim >= 3:
            if required_dim <= state_arr.shape[0] <= action_dim:
                # Channel-first [C, F, H]. Use newest frame and first horizon slot.
                state_vec = state_arr[:, -1, 0]
            elif state_arr.shape[0] >= action_dim:
                state_vec = state_arr[:action_dim, -1, 0]
            elif state_arr.shape[-1] >= required_dim:
                state_vec = state_arr.reshape(-1, state_arr.shape[-1])[-1]
            else:
                raise ValueError(
                    f"Unsupported {field_name} shape for pi05_delta: "
                    f"{state_arr.shape}"
                )
        else:
            raise ValueError(
                f"Unsupported {field_name} shape for pi05_delta: {state_arr.shape}"
            )
        return self._pad_state_vector(state_vec, field_name)

    def _canonical_action_chunk(self, action, field_name='state'):
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim != 3:
            raise ValueError(
                f"{field_name} must have shape [C, F, H], got {action_arr.shape}"
            )
        action_dim = int(self.job_config.action_dim)
        if action_arr.shape[0] == action_dim:
            return action_arr.astype(np.float32, copy=True)
        if action_arr.shape[0] > action_dim:
            raise ValueError(
                f"{field_name} has {action_arr.shape[0]} channels, expected "
                f"<= action_dim={action_dim}."
            )

        padded = np.zeros(
            (action_dim, action_arr.shape[1], action_arr.shape[2]),
            dtype=np.float32,
        )
        used_ids = list(getattr(self.job_config, 'used_action_channel_ids', []))
        if len(used_ids) == action_arr.shape[0]:
            for src_i, dst_i in enumerate(used_ids):
                dst_i = int(dst_i)
                if 0 <= dst_i < action_dim:
                    padded[dst_i] = action_arr[src_i]
        else:
            padded[:action_arr.shape[0]] = action_arr
        return padded

    def preprocess_action(self, action, action_anchor_state=None, action_format=None, cold_first_frame=False):
        action_model_input_np = self._canonical_action_chunk(action)
        if self._uses_pi05_delta_actions():
            action_format = str(True).lower()
            if action_format in {'absolute', 'absolute_target', 'target'}:
                anchor_state = self._extract_current_state_vector(action_anchor_state, field_name='action_anchor_state')
                for dim in self._pi05_delta_channel_ids():
                    frame_anchors = [float(anchor_state[dim])]
                    for f in range(1, action_model_input_np.shape[1]):
                        frame_anchors.append(float(action_model_input_np[dim, f - 1, -1]))
                    for f in range(action_model_input_np.shape[1]):
                        action_model_input_np[dim, f] -= frame_anchors[f]
            elif not action_format in {'pi05_delta', 'openpi_delta', 'pi0.5_delta', 'delta'}:
                raise ValueError(f'Unsupported state_action_format for pi05_delta: {action_format!r}')
        action_model_input = torch.from_numpy(action_model_input_np)
        CA, FA, HA = action_model_input.shape
        action_model_input_paded = F.pad(action_model_input, [0, 0, 0, 0, 0, 1], mode='constant', value=0)
        action_model_input = action_model_input_paded[self.job_config.inverse_used_action_channel_ids]
        if str(self.action_norm_method).lower() in {'quantiles', 'q01q99'}:
            action_model_input = (action_model_input - self.actions_q01) / (self.actions_q99 - self.actions_q01 + 1e-06) * 2.0 - 1.0
        else:
            raise NotImplementedError(f'Unsupported action_norm_method: {self.action_norm_method!r}')
        if cold_first_frame and self._uses_pi05_delta_actions() and (action_model_input.shape[1] > 0):
            action_model_input[:, 0, :] = 0.0
            logger.info('[cold-anchor-fix] preprocess: zeroed normalized frame0 token for cold-chunk KV grounding')
        return action_model_input.unsqueeze(0).unsqueeze(-1)

    def _tactile_image_size(self):
        tactile_resize = int(getattr(self.job_config, 'tactile_resize', 0))
        if tactile_resize > 0:
            return (tactile_resize, tactile_resize)
        return (int(getattr(self.job_config, 'tactile_height', 64)), int(getattr(self.job_config, 'tactile_width', 64)))

    def _preprocess_single_tactile_frame(self, frame):
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.shape[-1] not in (1, 3):
            raise ValueError(f"Unsupported tactile frame shape: {frame.shape}")
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1)
        if frame_tensor.shape[0] == 1:
            frame_tensor = frame_tensor.repeat(3, 1, 1)
        tactile_height, tactile_width = self._tactile_image_size()
        frame_tensor = F.interpolate(
            frame_tensor.unsqueeze(0),
            size=(tactile_height, tactile_width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        return frame_tensor

    def _build_tactile_tensor(self, obs):
        tactile = obs.get('tactile')
        if tactile is None:
            if getattr(self.job_config, 'synthetic_tactile_data', False):
                tactile_height, tactile_width = self._tactile_image_size()
                n_streams = len(self.job_config.tactile_keys)
                # match the number of frames the client sent for VIDEO (obs['obs']),
                # so the synthetic black tactile is frame-aligned and the streaming VAE
                # gets the same temporal length as video (a single frame on a >=3-frame
                # compute_kv_cache grounding would crash WAN's avg_shortcut conv).
                _vid = obs.get('obs')
                n_frames = len(_vid) if isinstance(_vid, list) and len(_vid) >= 1 else 1
                logger.info("Using synthetic zero-valued tactile (synthetic_tactile_data=True, F=%d)", n_frames)
                return torch.zeros(
                    n_streams,
                    3,
                    n_frames,
                    tactile_height,
                    tactile_width,
                    dtype=torch.float32,
                )
            return None

        tactile_history = tactile if isinstance(tactile, list) else [tactile]
        tactile_streams = []
        for key in self.job_config.tactile_keys:
            frames = []
            for tactile_frame_dict in tactile_history:
                if key not in tactile_frame_dict:
                    logger.warning("Missing tactile key %s; dropping tactile condition", key)
                    return None
                frames.append(self._preprocess_single_tactile_frame(tactile_frame_dict[key]))
            tactile_streams.append(torch.stack(frames, dim=1))

        if not tactile_streams:
            return None
        return torch.stack(tactile_streams, dim=0).contiguous()

    def _encode_tactile_residual_latent(self, residual, streaming_vae):
        vae_device = next(streaming_vae.vae.parameters()).device
        residual = residual.to(device=vae_device, dtype=self.dtype)
        enc_out = streaming_vae.encode_chunk(residual)
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return mu_norm.unsqueeze(0).to(self.device, dtype=self.dtype)

    def _encode_tactile_obs(self, obs):
        if not self.job_config.tactile_keys:
            raise ValueError('tactile cond server requires non-empty job_config.tactile_keys.')
        tactile_tensor = self._build_tactile_tensor(obs)
        if tactile_tensor is None:
            raise ValueError(f"tactile cond server requires obs['tactile'] with all configured tactile keys: {self.job_config.tactile_keys}")
        scale = 255.0 if float(tactile_tensor.max()) > 1.5 else 1.0
        sys2_rgb = tactile_tensor / scale if self._sys2_enabled() else None
        first_frames = self.tactile_first_frames
        prev_frames = self.tactile_prev_frames
        if first_frames is None:
            first_frames = tactile_tensor[:, :, 0].clone()
        if prev_frames is None:
            prev_frames = tactile_tensor[:, :, 0].clone()
        global_residual = (tactile_tensor - first_frames[:, :, None]) / scale
        local_mode = str(getattr(self.job_config, 'local_tactile_mode', 'current'))
        if local_mode == 'current':
            local_residual = tactile_tensor / scale * 2.0 - 1.0
        else:
            local_residual = torch.empty_like(tactile_tensor)
            local_residual[:, :, 0] = (tactile_tensor[:, :, 0] - prev_frames) / scale
            if tactile_tensor.shape[2] > 1:
                local_residual[:, :, 1:] = (tactile_tensor[:, :, 1:] - tactile_tensor[:, :, :-1]) / scale
        global_latent = self._encode_tactile_residual_latent(global_residual, self.tactile_global_vae)
        local_latent = self._encode_tactile_residual_latent(local_residual, self.tactile_local_vae)
        if bool(getattr(self.job_config, 'tactile_global_zero', False)):
            global_latent = torch.zeros_like(global_latent)
        if sys2_rgb is not None:
            nframes = global_latent.shape[3]
            raw_frames = sys2_rgb.shape[2]
            stride = self._video_vae_temporal_stride()
            anchors = [raw_frames - 1 - stride * i for i in reversed(range(nframes))]
            if not anchors or anchors[0] < 0:
                raise ValueError('Qwen tactile/WAN temporal mapping mismatch')
            _, ph, pw = self.job_config.patch_size
            target = (global_latent.shape[-2] // ph, global_latent.shape[-1] // pw)
            self._current_tactile_sys2_index = self._get_sys2_client().index(sys2_rgb, anchors, target, tactile=True, contact=obs.get('tactile_contact'), device=self.device)
        sensor_id_map = getattr(self.job_config, 'tactile_sensor_id_map', {}) or {}
        sensor_ids = torch.tensor([int(sensor_id_map.get(key, idx)) for idx, key in enumerate(self.job_config.tactile_keys)], dtype=torch.long, device=self.device)[None]
        self.tactile_first_frames = first_frames.detach()
        self.tactile_prev_frames = tactile_tensor[:, :, -1].detach()
        logger.info('encoded tactile latents: global=%s local=%s sensor_ids=%s', tuple(global_latent.shape), tuple(local_latent.shape), sensor_ids.detach().cpu().tolist())
        return {'tactile_global_latent': global_latent, 'tactile_local_latent': local_latent, 'tactile_sensor_ids': sensor_ids}

    def _reset_tactile_state(self):
        self.tactile_global_vae.clear_cache()
        self.tactile_local_vae.clear_cache()
        self.tactile_first_frames = None
        self.tactile_prev_frames = None
        self._last_gen_tactile = None          # [tactile-pred-eval] don't cross episodes
        self._last_gen_tactile_fsid = None

    def postprocess_action(self, action, current_state=None, output_format=None, cold_first_frame=False):
        action = action.cpu()
        action = action[0, ..., 0]
        if str(self.action_norm_method).lower() in {'quantiles', 'q01q99'}:
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 + 1e-06) + self.actions_q01
        else:
            raise NotImplementedError(f'Unsupported action_norm_method: {self.action_norm_method!r}')
        action = action.squeeze(0).detach().cpu().numpy()
        active_ids = [int(v) for v in getattr(self.job_config, 'used_action_channel_ids', [])]
        inactive_ids = [i for i in range(action.shape[0]) if i not in active_ids]
        if inactive_ids:
            action[inactive_ids] = 0.0
        if self._uses_pi05_delta_actions():
            if cold_first_frame and action.shape[1] > 0:
                for dim in self._pi05_delta_channel_ids():
                    action[dim, 0, :] = 0.0
                logger.info('[cold-anchor-fix] postprocess: zeroed frame0 deltas (cold chunk, frame1 re-anchors to current_state)')
            if bool(getattr(self.job_config, 'delta_smooth', False)):
                dims = list(self._pi05_delta_channel_ids())
                if cold_first_frame:
                    self._delta_smooth_prev = None
                prev = getattr(self, '_delta_smooth_prev', None)
                _, n_frames, n_slots = action.shape
                seq = action[dims].reshape(len(dims), -1).copy()
                if prev is not None and seq.shape[1] > 0:
                    ramp_k = max(1, int(getattr(self.job_config, 'delta_smooth_k', 3)))
                    for i in range(min(ramp_k, seq.shape[1])):
                        w = (i + 1.0) / (ramp_k + 1.0)
                        seq[:, i] = w * seq[:, i] + (1.0 - w) * prev
                    action[dims] = seq.reshape(len(dims), n_frames, n_slots)
                    logger.info('[delta-smooth] ramped first %d actions into prev motion', min(ramp_k, seq.shape[1]))
                self._delta_smooth_prev = seq[:, -1].copy() if seq.shape[1] > 0 else prev
            output_format = str(output_format or getattr(self.job_config, 'server_action_output_format', 'absolute')).lower()
            if output_format in {'absolute', 'absolute_target', 'target'}:
                state_vec = self._extract_current_state_vector(current_state, field_name='current_state')
                for dim in self._pi05_delta_channel_ids():
                    anchor = float(state_vec[dim])
                    for f in range(action.shape[1]):
                        action[dim, f] += anchor
                        anchor = float(action[dim, f, -1])
            elif not output_format in {'pi05_delta', 'openpi_delta', 'pi0.5_delta', 'delta'}:
                raise ValueError(f'Unsupported server_action_output_format for pi05_delta: {output_format!r}')
        return_ids = getattr(self.job_config, 'server_return_action_channel_ids', self.job_config.used_action_channel_ids)
        return action[[int(v) for v in return_ids]]

    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
            if 'tactile_global_latent' in input_dict:
                input_dict['tactile_global_latent'] = input_dict['tactile_global_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_local_latent' in input_dict:
                input_dict['tactile_local_latent'] = input_dict['tactile_local_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_sensor_ids' in input_dict:
                input_dict['tactile_sensor_ids'] = input_dict['tactile_sensor_ids'].repeat(2, 1)
            if 'tactile_noisy_latent' in input_dict:
                input_dict['tactile_noisy_latent'] = input_dict['tactile_noisy_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_timesteps' in input_dict:
                reps = [2] + [1] * (input_dict['tactile_timesteps'].dim() - 1)
                input_dict['tactile_timesteps'] = input_dict['tactile_timesteps'].repeat(*reps)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _global_index_enabled(self):
        return getattr(self.job_config, 'kv_cache_policy', 'fifo') == 'global'

    def _predicted_dino_enabled(self):
        return self._global_index_enabled() and True and bool(getattr(self.job_config, 'kv_index_predicted_dino', True))

    @torch.no_grad()
    def _decode_prediction_for_index(self, latents, frame_st_id):
        """Decode camera streams independently, without advancing encoder state.

        A warm chunk gets one real latent as decoder-only causal context. Its
        RGB is excluded from the returned anchors; it is NOT model feedback.
        This bounded context is an approximation to full-episode VAE decoding.
        """
        if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[2] < 1:
            raise ValueError('predicted index decode requires [1,C,F,H,W] latents')
        frames = latents.shape[2]
        cameras = len(self.job_config.obs_cam_keys)
        if not cameras or latents.shape[-1] % cameras:
            raise ValueError('predicted latent width must split evenly across cameras')
        prefix = int(frame_st_id != 0)
        decode_latents = latents
        if prefix:
            previous = getattr(self, '_last_observed_video_latent', None)
            if previous is None or previous.shape != (1, latents.shape[1], 1, latents.shape[3], latents.shape[4]):
                raise ValueError('predicted continuation index needs the most recent real latent')
            decode_latents = torch.cat((previous.to(latents), latents), dim=2)
        decode_latents = torch.cat(decode_latents.chunk(cameras, dim=-1), dim=0)
        parameter = next(self.vae.parameters())
        decode_latents = decode_latents.to(device=parameter.device, dtype=parameter.dtype)
        mean = torch.as_tensor(self.vae.config.latents_mean, device=parameter.device, dtype=parameter.dtype).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(self.vae.config.latents_std, device=parameter.device, dtype=parameter.dtype).view(1, -1, 1, 1, 1)
        decode_latents = decode_latents * std + mean
        cache_fields = ('_conv_num', '_conv_idx', '_feat_map', '_enc_conv_num', '_enc_conv_idx', '_enc_feat_map')
        old_state = {name: getattr(self.vae, name) for name in cache_fields if hasattr(self.vae, name)}
        try:
            video = self.vae.decode(decode_latents, return_dict=False)[0]
        finally:
            for name in cache_fields:
                if name in old_state:
                    setattr(self.vae, name, old_state[name])
                elif hasattr(self.vae, name):
                    delattr(self.vae, name)
        stride = self._video_vae_temporal_stride()
        expected_frames = 1 + (frames + prefix - 1) * stride
        expected_shape = (cameras, 3, expected_frames, self.height, self.width)
        if tuple(video.shape) != expected_shape or not torch.isfinite(video).all():
            raise ValueError(f'prediction decoder must return finite RGB {expected_shape}, got {tuple(video.shape)}')
        anchors = torch.arange(prefix, frames + prefix, device=video.device) * stride
        return (video.float().clamp(-1, 1).add(1).mul(0.5), anchors)

    @torch.no_grad()
    def _backfill_predicted_video_index(self, latents, frame_st_id, handle):
        """Output RGB -> DINO -> existing KV metadata; never another forward."""
        if dict(getattr(self.job_config, "kv_retention", {})).get("history_storage", "evict") == "cpu_archive":
            return 0  # Temporary predictions do not participate in IKV scoring.
        from n0_twam.preprocessing.kv_index import encode_dense_dino
        pt, ph, pw = self.job_config.patch_size
        if pt != 1 or latents.shape[-2] % ph or latents.shape[-1] % pw:
            raise ValueError('predicted index requires a complete dense WAN patch grid')
        frames, height, width = latents.shape[2], latents.shape[3] // ph, latents.shape[4] // pw
        # Verify full input-token ordering as well as count, even if physical
        # slots were scattered by a global eviction plan.
        ff, hh, ww = torch.meshgrid(
            torch.arange(frame_st_id, frame_st_id + frames, device=self.device),
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device), indexing='ij')
        position = torch.stack((hh.flatten(), ww.flatten(), torch.zeros_like(ww.flatten())), dim=1)
        if (not torch.equal(handle['world_time_id'], ff.flatten())
                or ('grid_position' in handle and not torch.equal(handle['grid_position'], position))):
            raise ValueError('predicted DINO handle does not match the generated video grid')
        if handle['observation_flag'].all():
            return 0
        videos, anchors = self._decode_prediction_for_index(latents, frame_st_id)
        cameras = len(self.job_config.obs_cam_keys)
        if width % cameras:
            raise ValueError('WAN patches must not cross camera boundaries')
        features = encode_dense_dino(
            videos, anchors, (height, width // cameras), self._get_kv_dino_encoder())
        return self.transformer.annotate_video_dino(self.cache_name, handle, features)

    def _get_kv_dino_encoder(self):
        """Real RGB only; independent of the legacy motion detector."""
        encoder = getattr(self, '_kv_dino_encoder', None)
        if encoder is None:
            from n0_twam.preprocessing.dinov3 import FrozenDinoV3PatchEncoder
            device = getattr(self.job_config, 'kv_index_dino_device', 'cpu')
            if str(device) == 'server':
                device = self.device
            from n0_twam.preprocessing.semantic_patch import load_patch_encoder
            semantic_config = dict(getattr(self.job_config, 'kv_semantic_encoder', {}) or {})
            semantic_config.setdefault('model', getattr(self.job_config,
                'kv_index_dino_model_name_or_path', 'facebook/dinov3-vitl16-pretrain-lvd1689m'))
            semantic_config.setdefault('image_size', getattr(self.job_config,
                'kv_index_dino_image_size', (224, 224)))
            encoder = load_patch_encoder(semantic_config, device=device,
                prompt=getattr(self, '_semantic_prompt', None))
            expected = getattr(self.job_config, 'kv_semantic_provenance', None)
            if expected is not None:
                if not hasattr(encoder, 'provenance') or encoder.provenance() != expected:
                    raise ValueError('loaded semantic encoder differs from trained feature provenance')
            self._kv_dino_encoder = encoder
        return encoder

    def _sys2_enabled(self):
        return dict(getattr(self.job_config, "kv_retention", {})).get("score_backend") == "qwen_sys2"

    def _get_sys2_client(self):
        from n0_twam.preprocessing.sys2 import Sys2Client
        c = dict(self.job_config.kv_retention)
        return Sys2Client(c["sys2_endpoint"], c["sys2_task"], c.get("sys2_timeout", 300))

    def _prepare_dense_observed_index(self, obs, videos):
        from n0_twam.preprocessing.kv_index import observed_index, encode_dense_dino
        anchors = self._video_streaming_anchor_indices(videos.shape[2])
        pt, ph, pw = self.job_config.patch_size
        if pt != 1 or self.height % (16 * ph) or self.width % (16 * pw):
            raise ValueError('dense index requires temporal patch_size=1 and spatially divisible camera sizes')
        target = (self.height // (16 * ph), self.width // (16 * pw))
        count = len(anchors) * target[0] * target[1] * len(videos)
        if self._sys2_enabled():
            if obs.get('kv_index'):
                raise ValueError('online Qwen index conflicts with supplied visual index')
            return self._get_sys2_client().index(videos, anchors, target, device=self.device)
        payload = dict(obs.get('kv_index') or {})
        semantic = dict(getattr(self.job_config, 'kv_semantic_encoder', {}) or {})
        semantic_enabled = semantic.get('backend', 'dinov3') != 'dinov3'
        task_enabled = bool(dict(getattr(self.job_config, 'kv_retention', {})).get('task_weight', 0))
        if task_enabled and (not semantic_enabled) and ('task_relevance' not in payload):
            raise ValueError('task_weight requires a semantic backend or supplied patch relevance')
        index = observed_index(payload, count, self.device)
        if semantic_enabled and ('dino' in payload or 'task_relevance' in payload):
            if not {'dino', 'task_relevance'} <= set(payload):
                raise ValueError('semantic observations require both patch features and task relevance')
            encoder = self._get_kv_dino_encoder()
            if obs.get('kv_semantic_provenance') != encoder.provenance():
                raise ValueError('supplied semantic index provenance differs from online encoder')
        if semantic_enabled and ('dino' not in payload or 'task_relevance' not in payload):
            from n0_twam.preprocessing.kv_index import encode_dense_semantic
            generated = encode_dense_semantic(videos, anchors, target, self._get_kv_dino_encoder())
            for name, value in generated.items():
                payload.setdefault(name, value)
            index = observed_index(payload, count, self.device)
        elif 'dino' not in payload and bool(getattr(self.job_config, 'kv_index_dino_online', True)):
            payload['dino'] = encode_dense_dino(videos, anchors, target, self._get_kv_dino_encoder())
            index = observed_index(payload, count, self.device)
        if task_enabled and 'task_relevance' not in payload:
            raise ValueError('task scoring requires observed task relevance')
        return index

    def _prepare_online_contacts(self, obs, *, cold=False):
        if self._sys2_enabled():
            if any((key in obs for key in ('contact_pairs', 'contact_index', 'tactile_kv_index'))):
                raise ValueError('Qwen tactile mode requires tactile_contact, not legacy indices')
            if self.job_config.tactile_keys and 'tactile_contact' not in obs:
                raise ValueError('Qwen tactile index requires reliable tactile_contact labels')
            return obs
        if not (self._global_index_enabled() and bool(getattr(self.job_config, 'kv_neoforce_online', False))):
            return obs
        if 'contact_pairs' in obs or 'contact_index' in obs or 'tactile_kv_index' in obs:
            raise ValueError('online contact encoding conflicts with supplied contact metadata')
        from n0_twam.preprocessing.online_contact import build_online_contact_pairs
        from n0_twam.preprocessing.neoforce import FrozenNeoForceEncoder
        encoder = getattr(self, '_kv_neoforce_encoder', None)
        if encoder is None:
            encoder = FrozenNeoForceEncoder.from_checkpoint(self.job_config.kv_neoforce_checkpoint, device=self.device)
            self._kv_neoforce_encoder = encoder
        frames = len(obs['obs']) if isinstance(obs['obs'], list) else 1
        anchors = self._video_streaming_anchor_indices(frames, streaming_vae_warm=False if cold else None)
        _, ph, pw = self.job_config.patch_size
        th, tw = self._tactile_image_size()
        pairs = build_online_contact_pairs(obs, encoder=encoder, anchors=anchors, camera_keys=self.job_config.obs_cam_keys, tactile_keys=self.job_config.tactile_keys, visual_grid=(self.height // (16 * ph), self.width // (16 * pw)), tactile_grid=(th // (16 * ph), tw // (16 * pw)), paired_camera=self.job_config.obs_cam_keys[0] if dict(getattr(self.job_config, 'kv_retention', {})).get('version', 2) == 2 else self.job_config.kv_contact_pair_camera_keys[0], frame_level=dict(getattr(self.job_config, 'kv_retention', {})).get('version', 2) == 2, device=self.device, min_similarity=getattr(self.job_config, 'kv_contact_min_similarity', 0.7), min_margin=getattr(self.job_config, 'kv_contact_min_margin', 0.15))
        if dict(getattr(self.job_config, 'kv_retention', {})).get('version', 2) == 2:
            return dict(obs, contact_index=pairs)
        active = pairs['response'] > 0
        paired = active & (pairs['visual_rows'] >= 0)
        logger.info('[online-contact] cold=%s active=%s paired=%s tactile_only=%s', cold, active.sum().item(), paired.sum().item(), (active & ~paired).sum().item())
        if cold and (active & ~paired).any():
            logger.info('[cold-contact] unmatched=%s deferred_until_observed_grounding=True', (active & ~paired).sum().item())
        return dict(obs, contact_pairs=pairs)

    def _attach_observed_contact_pairs(self, obs, input_dict, dense_index,
                                       video_latents, tactile_latents, *,
                                       cold_seed_frames=0):
        """Route caller-confirmed RGB matches into this grounding transaction.

        No depth, new tokens, encoder calls, or mutation of stored historical KV.
        This is an input boundary, not an automatic RGB contact localizer.
        Rows of contact_pairs follow the existing tactile (sensor,F,H,W) order.
        """
        if dict(getattr(self.job_config, 'kv_retention', {})).get('version', 2) == 2:
            return self._attach_frame_contacts(obs, input_dict, dense_index,
                video_latents, tactile_latents, cold_seed_frames=cold_seed_frames)
        from n0_twam.preprocessing.kv_index import route_contact_index
        pairs = obs['contact_pairs']
        if set(pairs) != {'neoforce', 'response', 'visual_rows'}:
            raise ValueError('contact_pairs requires neoforce, response and visual_rows')
        if 'tactile_kv_index' in obs:
            raise ValueError('contact_pairs conflicts with pre-aligned tactile_kv_index')
        if tactile_latents is None or 'tactile_global_latent' not in tactile_latents:
            raise ValueError('contact_pairs requires real tactile latents')
        tactile = tactile_latents['tactile_global_latent']
        pt, ph, pw = self.job_config.patch_size
        cameras = tuple(self.job_config.obs_cam_keys)
        if pt != 1 or tactile.ndim != 6 or video_latents.ndim != 5:
            raise ValueError('contact_pairs requires dense RGB/tactile latents and temporal patch size 1')
        _, sensors, _, frames, th, tw = tactile.shape
        vf, vh, vw = video_latents.shape[2:]
        cold_mixed_layout = (
            cold_seed_frames > 0 and vf == frames + cold_seed_frames)
        if ((frames != vf and not cold_mixed_layout)
                or sensors != len(self.job_config.tactile_keys)
                or vh % ph or vw % (pw * len(cameras)) or th % ph or tw % pw):
            raise ValueError('contact_pairs video/tactile frame or patch layout mismatch')
        h, w = vh // ph, vw // (pw * len(cameras))
        count = sensors * frames * (th // ph) * (tw // pw)
        features = torch.as_tensor(pairs['neoforce'], device=self.device)
        if features.ndim != 2 or len(features) != count:
            raise ValueError('contact_pairs must cover exactly the existing tactile token rows')
        if cold_mixed_layout:
            # The released dense path prepends the RGB/action seed, but not a
            # tactile frame. Keep the existing latent/token/RoPE contract;
            # do not invent a same-time RGB match for this shorter tactile tail.
            # Real tactile descriptors remain available as unpaired metadata.
            pairs = dict(pairs, visual_rows=torch.full_like(
                torch.as_tensor(pairs['visual_rows'], device=self.device), -1))
            logger.info('[cold-contact] RGB association deferred for mixed seed layout')
        camera_ids = torch.arange(len(cameras), device=self.device).repeat_interleave(w).repeat(vf * h)
        paired_keys = tuple(getattr(self.job_config, 'kv_contact_pair_camera_keys',
                                   ('observation.images.top', 'observation.images.third_view')))
        allowed = [i for i, key in enumerate(cameras) if key in paired_keys]
        visual, tail, paired = route_contact_index(
            dense_index, features, pairs['response'], pairs['visual_rows'], camera_ids, allowed)
        rows = torch.as_tensor(pairs['visual_rows'], device=self.device)
        contact_frames = torch.arange(frames, device=self.device).repeat_interleave(
            (th // ph) * (tw // pw)).repeat(sensors)
        if (rows[paired] // (h * w * len(cameras)) != contact_frames[paired]).any():
            raise ValueError('contact_pairs cannot attach a contact to another observation time')
        # All validation precedes changes to the local forward inputs.
        input_dict['latent_res_lst']['kv_index'] = visual
        input_dict['latent_res_lst']['tactile_kv_index'] = tail
        input_dict['action_res_lst']['tactile_kv_index'] = tail
        logger.info('[contact-index-prepare] sensors=%s frames=%s paired=%s tactile_rows=%s', sensors, frames, paired.sum().item(), count)

    def _attach_frame_contacts(self, obs, inputs, dense_index, video, tactile, *, cold_seed_frames=0):
        """Frame-level descriptors; no RGB/tactile patch correspondence."""
        if "contact_pairs" in obs:
            raise ValueError("v2 uses contact_index {neoforce,response}; remove legacy visual_rows")
        if "tactile_kv_index" in obs:
            raise ValueError("contact_index conflicts with tactile_kv_index")
        packet = obs.get("contact_index")
        if packet is None or set(packet) != {"neoforce", "response"}:
            raise ValueError("contact_index requires neoforce and response")
        if tactile is None:
            raise ValueError("contact_index requires aligned observed tactile latents")
        _, sensors, _, frames, h, w = tactile["tactile_global_latent"].shape
        _, ph, pw = self.job_config.patch_size
        patches = (h // ph) * (w // pw)
        features = torch.as_tensor(packet["neoforce"], device=self.device).float()
        response = torch.as_tensor(packet["response"], device=self.device).float()
        if (features.ndim != 2 or len(features) != sensors * frames * patches
                or response.shape != features.shape[:1] or not torch.isfinite(features).all()
                or not torch.isfinite(response).all() or (response < 0).any()):
            raise ValueError("contact_index must align with finite sensor/frame/patch rows")
        features = features.masked_fill((response <= 0)[:, None], 0)
        shaped = features.reshape(sensors, frames, patches, -1)
        active = response.reshape(sensors, frames, patches) > 0
        frame_features = shaped.sum((0, 2)) / active.sum((0, 2)).clamp_min(1)[:, None]
        vf, vh, vw = video.shape[2:]
        offset = vf - frames
        if offset not in (0, cold_seed_frames):
            raise ValueError("NeoForce/visual frame alignment mismatch")
        visual = dict(dense_index)
        neo = features.new_zeros(vf, (vh // ph) * (vw // pw), features.shape[-1])
        # Preserve the already observed seed descriptors in a mixed cold layout.
        seed_rows = offset * neo.shape[1]
        if seed_rows:
            previous = dense_index["neoforce"]
            if previous.shape[-1]:
                if previous.shape[-1] != features.shape[-1]:
                    raise ValueError("NeoForce width changed at cold grounding")
                neo[:offset] = previous[:seed_rows].reshape(offset, neo.shape[1], -1)
        neo[offset:] = frame_features[:, None]
        visual["neoforce"] = neo.flatten(0, 1)
        present = torch.zeros(vf, neo.shape[1], dtype=torch.bool, device=self.device)
        if seed_rows and "contact_present" in dense_index:
            present[:offset] = dense_index["contact_present"][:seed_rows].reshape(offset, neo.shape[1])
        present[offset:] = active.any(0).any(-1)[:, None]
        visual["contact_present"] = present.flatten()
        tail = {"neoforce": features, "contact_present": response > 0,
                "observation_flag": torch.ones(len(features), device=self.device)}
        inputs["latent_res_lst"]["kv_index"] = visual
        inputs["latent_res_lst"]["tactile_kv_index"] = tail
        inputs["action_res_lst"]["tactile_kv_index"] = tail

    def _rgb_only_inputs(self, obs):
        frames = obs.get('obs')
        frames = frames if isinstance(frames, list) else [frames]
        keys = self.job_config.obs_cam_keys
        return {'camera_keys': list(keys), 'cameras': {key: {'rgb': torch.stack([torch.as_tensor(frame[key]) for frame in frames])} for key in keys}}

    @staticmethod
    def _video_observation_image(value, *, camera_key, frame_index):
        """Normalize one image exactly as the WAN observation wire sees it."""
        try:
            if isinstance(value, torch.Tensor):
                image = value.detach().cpu()
            else:
                image = torch.as_tensor(np.asarray(value))
        except Exception as exc:
            raise TypeError(f"obs['obs'][{frame_index}][{camera_key!r}] must be an array-like RGB image") from exc
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"obs['obs'][{frame_index}][{camera_key!r}] must be HWC RGB with shape [H,W,3], got {tuple(image.shape)}.")
        if image.is_complex():
            raise TypeError('WAN observation RGB images must be real-valued')
        return image

    def _video_streaming_vae_is_warm(self):
        """Return whether the video streaming VAE has causal history.

        ``frame_st_id`` cannot answer this question: after cold imagination the
        clean seed is already present in the VAE cache while the first
        grounding request still has ``frame_st_id == 0``.  The causal feature
        cache is the source of truth for the next encoder call.
        """
        streaming_vae = getattr(self, 'streaming_vae', None)
        feat_cache = getattr(streaming_vae, 'feat_cache', None)
        if feat_cache is None:
            raise RuntimeError('video encoding cannot determine whether the streaming WAN VAE is cold: streaming_vae.feat_cache is unavailable.')
        try:
            return any((entry is not None for entry in feat_cache))
        except TypeError as exc:
            raise TypeError('streaming_vae.feat_cache must be an iterable causal cache') from exc

    def _video_vae_temporal_stride(self):
        """Read the causal video stride declared by the serving WAN VAE."""
        streaming_vae = getattr(self, 'streaming_vae', None)
        vae = getattr(streaming_vae, 'vae', None)
        if vae is None:
            vae = getattr(self, 'vae', None)
        config = getattr(vae, 'config', None)
        value = getattr(config, 'scale_factor_temporal', None)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'video encoding requires the loaded WAN VAE to declare integer config.scale_factor_temporal; got {value!r}.')
        stride = int(value)
        if stride < 1:
            raise ValueError(f'WAN VAE config.scale_factor_temporal must be positive, got {stride}.')
        return stride

    def _video_streaming_anchor_indices(self, raw_frames, *, streaming_vae_warm=None):
        """Derive raw-frame anchors for the next streaming encode call.

        A fresh WAN causal encoder consumes one seed frame and emits the seed
        latent. Once its feature cache is warm, every latent summarizes one
        complete ``scale_factor_temporal``-frame chunk and is anchored at that
        chunk's causal end. This online layout intentionally differs from the
        offline whole-video layout ``[0, stride, 2*stride, ...]``.
        """
        raw_frames = int(raw_frames)
        if raw_frames < 1:
            raise ValueError(f'video RGB input must contain frames, got T={raw_frames}')
        if streaming_vae_warm is None:
            streaming_vae_warm = self._video_streaming_vae_is_warm()
        elif not isinstance(streaming_vae_warm, (bool, np.bool_)):
            raise TypeError('streaming_vae_warm must be bool when provided')
        if not bool(streaming_vae_warm):
            if raw_frames != 1:
                raise ValueError(f'cold streaming WAN VAE RGB input must contain exactly one seed frame (T=1), got T={raw_frames}.')
            return torch.tensor([0], dtype=torch.long)
        stride = self._video_vae_temporal_stride()
        if raw_frames % stride:
            raise ValueError(f'warm streaming WAN VAE RGB input length must be divisible by config.scale_factor_temporal: T={raw_frames}, stride={stride}.')
        return torch.arange(stride - 1, raw_frames, stride, dtype=torch.long)

    @classmethod
    def _clone_streaming_cache_value(cls, value):
        """Copy cache containers while retaining immutable tensor references.

        WAN causal convolutions replace ``feat_cache[i]`` entries instead of
        modifying the old tensors in place.  A shallow container snapshot is
        therefore a complete undo record and avoids cloning three large VAE
        feature-cache trees on GPU.
        """
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return tuple(value)
        if isinstance(value, dict):
            return dict(value)
        return value

    @classmethod
    def _snapshot_streaming_vae_cache(cls, streaming_vae):
        if streaming_vae is None or not hasattr(streaming_vae, 'feat_cache'):
            return None
        return cls._clone_streaming_cache_value(streaming_vae.feat_cache)

    @staticmethod
    def _restore_streaming_vae_cache(streaming_vae, snapshot):
        if streaming_vae is not None and snapshot is not None:
            streaming_vae.feat_cache = snapshot

    def _snapshot_grounding_state(self):
        """Capture mutable encoder/episode state needed for a safe retry.

        The same snapshot is used by grounding and plain inference.  Tensor
        fields are replaced, rather than mutated in place, by those request
        paths; retaining their references is therefore a complete undo record.
        Streaming VAE feature caches receive their own shallow container copy
        because the encoder replaces individual cache entries.
        """
        return {'video_vae': self._snapshot_streaming_vae_cache(getattr(self, 'streaming_vae', None)), 'tactile_global_vae': self._snapshot_streaming_vae_cache(getattr(self, 'tactile_global_vae', None)), 'tactile_local_vae': self._snapshot_streaming_vae_cache(getattr(self, 'tactile_local_vae', None)), 'tactile_first_frames': getattr(self, 'tactile_first_frames', None), 'tactile_prev_frames': getattr(self, 'tactile_prev_frames', None), 'last_tactile_latents': getattr(self, 'last_tactile_latents', None), 'init_latent': getattr(self, 'init_latent', None), 'current_tactile_sys2_index': getattr(self, '_current_tactile_sys2_index', None), 'current_observed_kv_index': getattr(self, '_current_observed_kv_index', None), 'init_kv_index': getattr(self, '_init_kv_index', None), 'last_observed_video_latent': getattr(self, '_last_observed_video_latent', None), 'last_gen_tactile': getattr(self, '_last_gen_tactile', None), 'last_gen_tactile_fsid': getattr(self, '_last_gen_tactile_fsid', None), 'delta_smooth_prev': getattr(self, '_delta_smooth_prev', None), 'previous_raw_frames': None, 'frame_st_id': getattr(self, 'frame_st_id', 0)}

    def _restore_grounding_state(self, snapshot):
        self._restore_streaming_vae_cache(getattr(self, 'streaming_vae', None), snapshot['video_vae'])
        self._restore_streaming_vae_cache(getattr(self, 'tactile_global_vae', None), snapshot['tactile_global_vae'])
        self._restore_streaming_vae_cache(getattr(self, 'tactile_local_vae', None), snapshot['tactile_local_vae'])
        self.tactile_first_frames = snapshot['tactile_first_frames']
        self.tactile_prev_frames = snapshot['tactile_prev_frames']
        self.last_tactile_latents = snapshot['last_tactile_latents']
        self.init_latent = snapshot['init_latent']
        self._current_tactile_sys2_index = snapshot['current_tactile_sys2_index']
        self._current_observed_kv_index = snapshot['current_observed_kv_index']
        self._init_kv_index = snapshot['init_kv_index']
        self._last_observed_video_latent = snapshot['last_observed_video_latent']
        self._last_gen_tactile = snapshot['last_gen_tactile']
        self._last_gen_tactile_fsid = snapshot['last_gen_tactile_fsid']
        self._delta_smooth_prev = snapshot['delta_smooth_prev']
        self.frame_st_id = snapshot['frame_st_id']

    def _prepare_latent_input(self, latent_model_input, action_model_input, latent_t=0, action_t=0, latent_cond=None, action_cond=None, frame_st_id=0, tactile_latents=None):
        logger.info(f'FRAME START ID: {frame_st_id}')
        input_dict = dict()
        patch_size = tuple(self.job_config.patch_size)
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {'noisy_latents': latent_model_input, 'timesteps': torch.ones([latent_model_input.shape[2]], dtype=torch.float32, device=self.device) * latent_t, 'grid_id': get_mesh_id(latent_model_input.shape[-3] // patch_size[0], latent_model_input.shape[-2] // patch_size[1], latent_model_input.shape[-1] // patch_size[2], 0, 1, frame_st_id).to(self.device), 'text_emb': self.prompt_embeds.to(self.dtype).clone()}
            if latent_cond is not None:
                input_dict['latent_res_lst']['noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0
            if tactile_latents is not None:
                input_dict['latent_res_lst']['tactile_global_latent'] = tactile_latents['tactile_global_latent']
                input_dict['latent_res_lst']['tactile_sensor_ids'] = tactile_latents['tactile_sensor_ids']
        if action_model_input is not None:
            input_dict['action_res_lst'] = {'noisy_latents': action_model_input, 'timesteps': torch.ones([action_model_input.shape[2]], dtype=torch.float32, device=self.device) * action_t, 'grid_id': get_mesh_id(action_model_input.shape[-3], action_model_input.shape[-2], action_model_input.shape[-1], 1, 1, frame_st_id, action=True).to(self.device), 'text_emb': self.prompt_embeds.to(self.dtype).clone()}
            if tactile_latents is not None:
                input_dict['action_res_lst'].update(tactile_latents)
            if action_cond is not None:
                input_dict['action_res_lst']['noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
                if self._global_index_enabled() and dict(getattr(self.job_config, 'kv_retention', {})).get('version', 2) != 2:
                    tokens_per_frame = int(action_model_input.shape[-2] * action_model_input.shape[-1])
                    flags = torch.zeros(action_model_input.shape[2] * tokens_per_frame, dtype=torch.bool, device=self.device)
                    flags[:tokens_per_frame] = True
                    input_dict['action_res_lst']['kv_index'] = {'observation_flag': flags}
            input_dict['action_res_lst']['noisy_latents'][:, ~self.action_mask] *= 0
        if hasattr(self.job_config, 'tactile_keys') and (not self.job_config.tactile_keys):
            for branch in input_dict.values():
                if isinstance(branch, dict):
                    if any((k.startswith('tactile_') for k in branch)):
                        raise ValueError('RGB-only configuration contains tactile inputs')
                    branch['rgb_only'] = True
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            if self._global_index_enabled():
                self._current_observed_kv_index = None
            return None
        videos = []
        for k in self.job_config.obs_cam_keys:
            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(self.height, self.width),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        videos = torch.cat(videos, dim=0) / 255.0
        dense_index = (self._prepare_dense_observed_index(obs, videos)
                       if self._global_index_enabled() else None)
        videos = videos * 2.0 - 1.0
        vae_device = next(self.streaming_vae.vae.parameters()).device
        videos_chunk = videos.to(vae_device).to(self.dtype)
        enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        if dense_index is not None:
            pt, ph, pw = self.job_config.patch_size
            count = (video_latent.shape[2] // pt * (video_latent.shape[3] // ph)
                     * (video_latent.shape[4] // pw))
            if count != len(dense_index['observation_flag']):
                raise ValueError("observed dense index does not match encoded WAN tokens")
            self._current_observed_kv_index = dense_index
        return video_latent.to(self.device)

    def _reset(self, prompt=None):
        from n0_twam.input_contract import validate_dense_rgb_config
        validate_dense_rgb_config(self.job_config)
        logger.info('Reset.')
        if self._sys2_enabled():
            task = prompt if prompt is not None else getattr(self.job_config, 'prompt', None)
            if not isinstance(task, str) or not task.strip():
                raise ValueError('Qwen Sys2 requires the current episode task prompt')
            self.job_config.kv_retention = dict(self.job_config.kv_retention, sys2_task=task)
        cache_policy = getattr(self.job_config, 'kv_cache_policy', 'fifo')
        if cache_policy not in ('fifo', 'global'):
            raise ValueError("kv_cache_policy must be 'fifo' or 'global'")
        if cache_policy == 'global':
            from n0_twam.models.global_kv_retention import RetentionConfig
            RetentionConfig(**dict(getattr(self.job_config, 'kv_retention', {})))
            if self.job_config.patch_size[0] != 1:
                raise ValueError('global RGB index currently requires temporal patch_size=1')
            _, ph, pw = self.job_config.patch_size
            if self.job_config.height % (16 * ph) or self.job_config.width % (16 * pw):
                raise ValueError('global RGB index requires camera sizes divisible by WAN patch stride')
        logger.info('[serve-contract] cache_policy=%s attn_window=%s', cache_policy, self.job_config.attn_window)
        self.use_cfg = self.job_config.guidance_scale > 1 or self.job_config.action_guidance_scale > 1
        self.frame_st_id = 0
        self.init_latent = None
        self._current_observed_kv_index = None
        self._init_kv_index = None
        self.last_tactile_latents = None
        self._last_observed_video_latent = None
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        self._reset_tactile_state()
        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = (self.job_config.height, self.job_config.width)
        self.latent_height, self.latent_width = (self.height // 16, self.width // 16 * len(self.job_config.obs_cam_keys))
        patch_size = self.job_config.patch_size
        latent_token_per_chunk = self.job_config.frame_chunk_size * self.latent_height * self.latent_width // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        if self.job_config.tactile_keys:
            tactile_latent_height = int(getattr(self.job_config, 'tactile_latent_height', 8))
            tactile_latent_width = int(getattr(self.job_config, 'tactile_latent_width', 8))
            tactile_token_per_chunk = len(self.job_config.tactile_keys) * self.job_config.frame_chunk_size * tactile_latent_height * tactile_latent_width // (patch_size[0] * patch_size[1] * patch_size[2])
            latent_token_per_chunk += tactile_token_per_chunk
            action_token_per_chunk += tactile_token_per_chunk
            logger.info('tactile cache tokens per chunk: %d', tactile_token_per_chunk)
        trained_ikv = bool(getattr(self.job_config, 'use_ikv_training', False))
        multimodal_cfg = dict(getattr(self.job_config, 'kv_retention', {}))
        multimodal = cache_policy == 'global' and multimodal_cfg.get('version', 2) == 2
        if multimodal:
            capacity = sum((multimodal_cfg[name] for name in ('video_capacity', 'action_capacity', 'tactile_capacity')))
            if trained_ikv and capacity != int(self.job_config.ikv_train_capacity):
                raise ValueError('v2 serving capacity must match trained modality budgets')
            latent_token_per_chunk, action_token_per_chunk = (capacity, 0)
        elif trained_ikv:
            capacity = int(getattr(self.job_config, 'ikv_train_capacity', 0))
            if capacity <= 0 or cache_policy != 'global':
                raise ValueError('IKV-trained serving requires global policy and positive ikv_train_capacity')
            latent_token_per_chunk, action_token_per_chunk = (capacity, 0)
        self.transformer.create_empty_cache(self.cache_name, 2 if trained_ikv or multimodal else self.job_config.attn_window, latent_token_per_chunk, action_token_per_chunk, dtype=self.dtype, device=self.device, batch_size=2 if self.use_cfg else 1)
        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        if cache_policy == 'global':
            self.transformer.configure_global_retention(self.cache_name, **dict(getattr(self.job_config, 'kv_retention', {})))
        self.action_mask[self.job_config.used_action_channel_ids] = True
        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'], dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'], dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method
        if prompt is None:
            prompt = getattr(self.job_config, 'prompt', None)
        self._semantic_prompt = prompt
        semantic_encoder = getattr(self, '_kv_dino_encoder', None)
        if hasattr(semantic_encoder, 'set_prompt'):
            semantic_encoder.set_prompt(prompt)
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(prompt=prompt, negative_prompt=None, do_classifier_free_guidance=self.use_cfg, num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None, max_sequence_length=512, device=self.device, dtype=self.dtype)
        self.exp_name = f'{prompt}_{time.strftime('%Y%m%d_%H%M%S')}' if prompt else 'default'
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _tactile_pred_to_latent(self, tactile_pred, tactile_g):
        """Un-patchify the model's tactile velocity prediction (patch-sequence
        (B', S*F*spatial, 48)) back to the GlobalTactile latent layout
        (1, S, 48, F, H, W), matching tactile_g — mirrors train.py's tactile loss
        un-patchify (data_seq_to_patch). Drops the CFG duplicate if present."""
        _, S, C, F_lat, H_lat, W_lat = tactile_g.shape
        # CFG: model was fed a 2x-repeated batch -> take the conditional (first) half.
        if tactile_pred.shape[0] > 1:
            if self.job_config.guidance_scale > 1:
                cond = tactile_pred[:1]
                uncond = tactile_pred[1:2]
                tactile_pred = uncond + self.job_config.guidance_scale * (cond - uncond)
            else:
                tactile_pred = tactile_pred[:1]
        pred_dense = data_seq_to_patch(
            self.job_config.patch_size,
            tactile_pred.reshape(S, F_lat * H_lat * W_lat, C),
            F_lat, H_lat, W_lat,
            batch_size=S,
        ).reshape(1, S, C, F_lat, H_lat, W_lat)
        return pred_dense.to(tactile_g.dtype)

    def _infer(self, obs, frame_st_id=0):
        """Generate one chunk as one retry-safe server transaction.

        The final video and action denoising calls both use ``update_cache=1``.
        A request-level transaction keeps those writes indivisible and remains
        open through action postprocessing, while the state snapshot rolls back
        cold-seed VAE and semantic-support mutations on any exception.
        """
        state_snapshot = self._snapshot_grounding_state()
        transformer = getattr(self, 'transformer', None)
        transaction_factory = getattr(transformer, 'cache_transaction', None)
        cache_transaction = (
            transaction_factory(self.cache_name)
            if callable(transaction_factory) else nullcontext()
        )
        try:
            with cache_transaction:
                return self._infer_impl(obs, frame_st_id=frame_st_id)
        except BaseException:
            self._restore_grounding_state(state_snapshot)
            raise

    def _infer_impl(self, obs, frame_st_id=0):
        from n0_twam.input_contract import validate_dense_rgb_payload
        validate_dense_rgb_payload(obs)
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            obs = self._prepare_online_contacts(obs, cold=True)
            self.streaming_vae.clear_cache()
            self._reset_tactile_state()
            tactile_latents = self._encode_tactile_obs(obs) if self.job_config.tactile_keys else None
            self.last_tactile_latents = tactile_latents
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
            if self._global_index_enabled():
                self._init_kv_index = self._current_observed_kv_index
                if 'contact_pairs' in obs or 'contact_index' in obs:
                    seed_inputs = {'latent_res_lst': {}, 'action_res_lst': {}}
                    self._attach_observed_contact_pairs(obs, seed_inputs, self._init_kv_index, init_latent, tactile_latents)
                    self._init_kv_index = seed_inputs['latent_res_lst']['kv_index']
            self._last_observed_video_latent = init_latent[:, :, -1:].detach().clone()
        else:
            tactile_latents = self.last_tactile_latents
        tactile_denoise = bool(getattr(self.job_config, 'server_tactile_denoise', False))
        tactile_g = None
        if tactile_denoise and tactile_latents is not None:
            tactile_g = torch.randn_like(tactile_latents['tactile_global_latent'])
        latent_noise = torch.randn(1, 48, frame_chunk_size, self.latent_height, self.latent_width, device=self.device, dtype=self.dtype)
        latents = latent_noise
        actions = torch.randn(1, self.job_config.action_dim, frame_chunk_size, self.action_per_frame, 1, device=self.device, dtype=self.dtype)
        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step
        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps
        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]
        action_timesteps = F.pad(action_timesteps, (0, 1), mode='constant', value=0)
        video_index_cursor = self.transformer.global_cache_cursor(self.cache_name) if self._predicted_dino_enabled() else None
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(latents, None, t, t, latent_cond, None, frame_st_id=frame_st_id, tactile_latents=tactile_latents)
                if self._global_index_enabled():
                    from n0_twam.preprocessing.kv_index import prediction_index
                    count = input_dict['latent_res_lst']['grid_id'].shape[-1]
                    input_dict['latent_res_lst']['kv_index'] = prediction_index(count, self.device, self._init_kv_index if frame_st_id == 0 else None)
                if tactile_g is not None:
                    input_dict['latent_res_lst']['tactile_noisy_latent'] = tactile_g
                    input_dict['latent_res_lst']['tactile_timesteps'] = torch.ones([tactile_g.shape[1] * tactile_g.shape[3]], device=self.device) * t
                model_input = self._repeat_input_for_cfg(input_dict['latent_res_lst'])
                out = self.transformer(model_input, update_cache=1 if last_step else 0, cache_name=self.cache_name, action_mode=False)
                if isinstance(out, tuple):
                    video_noise_pred, tactile_noise_pred = out
                else:
                    video_noise_pred, tactile_noise_pred = (out, None)
                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(self.job_config.patch_size, video_noise_pred, frame_chunk_size, self.latent_height, self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred, t, latents, return_dict=False)
                    if tactile_g is not None and tactile_noise_pred is not None:
                        _pre = float(tactile_g.float().std())
                        tac_pred = self._tactile_pred_to_latent(tactile_noise_pred, tactile_g)
                        tactile_g = self.scheduler.step(tac_pred, t, tactile_g, return_dict=False)
                        logger.info('[tac-step] t=%s pre_std=%.4f post_std=%.4f pred_std=%.4f', int(t) if hasattr(t, '__int__') else t, _pre, float(tactile_g.float().std()), float(tac_pred.float().std()))
                if frame_st_id == 0:
                    latents[:, :, 0:1] = latent_cond
            if video_index_cursor is not None:
                handle = self.transformer.video_index_handle(self.cache_name, video_index_cursor)
                self._backfill_predicted_video_index(latents, frame_st_id, handle)
            if tactile_g is not None and tactile_latents is not None:
                logger.info('[tactile-denoise] generated GlobalTactile: gen_std=%.4f gen_mean=%.4f', float(tactile_g.float().std()), float(tactile_g.float().mean()))
                self._last_gen_tactile = tactile_g.detach().clone()
                self._last_gen_tactile_fsid = int(frame_st_id)
                tactile_latents = dict(tactile_latents)
                tactile_latents['tactile_global_latent'] = tactile_g
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                if frame_st_id != 0:
                    action_cond = None
                else:
                    _cs_mode = 'zeros' if self._uses_pi05_delta_actions() else str(getattr(self.job_config, 'cold_seed_mode', 'free')).lower()
                    _cs_val = obs.get('current_state')
                    if _cs_mode == 'free':
                        action_cond = None
                    elif _cs_mode == 'current_state' and _cs_val is not None:
                        _cs = np.asarray(_cs_val, dtype=np.float32).reshape(-1)
                        _adim = int(self.job_config.action_dim)
                        if _cs.shape[0] < _adim:
                            _cs = np.pad(_cs, (0, _adim - _cs.shape[0]))
                        _cs_chunk = np.repeat(_cs[:_adim].reshape(-1, 1, 1), self.action_per_frame, axis=2)
                        action_cond = self.preprocess_action(_cs_chunk).to(device=self.device, dtype=self.dtype)
                    else:
                        action_cond = torch.zeros([1, self.job_config.action_dim, 1, self.action_per_frame, 1], device=self.device, dtype=self.dtype)
                input_dict = self._prepare_latent_input(None, actions, t, t, None, action_cond, frame_st_id=frame_st_id, tactile_latents=tactile_latents)
                action_noise_pred = self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']), update_cache=1 if last_step else 0, cache_name=self.cache_name, action_mode=True)
                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred, 'b (f n) c -> b c f n 1', f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred, t, actions, return_dict=False)
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond
        actions[:, ~self.action_mask] *= 0
        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))
        current_state = obs.get('current_state')
        if current_state is None and 'state' in obs:
            current_state = obs['state']
        actions = self.postprocess_action(actions, current_state=current_state, cold_first_frame=frame_st_id == 0)
        torch.cuda.empty_cache()
        return (actions, latents)

    def _execute_grounding_transaction(self, obs, *, initial_latent, request_frame_st_id, grounding_frame_start, seed_is_already_cached):
        """Advance encoders and write both grounding experts inside one transaction."""
        obs = self._prepare_online_contacts(obs)
        if request_frame_st_id == 0 and self._global_index_enabled():
            logger.info('[cold-seed-rebuild] invalidate imagination-time KV0 and rebuild from observations')
            self.transformer.clear_pred_cache(self.cache_name, include_observed=True)
        else:
            self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{request_frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        dense_index = getattr(self, '_current_observed_kv_index', None)
        if request_frame_st_id == 0 and (not seed_is_already_cached):
            latent_model_input = torch.cat([self.init_latent, latent_model_input], dim=2) if latent_model_input is not None else self.init_latent
            if self._global_index_enabled():
                from n0_twam.preprocessing.kv_index import concat_indices
                dense_index = concat_indices(self._init_kv_index, dense_index)
        action_anchor_state = obs.get('action_anchor_state', obs.get('current_state'))
        action_format = obs.get('state_action_format', obs.get('action_format'))
        action_model_input = self.preprocess_action(obs['state'], action_anchor_state=action_anchor_state, action_format=action_format, cold_first_frame=request_frame_st_id == 0)
        if seed_is_already_cached:
            if action_model_input.shape[2] <= 1:
                raise ValueError('cold RGB-motion grounding needs at least one executed action frame after the seed')
            action_model_input = action_model_input[:, :, 1:]
        if 'contact_index' in obs and action_model_input.shape[2] != latent_model_input.shape[2]:
            raise ValueError(f'Grounding temporal mismatch: video latents={latent_model_input.shape[2]}, action frames={action_model_input.shape[2]}; refusing misaligned KV history')
        action_model_input = action_model_input.to(latent_model_input)
        tactile_latents = self._encode_tactile_obs(obs) if not hasattr(self.job_config, 'tactile_keys') or self.job_config.tactile_keys else None
        if self._last_gen_tactile is not None and tactile_latents is not None:
            _gen = self._last_gen_tactile.float()
            _real = tactile_latents['tactile_global_latent'].float()
            if _gen.shape == _real.shape:
                _err = (_gen - _real).abs().mean().item()
                _rstd = _real.std().item()
                logger.info('[tactile-pred-eval] predicted(fsid=%s) vs real-observed-after-exec: |pred-real|mean=%.4f  pred_std=%.4f  real_std=%.4f  rel=%.3f', self._last_gen_tactile_fsid, _err, _gen.std().item(), _rstd, _err / (_rstd + 1e-06))
            else:
                logger.info('[tactile-pred-eval] shape mismatch pred=%s real=%s (skip)', tuple(_gen.shape), tuple(_real.shape))
        logger.info(f'get KV cache obs: {latent_model_input.shape} {action_model_input.shape}')
        input_dict = self._prepare_latent_input(latent_model_input, action_model_input, frame_st_id=grounding_frame_start, tactile_latents=tactile_latents)
        if self._global_index_enabled():
            input_dict['latent_res_lst']['kv_index'] = dense_index
            if self._sys2_enabled() and tactile_latents is not None:
                tail = self._current_tactile_sys2_index
                input_dict['latent_res_lst']['tactile_kv_index'] = tail
                input_dict['action_res_lst']['tactile_kv_index'] = tail
            if 'tactile_kv_index' in obs:
                input_dict['latent_res_lst']['tactile_kv_index'] = obs['tactile_kv_index']
                input_dict['action_res_lst']['tactile_kv_index'] = obs['tactile_kv_index']
            if 'contact_pairs' in obs or 'contact_index' in obs:
                self._attach_observed_contact_pairs(obs, input_dict, dense_index, latent_model_input, tactile_latents, cold_seed_frames=self.init_latent.shape[2] if request_frame_st_id == 0 and (not seed_is_already_cached) else 0)
        last_observed_video_latent = latent_model_input[:, :, -1:].detach().clone()
        with torch.no_grad():
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']), update_cache=2, cache_name=self.cache_name, action_mode=False)
            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']), update_cache=2, cache_name=self.cache_name, action_mode=True)
        return (latent_model_input, tactile_latents, last_observed_video_latent)

    def _compute_kv_cache(self, obs):
        from n0_twam.input_contract import validate_dense_rgb_payload
        validate_dense_rgb_payload(obs)
        initial_latent = None
        request_frame_st_id = int(getattr(self, 'frame_st_id', 0))
        grounding_frame_start = request_frame_st_id
        seed_is_already_cached = False
        if request_frame_st_id == 0:
            initial_latent = getattr(self, 'init_latent', None)
            seed_is_already_cached = False
            if seed_is_already_cached:
                grounding_frame_start = self.init_latent.shape[2] // self.job_config.patch_size[0]
                initial_latent = None
        grounding_snapshot = self._snapshot_grounding_state()
        transaction_factory = getattr(self.transformer, 'cache_transaction', None)
        cache_transaction = transaction_factory(self.cache_name) if callable(transaction_factory) else nullcontext()
        try:
            with cache_transaction:
                latent_model_input, tactile_latents, last_observed_video_latent = self._execute_grounding_transaction(obs, initial_latent=initial_latent, request_frame_st_id=request_frame_st_id, grounding_frame_start=grounding_frame_start, seed_is_already_cached=seed_is_already_cached)
        except BaseException:
            self._restore_grounding_state(grounding_snapshot)
            raise
        self.last_tactile_latents = tactile_latents
        self._last_observed_video_latent = last_observed_video_latent
        torch.cuda.empty_cache()
        self.frame_st_id = grounding_frame_start + latent_model_input.shape[2]
        if self._global_index_enabled() and bool(getattr(self.job_config, 'kv_neoforce_online', False)):
            logger.info('[contact-index-write] committed=True observed=True frame_start=%s frames=%s sensors=%s', grounding_frame_start, latent_model_input.shape[2], len(self.job_config.tactile_keys))

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            # deterministic sampling for reproducible eval: seed torch/np RNG per episode
            # with the client-supplied eval seed, so the same seed reproduces the same
            # diffusion noise every run (fair A/B of cold-seed modes).
            # cfg.deterministic_episode_seed=False disables.
            _seed = obs.get("seed", None)
            if _seed is not None and bool(getattr(self.job_config,
                                                  'deterministic_episode_seed', True)):
                _s = int(_seed) & 0x7fffffff
                torch.manual_seed(_s)
                torch.cuda.manual_seed_all(_s)
                np.random.seed(_s)
                logger.info(f"deterministic manual_seed({_s}) on reset")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()
        else:
            logger.info(f"################# Infer One Chunk #################")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            # fsid discipline: imagination does NOT advance the time axis; only
            # grounding (_compute_kv_cache) does. Advancing on both double-counts
            # time and drifts KV RoPE off the training geometry.
            return dict(action=action)

    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video

    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs

    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        del self.transformer
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    import copy
    config = copy.deepcopy(TWAM_CONFIGS[args.config_name])
    if getattr(args, 'task_overrides', None):
        with open(args.task_overrides) as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict) or {'rank', 'local_rank', 'world_size'} & set(overrides):
            raise ValueError("Invalid task overrides")
        config.update(overrides)
    if getattr(args, 'bundle', None):
        config.wan22_pretrained_model_name_or_path = args.bundle
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = TWAM_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE i2va mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """Parse CLI args and launch the inference server."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='twam_server',
        help="config name (registered in TWAM_CONFIGS).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument("--task-overrides", help="Generated run/serve_overrides.json")
    parser.add_argument("--bundle", help="Trained serve bundle")
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")
if __name__ == '__main__':
    init_logger()
    main()
