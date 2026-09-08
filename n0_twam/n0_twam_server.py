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
from models.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from models.rgb_motion import SparsePatchGather, SparsePatchScatter
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class TWAM_Server:

    @staticmethod
    def _validate_rgb_motion_server_config(job_config):
        """Validate invariants shared by RGB-motion serving entry points."""
        enabled = bool(getattr(job_config, 'use_rgb_motion_tokens', False))
        online = bool(getattr(
            job_config, 'rgb_motion_online_preprocess', False))
        if online and not enabled:
            raise ValueError(
                "rgb_motion_online_preprocess=True requires "
                "use_rgb_motion_tokens=True."
            )
        if not enabled:
            return
        patch_size = tuple(getattr(job_config, 'patch_size', ()))
        if len(patch_size) != 3:
            raise ValueError(
                "RGB-motion serving requires patch_size=(time,height,width), "
                f"got {patch_size}."
            )
        if int(patch_size[0]) != 1:
            raise ValueError(
                "RGB-motion serving currently requires temporal "
                f"patch_size=1, got patch_size={patch_size}."
            )
        if online:
            max_tokens = int(getattr(
                job_config, 'rgb_motion_max_tokens', 0))
            if max_tokens <= 0:
                raise ValueError(
                    "online RGB-motion preprocessing requires "
                    "rgb_motion_max_tokens > 0."
                )
            first_policy = str(getattr(
                job_config, 'rgb_motion_first_frame_policy',
                'require_previous'))
            if first_policy not in ('require_previous', 'empty', 'all'):
                raise ValueError(
                    "rgb_motion_first_frame_policy must be one of "
                    "require_previous, empty, or all."
                )
            camera_keys = tuple(getattr(job_config, 'obs_cam_keys', ()))
            if not camera_keys or len(set(camera_keys)) != len(camera_keys):
                raise ValueError(
                    "online RGB-motion preprocessing requires a non-empty, "
                    "duplicate-free obs_cam_keys camera order."
                )

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.frame_st_id = 0  # defensive init: avoid AttributeError if _infer before _reset (WS reconnect bug)
        self.job_config = job_config
        self._validate_rgb_motion_server_config(job_config)
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_global_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_local_vae = WanVAEStreamingWrapper(self.vae)
        self.tactile_first_frames = None
        self.tactile_prev_frames = None
        # [tactile-pred-eval] last chunk's GENERATED GlobalTactile latent (+ the
        # frame_st_id it predicted for), so the NEXT compute_kv_cache (which carries
        # the REAL tactile observed AFTER executing this chunk's actions) can report
        # |predicted_future_tactile - real_observed_tactile| — the actual measure of
        # how good the tactile prediction is. None until the first generated chunk.
        self._last_gen_tactile = None
        self._last_gen_tactile_fsid = None
        self._last_rgb_motion = None
        self._last_observed_video_latent = None
        # The online RGB-D producer is lazy so precomputed-sidecar users never
        # import transformers or load DINO weights.  Raw previous frames are
        # episode state; unlike the frozen producer, they are cleared on reset.
        self._rgb_motion_preprocessor = None
        self._rgb_motion_dino_encoder = None
        self._rgb_motion_previous_raw_frames = None
        self.rgb_patch_gather = SparsePatchGather(job_config.patch_size)
        self.rgb_patch_scatter = SparsePatchScatter()

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        _tpath = os.path.join(job_config.wan22_pretrained_model_name_or_path, 'transformer')
        _is_mot = False
        try:
            import json as _json
            _is_mot = _json.load(open(os.path.join(_tpath, 'config.json'))).get('is_mot', False)
        except Exception:
            _is_mot = False
        if _is_mot:
            from models.utils import load_mot_checkpoint
            self.transformer = load_mot_checkpoint(
                _tpath, torch_dtype=self.dtype, torch_device=self.device,
                attn_mode='torch',
                config_overrides={
                    'use_rgb_motion_tokens': bool(getattr(
                        job_config, 'use_rgb_motion_tokens', False)),
                    'rgb_motion_require_index': bool(getattr(
                        job_config, 'rgb_motion_require_index', True)),
                })
        else:
            self.transformer = load_transformer(
                _tpath,
                torch_dtype=self.dtype,
                torch_device=self.device,
                max_tactile_streams=job_config.max_tactile_streams,
                target_action_dim=int(getattr(job_config, 'action_dim', 30)),
                attn_mode=getattr(job_config, 'attn_mode', 'flashattn'),
                use_rgb_motion_tokens=bool(getattr(
                    job_config, 'use_rgb_motion_tokens', False)),
                rgb_motion_require_index=bool(getattr(
                    job_config, 'rgb_motion_require_index', True)),
            )
        if bool(getattr(job_config, 'use_rgb_motion_tokens', False)) and not _is_mot:
            raise RuntimeError(
                "Sparse RGB-motion serving requires an MoT checkpoint. The "
                "legacy shared WanAttention inference cache has no per-token "
                "padding/index sidecar, while Video/Tactile/Action expert shared "
                "attention does. Train or convert an is_mot=True checkpoint.")
        logger.info('loaded transformer: %s (is_mot=%s) from %s',
                    type(self.transformer).__name__, _is_mot, _tpath)
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

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
        if q01 and q01 == [-1.0] * len(q01) and q99 == [1.0] * len(q99):
            raise RuntimeError(
                'norm_stat is the [-1, 1] placeholder — the stats file was '
                f'missing when the config was imported '
                f'(norm_stat_path={getattr(self.job_config, "norm_stat_path", None)!r}). '
                'Compute the norm stats and point the config at them before serving.')

        problems = []
        if serve_task:
            # per-task norm source of truth: the pool table, not the snapshot.
            per_path = Path(getattr(self.job_config, 'multitask_norm_path', ''))
            if not per_path.is_file():
                raise RuntimeError(
                    f'[consistency] serve_task={serve_task!r} but the per-task '
                    f'norm table is missing: {per_path}')
            per = json.loads(per_path.read_text()).get(serve_task)
            if per is None:
                raise RuntimeError(
                    f'[consistency] task {serve_task!r} not in {per_path}')
            for key in ('q01', 'q99'):
                want = np.asarray(per.get(key, []), dtype=np.float64)
                live = np.asarray(ns.get(key, []), dtype=np.float64)
                if want.shape != live.shape or not np.allclose(want, live, atol=1e-6):
                    problems.append(
                        f'norm_stat.{key} differs from the per-task table '
                        f'({per_path.name}[{serve_task}])')

        meta_path = (Path(self.job_config.wan22_pretrained_model_name_or_path)
                     / 'transformer').resolve().parent / 'train_meta.json'
        if not meta_path.is_file():
            if problems:
                raise RuntimeError(
                    '[consistency] serve config does not match the per-task '
                    'norm table:\n  ' + '\n  '.join(problems))
            logger.info('[consistency] no train_meta.json next to the checkpoint '
                        '(%s) — skipping the training-snapshot cross-check', meta_path)
            if serve_task:
                logger.info('[consistency] multi-task per-task norm check passed '
                            '(task=%s)', serve_task)
            return
        meta = json.loads(meta_path.read_text())
        if not serve_task:
            meta_ns = meta.get('norm_stat') or {}
            for key in ('q01', 'q99'):
                trained = np.asarray(meta_ns.get(key, []), dtype=np.float64)
                live = np.asarray(ns.get(key, []), dtype=np.float64)
                if trained.shape != live.shape or not np.allclose(trained, live, atol=1e-6):
                    problems.append(f'norm_stat.{key} differs from training')
        if bool(getattr(self.job_config, 'use_rgb_motion_tokens', False)):
            # motion_indices address the WAN grid after cameras are concatenated
            # along width, so matching the set is insufficient: order is part of
            # the token-address contract.  Keep this RGB-only so legacy dense
            # checkpoints (including old metadata without obs_cam_keys) retain
            # their existing startup behaviour.
            trained_cameras = meta.get('obs_cam_keys')
            live_cameras = list(getattr(self.job_config, 'obs_cam_keys', []))
            if trained_cameras is None:
                problems.append(
                    'obs_cam_keys is missing from RGB-motion training metadata; '
                    'camera-grid order cannot be verified')
            elif list(trained_cameras) != live_cameras:
                problems.append(
                    'obs_cam_keys order differs from RGB-motion training: '
                    f'train={list(trained_cameras)!r} serve={live_cameras!r}')
        for key in ('action_norm_method', 'action_delta_mode', 'action_dim',
                    'action_per_frame', 'pi05_action_horizon',
                    'used_action_channel_ids', 'use_local_tactile',
                    'local_tactile_mode', 'tactile_global_zero',
                    'use_rgb_motion_tokens', 'rgb_motion_require_index',
                    'rgb_motion_max_tokens', 'patch_size'):
            if key not in meta:
                continue
            live = getattr(self.job_config, key, None)
            if isinstance(meta[key], bool):
                live = bool(live)
            elif isinstance(meta[key], list):
                live = [int(v) for v in (live or [])]
                meta[key] = [int(v) for v in meta[key]]
            if serve_task and key == 'used_action_channel_ids':
                # training records the union over all tasks; a served task uses
                # its own subset (e.g. 10 of 20 for a single-arm task).
                if not set(live) <= set(meta[key]):
                    problems.append(
                        f'{key}: serve {live!r} is not a subset of train {meta[key]!r}')
                continue
            if meta[key] != live:
                problems.append(f'{key}: train={meta[key]!r} serve={live!r}')
        if problems:
            raise RuntimeError(
                '[consistency] serve config does not match the training '
                f'snapshot {meta_path}:\n  ' + '\n  '.join(problems))
        logger.info('[consistency] train_meta.json cross-check passed (%s%s)',
                    meta_path,
                    f'; multi-task norm checked per-task ({serve_task})'
                    if serve_task else '')

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

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

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

    def preprocess_action(self, action, action_anchor_state=None, action_format=None,
                          cold_first_frame=False):
        action_model_input_np = self._canonical_action_chunk(action)
        if self._uses_pi05_delta_actions():
            action_format = str(action_format or 'absolute').lower()
            if action_format in {'absolute', 'absolute_target', 'target'}:
                anchor_state = self._extract_current_state_vector(
                    action_anchor_state,
                    field_name='action_anchor_state',
                )
                # Symmetric to postprocess_action's per-frame anchoring: recover the
                # per-frame raw deltas the model was trained on (and that the KV cache
                # expects) by subtracting each frame's own anchor (frame 0 -> supplied
                # anchor_state, frame f>0 -> frame f-1's last absolute target). Capture
                # the anchors from the ORIGINAL absolute values before subtracting, so
                # the grounding round-trip still cancels and the KV cache stays correct.
                for dim in self._pi05_delta_channel_ids():
                    frame_anchors = [float(anchor_state[dim])]
                    for f in range(1, action_model_input_np.shape[1]):
                        frame_anchors.append(float(action_model_input_np[dim, f - 1, -1]))
                    for f in range(action_model_input_np.shape[1]):
                        action_model_input_np[dim, f] -= frame_anchors[f]
            elif action_format in {'pi05_delta', 'openpi_delta', 'pi0.5_delta', 'delta'}:
                pass
            else:
                raise ValueError(
                    f"Unsupported state_action_format for pi05_delta: "
                    f"{action_format!r}"
                )

        action_model_input = torch.from_numpy(action_model_input_np)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if str(self.action_norm_method).lower() in {'quantiles', 'q01q99'}:
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method!r}")
        # cold-chunk grounding: training masks the whole
        # frame0 action token to zeros in NORMALIZED space
        # (pi05_condition_first_frame_zero). With the postprocess fix, frame0 now
        # round-trips as raw-zero deltas, which normalize to (0-q01)/(q99-q01)*2-1
        # != 0 for asymmetric quantiles — zero it here or the KV-cache history
        # token drifts off the training distribution.
        if (cold_first_frame and self._uses_pi05_delta_actions()
                and action_model_input.shape[1] > 0):
            action_model_input[:, 0, :] = 0.
            logger.info('[cold-anchor-fix] preprocess: zeroed normalized frame0 '
                        'token for cold-chunk KV grounding')
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def _tactile_image_size(self):
        tactile_resize = int(getattr(self.job_config, 'tactile_resize', 0) or 0)
        if tactile_resize > 0:
            return tactile_resize, tactile_resize
        return (
            int(getattr(self.job_config, 'tactile_height', 64)),
            int(getattr(self.job_config, 'tactile_width', 64)),
        )

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
            raise ValueError(
                "tactile cond server requires non-empty job_config.tactile_keys."
            )

        # Tactile mirrors video: the persistent streaming VAE is advanced ONLY by
        # compute_kv_cache (grounding) and by the frame_st_id==0 cold seed in _infer —
        # never by a mid-episode plain-infer. So no per-call clear: the cache is cold
        # only right after reset (the 1-frame seed) and warm for every kv_cache chunk,
        # exactly like streaming_vae for video.
        tactile_tensor = self._build_tactile_tensor(obs)
        if tactile_tensor is None:
            raise ValueError(
                "tactile cond server requires obs['tactile'] with all configured "
                f"tactile keys: {self.job_config.tactile_keys}"
            )

        scale = 255.0 if float(tactile_tensor.max()) > 1.5 else 1.0
        first_frames = self.tactile_first_frames
        prev_frames = self.tactile_prev_frames
        if first_frames is None:
            first_frames = tactile_tensor[:, :, 0].clone()
        if prev_frames is None:
            prev_frames = tactile_tensor[:, :, 0].clone()

        global_residual = (tactile_tensor - first_frames[:, :, None]) / scale

        # LocalTactile: 'residual' (legacy) = frame[t]-frame[t-1]; 'current' = the
        # current frame mapped to [-1,1] (no delta), matching encode_tactile_latent's
        # --local-mode current. Must match how the loaded ckpt was trained.
        local_mode = str(getattr(self.job_config, 'local_tactile_mode', 'current'))
        if local_mode == 'current':
            # (t/255)*2 - 1  ==  encode script's frames_u8/127.5 - 1  (tactile_tensor is [0,255])
            local_residual = (tactile_tensor / scale) * 2.0 - 1.0
        else:
            local_residual = torch.empty_like(tactile_tensor)
            local_residual[:, :, 0] = (tactile_tensor[:, :, 0] - prev_frames) / scale
            if tactile_tensor.shape[2] > 1:
                local_residual[:, :, 1:] = (
                    tactile_tensor[:, :, 1:] - tactile_tensor[:, :, :-1]
                ) / scale

        global_latent = self._encode_tactile_residual_latent(
            global_residual, self.tactile_global_vae)
        local_latent = self._encode_tactile_residual_latent(
            local_residual, self.tactile_local_vae)

        # global-ablation serve mode: zero ONLY GlobalTactile everywhere (clean
        # cond, denoise target, kv grounding); LocalTactile stays real.
        if bool(getattr(self.job_config, 'tactile_global_zero', False)):
            global_latent = torch.zeros_like(global_latent)

        sensor_id_map = getattr(self.job_config, 'tactile_sensor_id_map', {}) or {}
        sensor_ids = torch.tensor(
            [int(sensor_id_map.get(key, idx)) for idx, key in enumerate(self.job_config.tactile_keys)],
            dtype=torch.long,
            device=self.device,
        )[None]

        self.tactile_first_frames = first_frames.detach()
        self.tactile_prev_frames = tactile_tensor[:, :, -1].detach()
        logger.info(
            "encoded tactile latents: global=%s local=%s sensor_ids=%s",
            tuple(global_latent.shape),
            tuple(local_latent.shape),
            sensor_ids.detach().cpu().tolist(),
        )
        return {
            'tactile_global_latent': global_latent,
            'tactile_local_latent': local_latent,
            'tactile_sensor_ids': sensor_ids,
        }

    def _reset_tactile_state(self):
        self.tactile_global_vae.clear_cache()
        self.tactile_local_vae.clear_cache()
        self.tactile_first_frames = None
        self.tactile_prev_frames = None
        self._last_gen_tactile = None          # [tactile-pred-eval] don't cross episodes
        self._last_gen_tactile_fsid = None

    def postprocess_action(self, action, current_state=None, output_format=None,
                           cold_first_frame=False):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  # C, F, H
        if str(self.action_norm_method).lower() in {'quantiles', 'q01q99'}:
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError(
                f"Unsupported action_norm_method: {self.action_norm_method!r}")
        action = action.squeeze(0).detach().cpu().numpy()
        active_ids = [int(v) for v in getattr(self.job_config, 'used_action_channel_ids', [])]
        inactive_ids = [i for i in range(action.shape[0]) if i not in active_ids]
        if inactive_ids:
            action[inactive_ids] = 0.0

        if self._uses_pi05_delta_actions():
            # cold chunk (frame_st_id==0): frame0 is the
            # zero-clamped condition slot, not a prediction. De-normalized it lands
            # on the q01/q99 midpoint (a small constant offset), and the
            # sequential anchoring below carries that constant bias into EVERY
            # frame of the chunk. Zero the frame0 deltas so frame0 == current_state
            # exactly and frame1 re-anchors to the real current state.
            if cold_first_frame and action.shape[1] > 0:
                for dim in self._pi05_delta_channel_ids():
                    action[dim, 0, :] = 0.0
                logger.info('[cold-anchor-fix] postprocess: zeroed frame0 deltas '
                            '(cold chunk, frame1 re-anchors to current_state)')
            # [delta-smooth] opt-in: ramp the first delta_smooth_k actions of a
            # WARM chunk into the previous chunk's last delta (cold chunks reset).
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
                    logger.info('[delta-smooth] ramped first %d actions into prev motion',
                                min(ramp_k, seq.shape[1]))
                self._delta_smooth_prev = seq[:, -1].copy() if seq.shape[1] > 0 else prev
            output_format = str(
                output_format
                or getattr(self.job_config, 'server_action_output_format', 'absolute')
            ).lower()
            if output_format in {'absolute', 'absolute_target', 'target'}:
                state_vec = self._extract_current_state_vector(
                    current_state,
                    field_name='current_state',
                )
                # Per-frame anchoring (fix intra-chunk retreat seam): each predicted
                # frame's deltas were trained relative to that frame's OWN anchor
                # state, and consecutive frame anchors are action_per_frame steps
                # apart (see _build_pi05_delta_actions in
                # lerobot_latent_dataset_pi05_delta.py). Reconstructing every frame against one shared current_state
                # short-changed every frame after the first by ~one frame of motion,
                # producing a backward step at each frame0->frame1 seam. Instead
                # reconstruct sequentially: frame f anchors to frame f-1's last
                # absolute target so the executed trajectory stays continuous.
                for dim in self._pi05_delta_channel_ids():
                    anchor = float(state_vec[dim])
                    for f in range(action.shape[1]):
                        action[dim, f] += anchor
                        anchor = float(action[dim, f, -1])
            elif output_format in {'pi05_delta', 'openpi_delta', 'pi0.5_delta', 'delta'}:
                pass
            else:
                raise ValueError(
                    f"Unsupported server_action_output_format for pi05_delta: "
                    f"{output_format!r}"
                )
        return_ids = getattr(
            self.job_config,
            'server_return_action_channel_ids',
            self.job_config.used_action_channel_ids,
        )
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
            # tactile-denoise (video loop) injects these AFTER the other keys; must
            # also be CFG-doubled or the cond/uncond batch sizes mismatch -> crash.
            if 'tactile_noisy_latent' in input_dict:
                input_dict['tactile_noisy_latent'] = input_dict['tactile_noisy_latent'].repeat(2, 1, 1, 1, 1, 1)
            if 'tactile_timesteps' in input_dict:
                reps = [2] + [1] * (input_dict['tactile_timesteps'].dim() - 1)
                input_dict['tactile_timesteps'] = input_dict['tactile_timesteps'].repeat(*reps)
            self._repeat_rgb_motion_batch(input_dict, 2)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _global_index_enabled(self):
        return getattr(self.job_config, 'kv_cache_policy', 'fifo') == 'global'

    def _predicted_dino_enabled(self):
        return self._global_index_enabled() and bool(getattr(
            self.job_config, 'kv_index_predicted_dino', True))

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
            if (previous is None or previous.shape !=
                    (1, latents.shape[1], 1, latents.shape[3], latents.shape[4])):
                raise ValueError('predicted continuation index needs the most recent real latent')
            decode_latents = torch.cat((previous.to(latents), latents), dim=2)
        decode_latents = torch.cat(decode_latents.chunk(cameras, dim=-1), dim=0)
        parameter = next(self.vae.parameters())
        decode_latents = decode_latents.to(device=parameter.device, dtype=parameter.dtype)
        mean = torch.as_tensor(self.vae.config.latents_mean, device=parameter.device,
                               dtype=parameter.dtype).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(self.vae.config.latents_std, device=parameter.device,
                              dtype=parameter.dtype).view(1, -1, 1, 1, 1)
        decode_latents = decode_latents * std + mean
        # Diffusers' decode clears its internal encoder/decoder containers.
        # Our streaming wrapper owns a separate feat_cache; preserve both its
        # semantics and any existing VAE internal containers, including errors.
        cache_fields = ('_conv_num', '_conv_idx', '_feat_map',
                        '_enc_conv_num', '_enc_conv_idx', '_enc_feat_map')
        old_state = {name: getattr(self.vae, name) for name in cache_fields
                     if hasattr(self.vae, name)}
        try:
            video = self.vae.decode(decode_latents, return_dict=False)[0]
        finally:
            for name in cache_fields:
                if name in old_state:
                    setattr(self.vae, name, old_state[name])
                elif hasattr(self.vae, name):
                    delattr(self.vae, name)
        stride = self._rgb_motion_vae_temporal_stride()
        expected_frames = 1 + (frames + prefix - 1) * stride
        expected_shape = (cameras, 3, expected_frames, self.height, self.width)
        if tuple(video.shape) != expected_shape or not torch.isfinite(video).all():
            raise ValueError(f'prediction decoder must return finite RGB {expected_shape}, got {tuple(video.shape)}')
        anchors = torch.arange(prefix, frames + prefix, device=video.device) * stride
        return video.float().clamp(-1, 1).add(1).mul(.5), anchors

    @torch.no_grad()
    def _backfill_predicted_video_index(self, latents, frame_st_id, handle):
        """Output RGB -> DINO -> existing KV metadata; never another forward."""
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
                or not torch.equal(handle['grid_position'], position)):
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
            from n0_twam.preprocessing.dinov2 import FrozenDinoV2PatchEncoder
            device = getattr(self.job_config, 'kv_index_dino_device', 'cpu')
            if str(device) == 'server':
                device = self.device
            encoder = FrozenDinoV2PatchEncoder.from_pretrained(
                getattr(self.job_config, 'kv_index_dino_model_name_or_path',
                        'facebook/dinov2-base'),
                local_files_only=True, device=device, torch_dtype=torch.float32,
                image_size=getattr(self.job_config, 'kv_index_dino_image_size', (224, 224)),
                float_input_range='0_1', output_dtype=torch.float32)
            self._kv_dino_encoder = encoder
        return encoder

    def _prepare_dense_observed_index(self, obs, videos):
        from n0_twam.preprocessing.kv_index import observed_index, encode_dense_dino
        anchors = self._rgb_motion_streaming_anchor_indices(videos.shape[2])
        pt, ph, pw = self.job_config.patch_size
        if pt != 1 or self.height % (16 * ph) or self.width % (16 * pw):
            raise ValueError("dense index requires temporal patch_size=1 and spatially divisible camera sizes")
        target = (self.height // (16 * ph), self.width // (16 * pw))
        count = len(anchors) * target[0] * target[1] * len(videos)
        payload = dict(obs.get('kv_index') or {})
        # Validate supplied features before spending time on DINO.
        index = observed_index(payload, count, self.device)
        if 'dino' not in payload and bool(getattr(self.job_config, 'kv_index_dino_online', True)):
            payload['dino'] = encode_dense_dino(
                videos, anchors, target, self._get_kv_dino_encoder())
            index = observed_index(payload, count, self.device)
        return index

    def _get_rgb_motion_preprocessor(self):
        """Lazily construct the local-only online RGB-D producer.

        This method is reached only for a real observation that has raw
        ``rgb_motion_inputs`` and no precomputed sidecar.  In particular it is
        never called from a diffusion denoising iteration.
        """
        existing = getattr(self, '_rgb_motion_preprocessor', None)
        if existing is not None:
            return existing
        if not bool(getattr(
                self.job_config, 'rgb_motion_online_preprocess', False)):
            raise RuntimeError("online RGB-motion preprocessing is disabled")

        from n0_twam.models.rgb_motion import (
            EgoMotionCompensatedMotionDetector,
        )
        from n0_twam.preprocessing.dinov2 import FrozenDinoV2PatchEncoder
        from n0_twam.preprocessing.rgb_motion_sequence import (
            RGBMotionSequencePreprocessor,
        )

        dino_encoder = getattr(self, '_rgb_motion_dino_encoder', None)
        if dino_encoder is None:
            source = getattr(
                self.job_config,
                'rgb_motion_dino_model_name_or_path',
                'facebook/dinov2-base',
            )
            if not isinstance(source, (str, os.PathLike)) or not str(source):
                raise ValueError(
                    "rgb_motion_dino_model_name_or_path must name a local "
                    "DINOv2 checkpoint or an already-cached model id."
                )
            dino_device = getattr(
                self.job_config, 'rgb_motion_dino_device', 'cpu')
            if str(dino_device).lower() == 'server':
                dino_device = self.device
            # Serving is deliberately local-only.  There is no config switch
            # which can accidentally authorize a network download.
            dino_encoder = FrozenDinoV2PatchEncoder.from_pretrained(
                source,
                local_files_only=True,
                device=dino_device,
                torch_dtype=torch.float32,
                image_size=getattr(
                    self.job_config, 'rgb_motion_dino_image_size', (224, 224)),
                float_input_range=getattr(
                    self.job_config,
                    'rgb_motion_dino_float_input_range',
                    '0_1',
                ),
                output_dtype=torch.float32,
            )
            self._rgb_motion_dino_encoder = dino_encoder

        detector = EgoMotionCompensatedMotionDetector(
            depth_threshold=float(getattr(
                self.job_config, 'rgb_motion_depth_threshold', 0.02)),
            dino_threshold=float(getattr(
                self.job_config, 'rgb_motion_dino_threshold', 0.2)),
            depth_weight=float(getattr(
                self.job_config, 'rgb_motion_depth_weight', 1.0)),
            dino_weight=float(getattr(
                self.job_config, 'rgb_motion_dino_weight', 1.0)),
            dilation_radius=int(getattr(
                self.job_config, 'rgb_motion_dilation_radius', 1)),
            max_tokens=None,
            min_depth=float(getattr(
                self.job_config, 'rgb_motion_min_depth', 1e-6)),
            # The wire field is explicitly named world_from_camera.  Robot
            # state/action values are never used as camera extrinsics.
            pose_convention='world_from_camera',
        )

        patch_size = tuple(self.job_config.patch_size)
        height = int(self.job_config.height)
        width = int(self.job_config.width)
        if height % 16 or width % 16:
            raise ValueError(
                "online RGB-motion preprocessing requires server height and "
                "width divisible by the WAN VAE spatial factor 16."
            )
        latent_height, latent_width = height // 16, width // 16
        if (latent_height % int(patch_size[1])
                or latent_width % int(patch_size[2])):
            raise ValueError(
                "online RGB-motion preprocessing cannot form the configured "
                f"WAN patch grid from latent size {(latent_height, latent_width)} "
                f"and patch_size={patch_size}."
            )
        per_camera_wan_grid = (
            latent_height // int(patch_size[1]),
            latent_width // int(patch_size[2]),
        )
        preprocessor = RGBMotionSequencePreprocessor(
            dino_encoder,
            detector,
            max_tokens=int(self.job_config.rgb_motion_max_tokens),
            wan_grid_size=per_camera_wan_grid,
            first_frame_policy=str(getattr(
                self.job_config,
                'rgb_motion_first_frame_policy',
                'require_previous',
            )),
            camera_keys=tuple(self.job_config.obs_cam_keys),
            patch_size=patch_size,
        )
        self._rgb_motion_preprocessor = preprocessor
        logger.info(
            "initialized local-only online RGB-motion producer: cameras=%s, "
            "WAN grid/camera=%s, DINO=%s",
            list(self.job_config.obs_cam_keys),
            per_camera_wan_grid,
            type(dino_encoder).__name__,
        )
        return preprocessor

    @staticmethod
    def _rgb_motion_input_tensor(value, name):
        try:
            return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        except Exception as exc:
            raise TypeError(
                f"{name} must be tensor/array-like"
            ) from exc

    @classmethod
    def _rgb_motion_integer_tensor(cls, value, name):
        tensor = cls._rgb_motion_input_tensor(value, name)
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise TypeError(f"{name} must contain integer values")
        if torch.is_floating_point(tensor) and tensor.numel():
            if (not torch.isfinite(tensor).all()
                    or not torch.equal(tensor, tensor.round())):
                raise ValueError(
                    f"{name} must contain finite integer values"
                )
        return tensor.to(dtype=torch.long)

    @classmethod
    def _rgb_motion_binary_tensor(cls, value, name):
        tensor = cls._rgb_motion_input_tensor(value, name)
        if tensor.is_complex() or (
                tensor.numel()
                and not torch.all((tensor == 0) | (tensor == 1))):
            raise ValueError(f"{name} must contain only 0 or 1")
        return tensor.to(dtype=torch.bool)

    def _rgb_motion_camera_sequences(self, payload, *, label):
        """Parse one explicitly camera-ordered raw RGB-D observation."""
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must be a dict")
        camera_order = payload.get('camera_keys')
        if camera_order is None:
            raise KeyError(
                f"{label} is missing camera_keys; online motion indices need "
                "an explicit camera order."
            )
        if isinstance(camera_order, (str, bytes)):
            raise TypeError(f"{label}.camera_keys must be a sequence of names")
        camera_order = tuple(camera_order)
        configured_order = tuple(getattr(self.job_config, 'obs_cam_keys', ()))
        if camera_order != configured_order:
            raise ValueError(
                f"{label}.camera_keys must exactly match obs_cam_keys order: "
                f"expected {list(configured_order)}, got {list(camera_order)}."
            )

        cameras_payload = payload.get('cameras')
        if not isinstance(cameras_payload, dict):
            raise KeyError(f"{label} must contain a cameras mapping")
        if set(cameras_payload) != set(camera_order):
            raise ValueError(
                f"{label}.cameras must contain exactly {list(camera_order)}, "
                f"got {list(cameras_payload)}."
            )

        from n0_twam.preprocessing.rgb_motion_sequence import RGBDCameraSequence

        sequences = {}
        frame_counts = set()
        for camera_key in camera_order:
            camera = cameras_payload[camera_key]
            if not isinstance(camera, dict):
                raise TypeError(
                    f"{label}.cameras[{camera_key!r}] must be a dict"
                )
            required = ('rgb', 'depth', 'world_from_camera', 'intrinsics')
            missing = [name for name in required if name not in camera]
            if missing:
                raise KeyError(
                    f"{label}.cameras[{camera_key!r}] is missing {missing}. "
                    "world_from_camera is a camera extrinsic; obs['state'] is "
                    "robot state and is not a substitute."
                )

            rgb = self._rgb_motion_input_tensor(
                camera['rgb'], f"{camera_key}.rgb")
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            depth = self._rgb_motion_input_tensor(
                camera['depth'], f"{camera_key}.depth")
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if not torch.is_floating_point(depth):
                raise TypeError(
                    f"{label}.cameras[{camera_key!r}].depth must be calibrated "
                    "floating-point z-depth; integer depth units are ambiguous."
                )
            depth = depth.to(dtype=torch.float32)
            pose = self._rgb_motion_input_tensor(
                camera['world_from_camera'],
                f"{camera_key}.world_from_camera",
            ).to(dtype=torch.float32)
            intrinsics = self._rgb_motion_input_tensor(
                camera['intrinsics'], f"{camera_key}.intrinsics"
            ).to(dtype=torch.float32)
            try:
                sequence = RGBDCameraSequence(
                    rgb=rgb,
                    depth=depth,
                    camera_pose=pose,
                    camera_intrinsics=intrinsics,
                    dino_grid_size=camera.get('dino_grid_size'),
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"{label}.cameras[{camera_key!r}]: {exc}"
                ) from exc
            channels_first = sequence.rgb.shape[1] == 3
            channels_last = sequence.rgb.shape[-1] == 3
            if channels_first == channels_last:
                raise ValueError(
                    f"{label}.cameras[{camera_key!r}].rgb must have exactly "
                    "one 3-channel axis ([T,3,H,W] or [T,H,W,3])."
                )
            rgb_spatial = (
                tuple(sequence.rgb.shape[-2:])
                if channels_first
                else tuple(sequence.rgb.shape[1:3])
            )
            depth_spatial = tuple(sequence.depth.shape[-2:])
            if rgb_spatial != depth_spatial:
                raise ValueError(
                    f"{label}.cameras[{camera_key!r}] RGB/depth spatial "
                    f"shapes must match, got {rgb_spatial} and {depth_spatial}."
                )
            sequences[camera_key] = sequence
            frame_counts.add(sequence.num_frames)

        if len(frame_counts) != 1:
            raise ValueError(
                f"{label} cameras must have the same aligned frame count, "
                f"got {sorted(frame_counts)}."
            )
        return sequences, frame_counts.pop()

    @staticmethod
    def _rgb_motion_capture_anchor_raw_frames(sequences, anchor_index):
        """Detach the newest selected anchor, not an unselected raw tail."""
        anchor_index = int(anchor_index)

        def _matrix_at(value):
            if value.ndim == 2:
                return value
            if value.shape[0] == 1:
                return value[0]
            return value[anchor_index]

        captured = {}
        for camera_key, sequence in sequences.items():
            pose = _matrix_at(sequence.camera_pose)
            intrinsics = _matrix_at(sequence.camera_intrinsics)
            captured[camera_key] = {
                'rgb': sequence.rgb[anchor_index].detach().cpu().clone(),
                'depth': sequence.depth[anchor_index].detach().cpu().clone(),
                'camera_pose': pose.detach().cpu().clone(),
                'camera_intrinsics': intrinsics.detach().cpu().clone(),
            }
        return captured

    @staticmethod
    def _rgb_motion_precomputed_payload(obs):
        """Return either supported precomputed representation, if present."""
        payload = obs.get('rgb_motion')
        if payload is not None:
            return payload
        canonical_names = (
            'motion_indices', 'motion_valid_mask', 'motion_scores',
            'world_time_id', 'dino_features', 'neoforce_features',
            'observation_flag', 'visual_valid', 'tactile_valid')
        if any(name in obs for name in canonical_names):
            names = canonical_names + (
                'camera_keys', 'patch_size', 'spatial_grid_shape')
            return {name: obs[name] for name in names if name in obs}
        return None

    def _validate_rgb_motion_payload_provenance(self, payload):
        """Bind multi-camera local indices to the server's width-concat grid."""
        names = ('camera_keys', 'patch_size', 'spatial_grid_shape')
        missing = [name for name in names if name not in payload]
        camera_order = tuple(getattr(self.job_config, 'obs_cam_keys', ()))
        if len(camera_order) > 1 and missing:
            raise KeyError(
                "multi-camera rgb_motion requires camera/grid provenance "
                f"fields {list(names)}; missing {missing}."
            )
        # Keep the legacy minimal schema for one camera.  If a producer starts
        # supplying provenance, require the complete atomic set rather than
        # trusting a partial grid description.
        if len(missing) == len(names):
            return
        if missing:
            raise KeyError(
                f"rgb_motion camera/grid provenance is incomplete; missing {missing}."
            )

        supplied_cameras = payload['camera_keys']
        if isinstance(supplied_cameras, (str, bytes)):
            raise TypeError("rgb_motion.camera_keys must be a sequence of names")
        supplied_cameras = tuple(supplied_cameras)
        if supplied_cameras != camera_order:
            raise ValueError(
                "rgb_motion.camera_keys must exactly match the WAN width-concat "
                f"order: expected {list(camera_order)}, got "
                f"{list(supplied_cameras)}."
            )

        supplied_patch = self._rgb_motion_integer_tensor(
            payload['patch_size'], 'rgb_motion.patch_size'
        ).flatten()
        expected_patch = tuple(int(v) for v in self.job_config.patch_size)
        if supplied_patch.numel() != 3 or tuple(
                supplied_patch.detach().cpu().tolist()) != expected_patch:
            raise ValueError(
                "rgb_motion.patch_size does not match the server: expected "
                f"{expected_patch}, got "
                f"{tuple(supplied_patch.detach().cpu().tolist())}."
            )

        height = int(self.job_config.height)
        width = int(self.job_config.width)
        if height % 16 or width % 16:
            raise ValueError(
                "server height/width must be divisible by WAN VAE factor 16"
            )
        latent_height = height // 16
        latent_width = (width // 16) * len(camera_order)
        if (latent_height % expected_patch[1]
                or latent_width % expected_patch[2]):
            raise ValueError(
                "server latent grid is not divisible by patch_size for "
                "RGB-motion provenance validation"
            )
        expected_grid = (
            latent_height // expected_patch[1],
            latent_width // expected_patch[2],
        )
        supplied_grid = self._rgb_motion_integer_tensor(
            payload['spatial_grid_shape'], 'rgb_motion.spatial_grid_shape'
        ).flatten()
        if supplied_grid.numel() != 2 or tuple(
                supplied_grid.detach().cpu().tolist()) != expected_grid:
            raise ValueError(
                "rgb_motion.spatial_grid_shape does not match the server's "
                f"multi-camera WAN grid: expected {expected_grid}, got "
                f"{tuple(supplied_grid.detach().cpu().tolist())}."
            )

    @staticmethod
    def _rgb_motion_observation_image(value, *, camera_key, frame_index):
        """Normalize one image exactly as the WAN observation wire sees it."""
        try:
            if isinstance(value, torch.Tensor):
                image = value.detach().cpu()
            else:
                image = torch.as_tensor(np.asarray(value))
        except Exception as exc:
            raise TypeError(
                f"obs['obs'][{frame_index}][{camera_key!r}] must be an "
                "array-like RGB image"
            ) from exc
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"obs['obs'][{frame_index}][{camera_key!r}] must be HWC RGB "
                f"with shape [H,W,3], got {tuple(image.shape)}."
            )
        if image.is_complex():
            raise TypeError("WAN observation RGB images must be real-valued")
        return image

    def _validate_rgb_motion_observation_binding(self, obs, sequences):
        """Require DINO/geometry RGB to be the exact RGB encoded by WAN VAE."""
        images = obs.get('obs')
        if images is None:
            raise KeyError(
                "online RGB-motion preprocessing requires obs['obs'] so raw "
                "RGB can be bound to the images encoded by the WAN VAE."
            )
        if not isinstance(images, list):
            images = [images]
        if not images:
            raise ValueError("obs['obs'] must contain at least one RGB frame")
        raw_frames = next(iter(sequences.values())).num_frames
        if len(images) != raw_frames:
            raise ValueError(
                "rgb_motion_inputs raw RGB and obs['obs'] must describe the "
                f"same frame sequence: raw T={raw_frames}, obs T={len(images)}."
            )

        camera_order = tuple(self.job_config.obs_cam_keys)
        for frame_index, observation in enumerate(images):
            if not isinstance(observation, dict):
                raise TypeError(
                    f"obs['obs'][{frame_index}] must be a camera mapping"
                )
            missing = [key for key in camera_order if key not in observation]
            if missing:
                raise KeyError(
                    f"obs['obs'][{frame_index}] is missing configured RGB "
                    f"cameras {missing}."
                )
            for camera_key in camera_order:
                sequence_rgb = sequences[camera_key].rgb
                raw_hwc = (
                    sequence_rgb[frame_index].permute(1, 2, 0)
                    if sequence_rgb.shape[1] == 3
                    else sequence_rgb[frame_index]
                ).detach().cpu()
                vae_hwc = self._rgb_motion_observation_image(
                    observation[camera_key],
                    camera_key=camera_key,
                    frame_index=frame_index,
                )
                if tuple(raw_hwc.shape) != tuple(vae_hwc.shape):
                    raise ValueError(
                        "rgb_motion_inputs RGB must be the same image used by "
                        f"the WAN VAE: camera={camera_key!r}, frame={frame_index}, "
                        f"raw shape={tuple(raw_hwc.shape)}, "
                        f"obs shape={tuple(vae_hwc.shape)}."
                    )
                if (torch.is_floating_point(raw_hwc)
                        and not torch.isfinite(raw_hwc).all()) or (
                        torch.is_floating_point(vae_hwc)
                        and not torch.isfinite(vae_hwc).all()):
                    raise ValueError(
                        f"RGB contains non-finite values for camera "
                        f"{camera_key!r}, frame {frame_index}."
                    )
                # Dtype differences such as uint8 versus float32(0..255) are
                # harmless, but any pixel-value difference means two distinct
                # RGB streams and is rejected rather than silently accepted.
                same_pixels = torch.equal(
                    raw_hwc.to(torch.float64), vae_hwc.to(torch.float64))
                if not same_pixels:
                    raise ValueError(
                        "rgb_motion_inputs RGB does not match the RGB encoded "
                        f"by the WAN VAE at camera={camera_key!r}, "
                        f"frame={frame_index}; do not send two different RGB "
                        "sources in obs['rgb_motion_inputs'] and obs['obs']."
                    )

    def _rgb_motion_streaming_vae_is_warm(self):
        """Return whether the video streaming VAE has causal history.

        ``frame_st_id`` cannot answer this question: after cold imagination the
        clean seed is already present in the VAE cache while the first
        grounding request still has ``frame_st_id == 0``.  The causal feature
        cache is the source of truth for the next encoder call.
        """
        streaming_vae = getattr(self, 'streaming_vae', None)
        feat_cache = getattr(streaming_vae, 'feat_cache', None)
        if feat_cache is None:
            raise RuntimeError(
                "online RGB-motion preprocessing cannot determine whether "
                "the streaming WAN VAE is cold: streaming_vae.feat_cache is "
                "unavailable."
            )
        try:
            return any(entry is not None for entry in feat_cache)
        except TypeError as exc:
            raise TypeError(
                "streaming_vae.feat_cache must be an iterable causal cache"
            ) from exc

    def _rgb_motion_vae_temporal_stride(self):
        """Read the causal video stride declared by the serving WAN VAE."""
        streaming_vae = getattr(self, 'streaming_vae', None)
        vae = getattr(streaming_vae, 'vae', None)
        if vae is None:
            vae = getattr(self, 'vae', None)
        config = getattr(vae, 'config', None)
        value = getattr(config, 'scale_factor_temporal', None)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError(
                "online RGB-motion preprocessing requires the loaded WAN VAE "
                "to declare integer config.scale_factor_temporal; got "
                f"{value!r}."
            )
        stride = int(value)
        if stride < 1:
            raise ValueError(
                "WAN VAE config.scale_factor_temporal must be positive, got "
                f"{stride}."
            )
        return stride

    def _rgb_motion_streaming_anchor_indices(
            self, raw_frames, *, streaming_vae_warm=None):
        """Derive raw-frame anchors for the next streaming encode call.

        A fresh WAN causal encoder consumes one seed frame and emits the seed
        latent. Once its feature cache is warm, every latent summarizes one
        complete ``scale_factor_temporal``-frame chunk and is anchored at that
        chunk's causal end. This online layout intentionally differs from the
        offline whole-video layout ``[0, stride, 2*stride, ...]``.
        """
        raw_frames = int(raw_frames)
        if raw_frames < 1:
            raise ValueError(
                f"online RGB-motion raw input must contain frames, got T={raw_frames}"
            )
        if streaming_vae_warm is None:
            streaming_vae_warm = self._rgb_motion_streaming_vae_is_warm()
        elif not isinstance(streaming_vae_warm, (bool, np.bool_)):
            raise TypeError("streaming_vae_warm must be bool when provided")

        if not bool(streaming_vae_warm):
            if raw_frames != 1:
                raise ValueError(
                    "cold streaming WAN VAE RGB-motion input must contain "
                    f"exactly one seed frame (T=1), got T={raw_frames}."
                )
            return torch.tensor([0], dtype=torch.long)

        stride = self._rgb_motion_vae_temporal_stride()
        if raw_frames % stride:
            raise ValueError(
                "warm streaming WAN VAE RGB-motion input length must be "
                "divisible by config.scale_factor_temporal: "
                f"T={raw_frames}, stride={stride}."
            )
        return torch.arange(
            stride - 1, raw_frames, stride, dtype=torch.long)

    def _rgb_motion_validate_streaming_anchors(
            self, declared, expected, *, raw_frames):
        """Validate optional client anchors against the causal VAE schedule."""
        if declared is None:
            return expected
        anchors = self._rgb_motion_integer_tensor(declared, 'anchor_indices')
        if anchors.ndim != 1:
            raise ValueError(
                "rgb_motion_inputs.anchor_indices must be one-dimensional")
        if (anchors < 0).any() or (anchors >= int(raw_frames)).any():
            raise IndexError(
                "rgb_motion_inputs.anchor_indices must address the bound raw "
                f"RGB sequence [0,{raw_frames}), got {anchors.tolist()}."
            )
        if not torch.equal(anchors.detach().cpu(), expected.detach().cpu()):
            raise ValueError(
                "rgb_motion_inputs.anchor_indices must exactly match the "
                "causal streaming WAN VAE anchors for this request: expected "
                f"{expected.tolist()}, got {anchors.detach().cpu().tolist()}."
            )
        return anchors

    def _rgb_motion_observed_frame_count(
            self, obs, *, streaming_vae_warm=None):
        """Infer the declared number of observed WAN rows without model mutation.

        A precomputed sidecar declares the count through ``motion_indices``.
        The raw path derives its count from the causal schedule of the next
        streaming VAE call and only accepts a client ``anchor_indices`` field
        when it exactly matches that schedule. This lets serving fully build
        and canonicalise the semantic sidecar before a streaming VAE or
        prediction cache is advanced.
        """
        payload = self._rgb_motion_precomputed_payload(obs)
        if payload is not None:
            if not isinstance(payload, dict):
                raise TypeError("obs['rgb_motion'] must be a dict")
            if 'motion_indices' not in payload:
                raise KeyError("obs['rgb_motion'] is missing motion_indices")
            indices = self._rgb_motion_integer_tensor(
                payload['motion_indices'], 'motion_indices')
            if indices.dim() == 3 and indices.shape[0] == 1:
                indices = indices[0]
            if indices.dim() != 2:
                raise ValueError(
                    "rgb_motion.motion_indices must be [F,K], got "
                    f"{tuple(indices.shape)}")
            return int(indices.shape[0])

        if not bool(getattr(
                self.job_config, 'rgb_motion_online_preprocess', False)):
            raise KeyError(
                "RGB-motion serving requires obs['rgb_motion'] containing "
                "motion_indices and the independent index "
                "{world_time_id,DINO,NeoForce,observation_flag}. Provide a "
                "precomputed sidecar, or enable rgb_motion_online_preprocess "
                "and send obs['rgb_motion_inputs'] with ordered RGB-D camera "
                "geometry.")
        raw = obs.get('rgb_motion_inputs')
        if raw is None:
            raise KeyError(
                "online RGB-motion preprocessing needs "
                "obs['rgb_motion_inputs'] when no precomputed sidecar exists.")
        _sequences, raw_frames = self._rgb_motion_camera_sequences(
            raw, label="obs['rgb_motion_inputs']")
        expected = self._rgb_motion_streaming_anchor_indices(
            raw_frames, streaming_vae_warm=streaming_vae_warm)
        self._rgb_motion_validate_streaming_anchors(
            raw.get('anchor_indices'), expected, raw_frames=raw_frames)
        return int(expected.numel())

    def _prepare_observed_rgb_motion(
            self, obs, frame_st_id, *, streaming_vae_warm=None):
        """Build one observed sidecar transaction without committing episode state.

        ``_rgb_motion_for_frames`` historically commits the newest semantic row
        and the raw previous-frame support as part of its public helper contract.
        Serving needs a stronger boundary: DINO loading, geometry, canonical
        validation and address checks must all finish before the prediction KV or
        streaming VAE changes.  Temporarily restoring the two episode fields lets
        us reuse that single canonicalisation path without running the producer a
        second time.
        """
        if not bool(getattr(self.job_config, 'use_rgb_motion_tokens', False)):
            return None
        num_frames = self._rgb_motion_observed_frame_count(
            obs, streaming_vae_warm=streaming_vae_warm)
        old_last = getattr(self, '_last_rgb_motion', None)
        old_previous = getattr(self, '_rgb_motion_previous_raw_frames', None)
        try:
            sidecar = self._rgb_motion_for_frames(
                obs,
                num_frames,
                frame_st_id,
                observed=True,
                streaming_vae_warm=streaming_vae_warm,
            )
            self._validate_rgb_motion_cache_feature_contract(sidecar)
            next_last = self._last_rgb_motion
            next_previous = self._rgb_motion_previous_raw_frames
        finally:
            # This preparation is a transaction even when DINO/geometry or the
            # canonical validator raises after constructing an intermediate raw
            # sidecar.
            self._last_rgb_motion = old_last
            self._rgb_motion_previous_raw_frames = old_previous
        return {
            'sidecar': sidecar,
            'num_frames': num_frames,
            'last_rgb_motion': next_last,
            'previous_raw_frames': next_previous,
        }

    def _validate_rgb_motion_cache_feature_contract(self, sidecar):
        """Reject semantic feature-width drift before touching a live KV pool."""
        transformer = getattr(self, 'transformer', None)
        accessor = getattr(transformer, 'get_semantic_cache', None)
        if not callable(accessor):
            # Lightweight test doubles and a freshly constructed server may not
            # expose the inspection hook. Production RGB serving already
            # requires the MoT transformer, where this method is available.
            return
        existing = accessor(self.cache_name, layer=0, valid_only=False)
        if existing is None:
            return
        expected_dino = int(existing['dino'].shape[-1])
        expected_neoforce = int(existing['neoforce'].shape[-1])
        actual_dino = int(sidecar['dino_features'].shape[-1])
        actual_neoforce = int(sidecar['neoforce_features'].shape[-1])
        if (actual_dino, actual_neoforce) != (
                expected_dino, expected_neoforce):
            raise ValueError(
                "RGB-motion semantic feature dimensions changed within the "
                "live KV cache: expected "
                f"DINO/NeoForce=({expected_dino},{expected_neoforce}), got "
                f"({actual_dino},{actual_neoforce}). Reset the episode before "
                "changing index encoders.")

    def _commit_prepared_rgb_motion(self, prepared):
        """Publish a successfully prepared observed sidecar to episode state."""
        if prepared is None:
            return
        self._last_rgb_motion = prepared['last_rgb_motion']
        self._rgb_motion_previous_raw_frames = prepared['previous_raw_frames']

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
        return {
            'video_vae': self._snapshot_streaming_vae_cache(
                getattr(self, 'streaming_vae', None)),
            'tactile_global_vae': self._snapshot_streaming_vae_cache(
                getattr(self, 'tactile_global_vae', None)),
            'tactile_local_vae': self._snapshot_streaming_vae_cache(
                getattr(self, 'tactile_local_vae', None)),
            'tactile_first_frames': getattr(self, 'tactile_first_frames', None),
            'tactile_prev_frames': getattr(self, 'tactile_prev_frames', None),
            'last_tactile_latents': getattr(self, 'last_tactile_latents', None),
            'init_latent': getattr(self, 'init_latent', None),
            'current_observed_kv_index': getattr(self, '_current_observed_kv_index', None),
            'init_kv_index': getattr(self, '_init_kv_index', None),
            'last_observed_video_latent': getattr(
                self, '_last_observed_video_latent', None),
            'last_gen_tactile': getattr(self, '_last_gen_tactile', None),
            'last_gen_tactile_fsid': getattr(
                self, '_last_gen_tactile_fsid', None),
            'delta_smooth_prev': getattr(self, '_delta_smooth_prev', None),
            'last_rgb_motion': getattr(self, '_last_rgb_motion', None),
            'previous_raw_frames': getattr(
                self, '_rgb_motion_previous_raw_frames', None),
            'frame_st_id': getattr(self, 'frame_st_id', 0),
        }

    def _restore_grounding_state(self, snapshot):
        self._restore_streaming_vae_cache(
            getattr(self, 'streaming_vae', None), snapshot['video_vae'])
        self._restore_streaming_vae_cache(
            getattr(self, 'tactile_global_vae', None),
            snapshot['tactile_global_vae'])
        self._restore_streaming_vae_cache(
            getattr(self, 'tactile_local_vae', None),
            snapshot['tactile_local_vae'])
        self.tactile_first_frames = snapshot['tactile_first_frames']
        self.tactile_prev_frames = snapshot['tactile_prev_frames']
        self.last_tactile_latents = snapshot['last_tactile_latents']
        self.init_latent = snapshot['init_latent']
        self._current_observed_kv_index = snapshot['current_observed_kv_index']
        self._init_kv_index = snapshot['init_kv_index']
        self._last_observed_video_latent = snapshot[
            'last_observed_video_latent']
        self._last_gen_tactile = snapshot['last_gen_tactile']
        self._last_gen_tactile_fsid = snapshot['last_gen_tactile_fsid']
        self._delta_smooth_prev = snapshot['delta_smooth_prev']
        self._last_rgb_motion = snapshot['last_rgb_motion']
        self._rgb_motion_previous_raw_frames = snapshot['previous_raw_frames']
        self.frame_st_id = snapshot['frame_st_id']

    def _rgb_motion_from_raw_inputs(
            self, obs, num_frames, frame_st_id, *, streaming_vae_warm=None):
        """Generate one observed canonical sidecar from raw RGB-D inputs."""
        raw = obs.get('rgb_motion_inputs')
        if raw is None:
            raise KeyError(
                "RGB-motion serving received no precomputed obs['rgb_motion'] "
                "and online preprocessing needs obs['rgb_motion_inputs']."
            )
        sequences, raw_frames = self._rgb_motion_camera_sequences(
            raw, label="obs['rgb_motion_inputs']")
        self._validate_rgb_motion_observation_binding(obs, sequences)

        expected_anchors = self._rgb_motion_streaming_anchor_indices(
            raw_frames, streaming_vae_warm=streaming_vae_warm)
        if expected_anchors.numel() != int(num_frames):
            raise ValueError(
                "RGB-motion frame count does not match the next streaming WAN "
                f"VAE call: derived F={expected_anchors.numel()} from raw "
                f"T={raw_frames}, but caller requested F={num_frames}."
            )
        anchors = self._rgb_motion_validate_streaming_anchors(
            raw.get('anchor_indices'), expected_anchors,
            raw_frames=raw_frames)
        if anchors.numel() > 1 and not torch.all(anchors[1:] > anchors[:-1]):
            raise ValueError(
                "rgb_motion_inputs.anchor_indices must be strictly increasing"
            )

        world_times = raw.get('world_time_ids')
        if world_times is None:
            world_times = torch.arange(
                int(frame_st_id), int(frame_st_id) + int(num_frames),
                dtype=torch.long,
            )
        else:
            world_times = self._rgb_motion_integer_tensor(
                world_times, 'world_time_ids')
            if world_times.ndim != 1 or world_times.numel() != int(num_frames):
                raise ValueError(
                    "rgb_motion_inputs.world_time_ids must contain exactly "
                    f"one entry per grounded WAN frame ({num_frames})."
                )
            expected_world_times = torch.arange(
                int(frame_st_id), int(frame_st_id) + int(num_frames),
                dtype=torch.long,
            )
            if not torch.equal(
                    world_times.detach().cpu(), expected_world_times):
                raise ValueError(
                    "rgb_motion_inputs.world_time_ids must equal the server's "
                    "grounded WAN-step coordinates "
                    f"{expected_world_times.tolist()}, got "
                    f"{world_times.detach().cpu().tolist()}."
                )

        previous = None
        explicit_previous = raw.get('previous')
        if explicit_previous is not None:
            previous, _ = self._rgb_motion_camera_sequences(
                explicit_previous,
                label="obs['rgb_motion_inputs']['previous']",
            )
        else:
            previous = getattr(
                self, '_rgb_motion_previous_raw_frames', None)

        first_policy = str(getattr(
            self.job_config,
            'rgb_motion_first_frame_policy',
            'require_previous',
        ))
        if first_policy == 'require_previous' and previous is None:
            raise ValueError(
                "cold-start online RGB-motion preprocessing with "
                "first_frame_policy='require_previous' needs "
                "rgb_motion_inputs['previous']; no prior raw camera frame is "
                "available after startup/reset."
            )

        processor = self._get_rgb_motion_preprocessor()
        sidecar = processor(
            sequences,
            anchor_indices=anchors,
            world_time_ids=world_times,
            previous_frames=previous,
            observation_flag=1,
        )
        if not isinstance(sidecar, dict):
            raise TypeError("RGBMotionSequencePreprocessor must return a dict")
        canonical_names = (
            'motion_indices', 'motion_valid_mask', 'motion_scores',
            'world_time_id', 'dino_features', 'neoforce_features',
            'observation_flag', 'visual_valid', 'tactile_valid')
        missing = [name for name in canonical_names if name not in sidecar]
        if missing:
            raise KeyError(
                f"online RGB-motion preprocessor omitted canonical fields {missing}"
            )

        # Commit only after a successful full preprocessing pass.  A malformed
        # observation must not poison the next call's temporal support.
        self._rgb_motion_previous_raw_frames = (
            self._rgb_motion_capture_anchor_raw_frames(
                sequences, int(anchors[-1].item())))
        return sidecar

    def _rgb_motion_for_frames(
            self, obs, num_frames, frame_st_id, *, observed,
            streaming_vae_warm=None):
        """Build the canonical sparse-RGB sidecar consumed by the transformer.

        A precomputed ``obs['rgb_motion']`` always wins.  If absent, an opted-in
        online producer may build it once from ``obs['rgb_motion_inputs']`` for
        a real observation. During imagination the most recent observed motion
        support/DINO identity is propagated to the requested future frames and
        marked predicted (0); raw preprocessing is never run there or inside a
        diffusion denoising iteration.
        """
        if not bool(getattr(self.job_config, 'use_rgb_motion_tokens', False)):
            return None
        payload = self._rgb_motion_precomputed_payload(obs)
        validate_provenance = payload is not None
        if payload is None:
            if observed and bool(getattr(
                    self.job_config,
                    'rgb_motion_online_preprocess',
                    False)):
                payload = self._rgb_motion_from_raw_inputs(
                    obs,
                    num_frames,
                    frame_st_id,
                    streaming_vae_warm=streaming_vae_warm,
                )
                validate_provenance = True
        if payload is None:
            if not observed and self._last_rgb_motion is not None:
                # A cached observation may contain several grounded frames. It
                # is a fallback support state, not a future trajectory: only
                # the newest observed row should seed every imagined frame.
                # Explicit caller-provided future payloads keep their per-frame
                # rows because they do not take this branch.
                payload = {
                    name: value[-1:]
                    for name, value in self._last_rgb_motion.items()
                }
            else:
                raise KeyError(
                    "RGB-motion serving requires obs['rgb_motion'] containing "
                    "motion_indices and the independent index "
                    "{world_time_id,DINO,NeoForce,observation_flag}. Provide a "
                    "precomputed sidecar, or enable rgb_motion_online_preprocess "
                    "and send obs['rgb_motion_inputs'] with ordered RGB-D camera "
                    "geometry.")
        if not isinstance(payload, dict):
            raise TypeError("obs['rgb_motion'] must be a dict")
        # Internal carry-forward was already validated when it was observed;
        # its compact cache intentionally stores only the canonical 9 fields.
        if validate_provenance:
            self._validate_rgb_motion_payload_provenance(payload)
        if 'motion_indices' not in payload:
            raise KeyError("obs['rgb_motion'] is missing motion_indices")
        if 'dino_features' not in payload:
            raise KeyError("obs['rgb_motion'] is missing dino_features")

        def _cpu_or_device(value, name, dtype=None):
            if dtype == torch.long:
                tensor = self._rgb_motion_integer_tensor(value, name)
            elif dtype == torch.bool:
                tensor = self._rgb_motion_binary_tensor(value, name)
            else:
                tensor = self._rgb_motion_input_tensor(value, name)
                if tensor.is_complex():
                    raise TypeError(f"rgb_motion.{name} must be real-valued")
            return tensor.to(device=self.device, dtype=dtype)

        source_indices = _cpu_or_device(
            payload['motion_indices'], 'motion_indices', torch.long)
        if source_indices.dim() == 3 and source_indices.shape[0] == 1:
            source_indices = source_indices[0]
        if source_indices.dim() != 2:
            raise ValueError(
                "rgb_motion.motion_indices must be [F,K], got "
                f"{tuple(source_indices.shape)}")
        source_frames, tokens_per_frame = source_indices.shape
        source_shape = (source_frames, tokens_per_frame)
        if source_frames < 1:
            raise ValueError("rgb_motion must contain at least one frame")
        if observed and source_frames != num_frames:
            raise ValueError(
                "Observed RGB-motion sidecars must contain exactly one row per "
                "grounded WAN frame: "
                f"sidecar F={source_frames}, grounding F={num_frames}. "
                "Carry-forward/repetition is only valid for predicted frames."
            )
        configured_k = int(getattr(
            self.job_config, 'rgb_motion_max_tokens', tokens_per_frame))
        if configured_k > 0 and tokens_per_frame > configured_k:
            raise ValueError(
                f"rgb_motion K={tokens_per_frame} exceeds configured "
                f"rgb_motion_max_tokens={configured_k}")

        def _frame_select(value, name, *, feature=False, dtype=None,
                          default=None):
            if value is None:
                value = default
            tensor = _cpu_or_device(value, name, dtype)
            if feature:
                if tensor.dim() == 4 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                if tensor.dim() != 3 or tuple(tensor.shape[:2]) != source_shape:
                    raise ValueError(
                        f"rgb_motion.{name} must be [F,K,D], got "
                        f"{tuple(tensor.shape)}")
            else:
                if tensor.dim() == 3 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                if tuple(tensor.shape) != source_shape:
                    raise ValueError(
                        f"rgb_motion.{name} must be [F,K], got "
                        f"{tuple(tensor.shape)}")
            if source_frames == num_frames:
                return tensor
            # Only prediction may resize a support trajectory: observed chunks
            # were required above to provide an exact row for every WAN frame.
            if source_frames > num_frames:
                return tensor[-num_frames:]
            tail = tensor[-1:]
            return torch.cat([tensor, tail.expand(
                num_frames - source_frames, *tail.shape[1:])], dim=0)

        valid = _frame_select(
            payload.get('motion_valid_mask'), 'motion_valid_mask',
            dtype=torch.bool, default=(source_indices >= 0))
        if (source_indices < -1).any():
            raise ValueError(
                "rgb_motion.motion_indices may only use -1 for padding")
        indices = _frame_select(
            source_indices, 'motion_indices', dtype=torch.long)
        valid = valid & (indices >= 0)
        # Preserve the canonical padding sentinel even for permissive online
        # clients that put an arbitrary value in a masked-out slot. This also
        # keeps invalid addresses harmless in every downstream gather.
        indices = indices.masked_fill(~valid, -1)
        patch_size = tuple(int(v) for v in self.job_config.patch_size)
        latent_height = int(self.job_config.height) // 16
        latent_width = (
            int(self.job_config.width) // 16
        ) * len(tuple(self.job_config.obs_cam_keys))
        spatial_tokens = (
            latent_height // patch_size[1]
        ) * (latent_width // patch_size[2])
        if valid.any() and (indices[valid] >= spatial_tokens).any():
            raise IndexError(
                "rgb_motion.motion_indices contains an address outside the "
                f"multi-camera WAN patch grid [0,{spatial_tokens}): "
                f"{indices[valid & (indices >= spatial_tokens)][:8].tolist()}")
        for frame_index in range(num_frames):
            selected = indices[frame_index, valid[frame_index]]
            if selected.numel() != torch.unique(selected).numel():
                raise ValueError(
                    "rgb_motion.motion_indices contains duplicate valid "
                    f"addresses in frame {frame_index}")
        dino = _frame_select(
            payload['dino_features'], 'dino_features', feature=True,
            dtype=torch.float32).masked_fill(~valid[..., None], 0)
        neo_value = payload.get('neoforce_features')
        if neo_value is None:
            neo_value = torch.empty(
                source_frames, tokens_per_frame, 0, device=self.device,
                dtype=dino.dtype)
        neo = _frame_select(
            neo_value, 'neoforce_features', feature=True,
            dtype=torch.float32).masked_fill(~valid[..., None], 0)
        if neo.shape[-1] > 0 and payload.get('tactile_valid') is None:
            raise KeyError(
                "rgb_motion supplies NeoForce features but is missing "
                "tactile_valid; numeric zero cannot encode modality presence")
        visual = _frame_select(
            payload.get('visual_valid'), 'visual_valid', dtype=torch.bool,
            default=(source_indices >= 0)) & valid
        tactile = _frame_select(
            payload.get('tactile_valid'), 'tactile_valid', dtype=torch.bool,
            default=torch.zeros_like(source_indices, dtype=torch.bool)) & valid
        if (valid & ~(visual | tactile)).any():
            raise ValueError("every RGB-motion token needs DINO or NeoForce")
        if visual.any() and dino.shape[-1] == 0:
            raise ValueError(
                "rgb_motion.dino_features must have non-zero width when "
                "visual_valid is true")
        if tactile.any() and neo.shape[-1] == 0:
            raise ValueError(
                "rgb_motion.neoforce_features must have non-zero width when "
                "tactile_valid is true")
        if visual.any() and not torch.isfinite(dino[visual]).all():
            raise ValueError(
                "rgb_motion.dino_features must be finite on visual_valid rows")
        if tactile.any() and not torch.isfinite(neo[tactile]).all():
            raise ValueError(
                "rgb_motion.neoforce_features must be finite on "
                "tactile_valid rows")

        # world_time_id describes the represented world state.  A prediction
        # generated from time t for t+1 is indexed by t+1, not by its creation
        # time.  Server frame_st_id is already the grounded WAN time coordinate.
        world_time = torch.arange(
            int(frame_st_id), int(frame_st_id) + int(num_frames),
            device=self.device, dtype=torch.long)[:, None].expand(
                num_frames, tokens_per_frame).clone()
        world_time.masked_fill_(~valid, -1)
        source_flag = torch.full(
            (num_frames, tokens_per_frame), 1 if observed else 0,
            device=self.device, dtype=torch.long)
        source_flag.masked_fill_(~valid, 0)
        if payload.get('motion_scores') is None:
            # `valid` has already been resized to the requested output frame
            # count; sending it through the source-shape validator again would
            # reject the normal one-observation -> future-chunk case.
            scores = valid.float()
        else:
            scores = _frame_select(
                payload['motion_scores'], 'motion_scores',
                dtype=torch.float32)
        if valid.any() and not torch.isfinite(scores[valid]).all():
            raise ValueError(
                "rgb_motion.motion_scores must be finite on valid rows")
        scores = scores.masked_fill(~valid, 0)
        result = {
            'motion_indices': indices[None],
            'motion_valid_mask': valid[None],
            'motion_scores': scores[None],
            'world_time_id': world_time[None],
            'dino_features': dino[None],
            'neoforce_features': neo[None],
            'observation_flag': source_flag[None],
            'visual_valid': visual[None],
            'tactile_valid': tactile[None],
        }
        if observed:
            # Keep an unbatched copy as the candidate support for the next
            # imagination call.  The prediction call overwrites world time/source.
            self._last_rgb_motion = {
                name: value[0].detach().clone() for name, value in result.items()
            }
        return result

    @staticmethod
    def _concat_rgb_motion(first, second):
        """Concatenate two canonical batched sidecars along world time."""
        if first is None:
            return second
        if second is None:
            return first
        result = {}
        for name in (
            'motion_indices', 'motion_valid_mask', 'motion_scores',
            'world_time_id', 'dino_features', 'neoforce_features',
            'observation_flag', 'visual_valid', 'tactile_valid'):
            left, right = first[name], second[name]
            if left.shape[0] != right.shape[0] or left.shape[2:] != right.shape[2:]:
                raise ValueError(
                    f"cannot concatenate rgb_motion.{name}: "
                    f"{tuple(left.shape)} vs {tuple(right.shape)}")
            result[name] = torch.cat([left, right], dim=1)
        return result

    @staticmethod
    def _overlay_rgb_motion_seed_frames(predicted, observed_seed):
        """Replace leading *frames* only; never use K as a frame count."""
        if observed_seed is None:
            return predicted
        names = (
            'motion_indices', 'motion_valid_mask', 'motion_scores',
            'world_time_id', 'dino_features', 'neoforce_features',
            'observation_flag', 'visual_valid', 'tactile_valid')
        missing = [
            name for name in names
            if name not in predicted or name not in observed_seed
        ]
        if missing:
            raise KeyError(
                f"cannot overlay RGB-motion seed; missing fields {missing}")
        predicted_frames = int(predicted['motion_indices'].shape[1])
        seed_frames = int(observed_seed['motion_indices'].shape[1])
        copy_frames = min(predicted_frames, seed_frames)
        for name in names:
            target = predicted[name]
            source = observed_seed[name]
            if target.ndim < 3 or source.ndim != target.ndim:
                raise ValueError(
                    f"rgb_motion.{name} must be batched [B,F,K,...] for "
                    "seed overlay")
            if target.shape[0] != source.shape[0] or target.shape[2:] != source.shape[2:]:
                raise ValueError(
                    f"cannot overlay rgb_motion.{name}: "
                    f"{tuple(source.shape)} onto {tuple(target.shape)}")
            target[:, :copy_frames, ...] = source[:, :copy_frames, ...]
        return predicted

    @staticmethod
    def _repeat_rgb_motion_batch(input_dict, repeats):
        for key in (
            'motion_indices', 'motion_valid_mask', 'motion_scores',
            'world_time_id', 'dino_features', 'neoforce_features',
            'observation_flag', 'visual_valid', 'tactile_valid'):
            if key in input_dict:
                value = input_dict[key]
                input_dict[key] = value.repeat(
                    repeats, *([1] * (value.dim() - 1)))

    def _sparse_video_prediction_to_dense(self, prediction, template,
                                          motion_input):
        """Scatter sparse proj_out cells into a zero dense velocity field."""
        B, C, F_lat, H_lat, W_lat = template.shape
        flat_indices, flat_valid = self._rgb_motion_flat_address(
            template, motion_input)
        zero = torch.zeros_like(template)
        sparse = self.rgb_patch_gather(
            zero, indices=flat_indices, valid_mask=flat_valid)
        patch_volume = int(np.prod(tuple(self.job_config.patch_size)))
        expected = flat_indices.shape[1] * patch_volume
        if prediction.shape != (B, expected, C):
            raise ValueError(
                "sparse video prediction shape mismatch: expected "
                f"{(B, expected, C)}, got {tuple(prediction.shape)}")
        # proj_out sequence order is (patch-cell, channel), whereas raw WAN
        # patchify order is (channel, patch-cell).
        raw_values = prediction.reshape(
            B, flat_indices.shape[1], patch_volume, C).permute(
                0, 1, 3, 2).reshape(B, flat_indices.shape[1], -1)
        raw_values = raw_values.masked_fill(~flat_valid[..., None], 0)
        return self.rgb_patch_scatter(
            sparse, base_latents=zero, values=raw_values)

    def _rgb_motion_flat_address(self, template, motion_input):
        """Map frame-local `[B,F,K]` indices to the flattened WAN grid."""
        B, _, F_lat, H_lat, W_lat = template.shape
        indices = motion_input['motion_indices']
        valid = motion_input['motion_valid_mask']
        if indices.shape[0] != B:
            if indices.shape[0] == 1:
                indices = indices.expand(B, -1, -1)
                valid = valid.expand(B, -1, -1)
            else:
                raise ValueError("motion/prediction batch mismatch")
        p_t, p_h, p_w = tuple(self.job_config.patch_size)
        Fp, Hp, Wp = F_lat // p_t, H_lat // p_h, W_lat // p_w
        if indices.shape[1] != Fp:
            raise ValueError(
                f"motion F={indices.shape[1]} does not match latent grid F={Fp}")
        spatial = Hp * Wp
        frame = torch.arange(Fp, device=indices.device)[None, :, None]
        if (valid & ((indices < 0) | (indices >= spatial))).any():
            bad = indices[valid & ((indices < 0) | (indices >= spatial))]
            raise ValueError(
                f"RGB-motion index outside [0,{spatial}): {bad[:8].tolist()}")
        flat_indices = (frame * spatial + indices.clamp_min(0)).reshape(B, -1)
        flat_valid = valid.reshape(B, -1)
        return flat_indices, flat_valid

    def _sparse_canvas_from_dense(self, values, base, motion_input):
        """Copy only selected patches from `values` onto a dense `base` canvas."""
        if values.shape != base.shape:
            raise ValueError(
                f"sparse canvas tensors differ: {values.shape} vs {base.shape}")
        flat_indices, flat_valid = self._rgb_motion_flat_address(
            values, motion_input)
        sparse = self.rgb_patch_gather(
            values, indices=flat_indices, valid_mask=flat_valid)
        return self.rgb_patch_scatter(sparse, base_latents=base)

    def _rgb_motion_background(self, template):
        """Broadcast the newest real observation over a future latent chunk."""
        source = self._last_observed_video_latent
        if source is None:
            source = self.init_latent
        if source is None:
            return torch.zeros_like(template)
        source = source.to(device=template.device, dtype=template.dtype)
        if (source.shape[0] != template.shape[0]
                or source.shape[1] != template.shape[1]
                or source.shape[-2:] != template.shape[-2:]):
            raise ValueError(
                "observed video latent cannot seed sparse canvas: "
                f"{tuple(source.shape)} vs {tuple(template.shape)}")
        return source[:, :, -1:].expand(
            -1, -1, template.shape[2], -1, -1).clone()

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              tactile_latents=None,
                              rgb_motion=None):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        # One source of truth: this grid must use the same patch geometry as the
        # loaded transformer, cache sizing, sparse gather/scatter, and the
        # train/serve consistency check.  A private-call default used to silently
        # construct a (1,2,2) grid even when job_config selected another layout.
        patch_size = tuple(self.job_config.patch_size)
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0
            if tactile_latents is not None:
                input_dict['latent_res_lst']['tactile_global_latent'] = (
                    tactile_latents['tactile_global_latent']
                )
                input_dict['latent_res_lst']['tactile_sensor_ids'] = (
                    tactile_latents['tactile_sensor_ids']
                )
            if rgb_motion is not None:
                input_dict['latent_res_lst'].update(rgb_motion)

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if tactile_latents is not None:
                input_dict['action_res_lst'].update(tactile_latents)

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
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
        logger.info('Reset.')
        self._validate_rgb_motion_server_config(self.job_config)
        cache_policy = getattr(self.job_config, 'kv_cache_policy', 'fifo')
        if cache_policy not in ('fifo', 'global'):
            raise ValueError("kv_cache_policy must be 'fifo' or 'global'")
        if cache_policy == 'global':
            from n0_twam.models.global_kv_retention import RetentionConfig
            RetentionConfig(**dict(getattr(self.job_config, 'kv_retention', {})))
            if bool(getattr(self.job_config, 'use_rgb_motion_tokens', False)):
                raise ValueError("global index mode requires use_rgb_motion_tokens=False")
            if self.job_config.patch_size[0] != 1:
                raise ValueError("global RGB index currently requires temporal patch_size=1")
            _, ph, pw = self.job_config.patch_size
            if self.job_config.height % (16 * ph) or self.job_config.width % (16 * pw):
                raise ValueError('global RGB index requires camera sizes divisible by WAN patch stride')
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self._current_observed_kv_index = None
        self._init_kv_index = None
        self.last_tactile_latents = None   # mirror init_latent: tactile cond reuse slot
        self._last_rgb_motion = None
        self._last_observed_video_latent = None
        self._rgb_motion_previous_raw_frames = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        self._reset_tactile_state()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
            self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        if bool(getattr(self.job_config, 'use_rgb_motion_tokens', False)):
            if self.job_config.frame_chunk_size % patch_size[0]:
                raise ValueError(
                    "frame_chunk_size must be divisible by temporal patch_size")
            max_motion_tokens = int(getattr(
                self.job_config, 'rgb_motion_max_tokens', 32))
            if max_motion_tokens <= 0:
                # Variable-K sidecars cannot determine a safe fixed streaming
                # cache capacity.  Fall back to the dense upper bound.
                max_motion_tokens = (
                    (self.latent_height // patch_size[1])
                    * (self.latent_width // patch_size[2]))
            latent_token_per_chunk = (
                self.job_config.frame_chunk_size // patch_size[0]
            ) * max_motion_tokens
        else:
            latent_token_per_chunk = (
                self.job_config.frame_chunk_size
                * self.latent_height * self.latent_width
            ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        if self.job_config.tactile_keys:
            tactile_latent_height = int(getattr(self.job_config, 'tactile_latent_height', 8))
            tactile_latent_width = int(getattr(self.job_config, 'tactile_latent_width', 8))
            tactile_token_per_chunk = (
                len(self.job_config.tactile_keys) *
                self.job_config.frame_chunk_size *
                tactile_latent_height *
                tactile_latent_width
            ) // (patch_size[0] * patch_size[1] * patch_size[2])
            latent_token_per_chunk += tactile_token_per_chunk
            action_token_per_chunk += tactile_token_per_chunk
            logger.info("tactile cache tokens per chunk: %d", tactile_token_per_chunk)
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        if self._global_index_enabled():
            self.transformer.configure_global_retention(
                self.cache_name, **dict(getattr(self.job_config, 'kv_retention', {})))
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt (bare reset falls back to the config prompt)
        if prompt is None:
            prompt = getattr(self.job_config, 'prompt', None)
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.use_cfg,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
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
        frame_chunk_size = self.job_config.frame_chunk_size
        rgb_motion_enabled = bool(getattr(
            self.job_config, 'use_rgb_motion_tokens', False))
        seed_rgb_motion = None
        prepared_seed_rgb_motion = None
        if frame_st_id == 0:
            # Complete DINO/geometry work and canonical address validation before
            # advancing any streaming encoder.  The preparation transaction keeps
            # `_last_rgb_motion` and raw previous-frame support unchanged until the
            # encoded WAN row count has also been checked.
            if rgb_motion_enabled:
                prepared_seed_rgb_motion = self._prepare_observed_rgb_motion(
                    obs, frame_st_id, streaming_vae_warm=False)
            # Preprocessing above is deliberately side-effect free.  Only once
            # its DINO/geometry/canonical checks succeed may a cold request
            # discard the previous streaming state and advance the encoders.
            self.streaming_vae.clear_cache()
            self._reset_tactile_state()
            # Cold seed — mirror video's init_latent: encode the current tactile once
            # (advances the persistent tactile streaming VAE, just like _encode_obs does
            # for streaming_vae), then commit it so later plain-infers can reuse it.
            # The persistent tactile VAE feat_cache must be FRESH for this 1-frame cold
            # seed (WAN avg_shortcut needs Rep padding or kernel(3)>input crashes); the
            # warm-cache discipline only holds for the >=3-frame kv_cache groundings after.
            tactile_latents = (
                self._encode_tactile_obs(obs) if self.job_config.tactile_keys else None
            )
            self.last_tactile_latents = tactile_latents
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
            if self._global_index_enabled():
                self._init_kv_index = self._current_observed_kv_index
            self._last_observed_video_latent = init_latent[:, :, -1:].detach().clone()
            if rgb_motion_enabled:
                seed_frames = init_latent.shape[2] // self.job_config.patch_size[0]
                if prepared_seed_rgb_motion['num_frames'] != seed_frames:
                    raise ValueError(
                        "cold RGB-motion/WAN latent frame mismatch: prepared "
                        f"F={prepared_seed_rgb_motion['num_frames']}, encoded "
                        f"F={seed_frames}")
                seed_rgb_motion = prepared_seed_rgb_motion['sidecar']
                self._commit_prepared_rgb_motion(prepared_seed_rgb_motion)
        else:
            # Mid-episode plain-infer: do NOT re-encode (that double-fed the streaming
            # VAE and corrupted the temporal grid). Reuse the tactile latent committed by
            # the last compute_kv_cache — video does the same here (no obs encode; it
            # reads the transformer KV cache).
            tactile_latents = self.last_tactile_latents

        rgb_motion = None
        if rgb_motion_enabled:
            if frame_chunk_size % self.job_config.patch_size[0]:
                raise ValueError(
                    "frame_chunk_size must be divisible by temporal patch_size")
            motion_frames = frame_chunk_size // self.job_config.patch_size[0]
            rgb_motion = self._rgb_motion_for_frames(
                obs, motion_frames, frame_st_id, observed=False)
            # The cold chunk contains an actually observed/clamped seed at its
            # front, followed by imagined frames.  Keep that distinction in
            # the independent source index instead of labelling the whole
            # chunk predicted merely because update_cache==1.
            if seed_rgb_motion is not None:
                rgb_motion = self._overlay_rgb_motion_seed_frames(
                    rgb_motion, seed_rgb_motion)

        # TACTILE DENOISE: co-generate GlobalTactile in the video loop (training-
        # consistent, default on); off = condition on the raw observed tactile.
        tactile_denoise = bool(getattr(self.job_config, 'server_tactile_denoise', False))
        tactile_g = None
        if tactile_denoise and tactile_latents is not None:
            # noisy GlobalTactile latent to denoise — same shape as the encoded
            # global latent (1, S, 48, F_lat, H_lat, W_lat).
            tactile_g = torch.randn_like(tactile_latents['tactile_global_latent'])

        latent_noise = torch.randn(1,
                                   48,
                                   frame_chunk_size,
                                   self.latent_height,
                                   self.latent_width,
                                   device=self.device,
                                   dtype=self.dtype)
        if rgb_motion is None:
            latents = latent_noise
        else:
            # Static cells are the most recent real world state.  Only selected
            # moving patches start from diffusion noise; zero scattered
            # velocity then keeps unselected cells exactly on this background.
            latents = self._sparse_canvas_from_dense(
                latent_noise, self._rgb_motion_background(latent_noise),
                rgb_motion)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

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

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        video_index_cursor = (self.transformer.global_cache_cursor(self.cache_name)
                              if self._predicted_dino_enabled() else None)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop (co-generates GlobalTactile when enabled)
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                    tactile_latents=tactile_latents,
                    rgb_motion=rgb_motion)

                if self._global_index_enabled():
                    from n0_twam.preprocessing.kv_index import prediction_index
                    count = input_dict['latent_res_lst']['grid_id'].shape[-1]
                    input_dict['latent_res_lst']['kv_index'] = prediction_index(
                        count, self.device,
                        self._init_kv_index if frame_st_id == 0 else None)

                # inject the current noisy GlobalTactile so the model denoises it
                # alongside the video (same timestep t, frame-aligned co-generation).
                if tactile_g is not None:
                    input_dict['latent_res_lst']['tactile_noisy_latent'] = tactile_g
                    input_dict['latent_res_lst']['tactile_timesteps'] = (
                        torch.ones([tactile_g.shape[1] * tactile_g.shape[3]],
                                   device=self.device) * t)

                model_input = self._repeat_input_for_cfg(
                    input_dict['latent_res_lst'])
                out = self.transformer(
                    model_input,
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)
                if isinstance(out, tuple):
                    video_noise_pred, tactile_noise_pred = out
                else:
                    video_noise_pred, tactile_noise_pred = out, None

                if not last_step or video_step != -1:
                    if rgb_motion is None:
                        video_noise_pred = data_seq_to_patch(
                            self.job_config.patch_size, video_noise_pred,
                            frame_chunk_size, self.latent_height,
                            self.latent_width,
                            batch_size=2 if self.use_cfg else 1)
                    else:
                        pred_batch = video_noise_pred.shape[0]
                        pred_template = latents.expand(
                            pred_batch, -1, -1, -1, -1)
                        video_noise_pred = self._sparse_video_prediction_to_dense(
                            video_noise_pred, pred_template, model_input)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)
                    # step the GlobalTactile latent with the SAME scheduler (same
                    # snr_shift as video in training), drop CFG duplicate if present.
                    if tactile_g is not None and tactile_noise_pred is not None:
                        _pre = float(tactile_g.float().std())
                        tac_pred = self._tactile_pred_to_latent(
                            tactile_noise_pred, tactile_g)
                        tactile_g = self.scheduler.step(
                            tac_pred, t, tactile_g, return_dict=False)
                        logger.info("[tac-step] t=%s pre_std=%.4f post_std=%.4f pred_std=%.4f",
                                    int(t) if hasattr(t, '__int__') else t, _pre,
                                    float(tactile_g.float().std()), float(tac_pred.float().std()))

                if frame_st_id == 0:
                    latents[:, :, 0:1] = latent_cond

            if video_index_cursor is not None:
                handle = self.transformer.video_index_handle(self.cache_name, video_index_cursor)
                self._backfill_predicted_video_index(latents, frame_st_id, handle)

            # video loop done: if we co-generated GlobalTactile, the action loop below
            # conditions on the GENERATED tactile (not the observed residual) — the
            # "predict tactile -> act on predicted touch" flow. NOTE: this only affects
            # the in-chunk action pass; _compute_kv_cache re-encodes the OBSERVED
            # tactile for the cross-chunk cache (so the cache stays grounded in reality).
            if tactile_g is not None and tactile_latents is not None:
                # diagnostic proof the tactile was actually denoised (not pass-through):
                # gen_std near randn(1.0) = never denoised (BUG); small = denoised.
                logger.info(
                    "[tactile-denoise] generated GlobalTactile: gen_std=%.4f gen_mean=%.4f",
                    float(tactile_g.float().std()), float(tactile_g.float().mean()))
                # [tactile-pred-eval] stash this chunk's GENERATED future tactile so the
                # NEXT compute_kv_cache (real tactile observed AFTER executing the actions
                # of this chunk) can score the prediction. This is the correct, time-
                # shifted comparison: predicted_future vs real_observed_after_execution.
                self._last_gen_tactile = tactile_g.detach().clone()
                self._last_gen_tactile_fsid = int(frame_st_id)
                tactile_latents = dict(tactile_latents)
                tactile_latents['tactile_global_latent'] = tactile_g

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                if frame_st_id != 0:
                    action_cond = None
                else:
                    # cold-seed frame0 mode. delta: zeros == "stay" (correct, forced).
                    # absolute (cfg.cold_seed_mode): free = do NOT clamp, let the model
                    # denoise frame0 (its learned cold-start; base loader trained frame0
                    # with loss ON, so this is the training-consistent mode — DEFAULT,
                    # ablation-validated); current_state = clean-clamp normalized current
                    # pose (semantically nice but never seen in training); zeros =
                    # original (de-normalizes to q01/q99 midpoint = OOD).
                    _cs_mode = ("zeros" if self._uses_pi05_delta_actions()
                                else str(getattr(self.job_config, 'cold_seed_mode', 'free')).lower())
                    _cs_val = obs.get("current_state")
                    if _cs_mode == "free":
                        action_cond = None
                    elif _cs_mode == "current_state" and _cs_val is not None:
                        _cs = np.asarray(_cs_val, dtype=np.float32).reshape(-1)
                        _adim = int(self.job_config.action_dim)
                        if _cs.shape[0] < _adim:
                            _cs = np.pad(_cs, (0, _adim - _cs.shape[0]))
                        _cs_chunk = np.repeat(_cs[:_adim].reshape(-1, 1, 1), self.action_per_frame, axis=2)
                        action_cond = self.preprocess_action(_cs_chunk).to(device=self.device, dtype=self.dtype)
                    else:
                        action_cond = torch.zeros(
                            [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                            device=self.device, dtype=self.dtype)

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id,
                    tactile_latents=tactile_latents)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        current_state = obs.get('current_state')
        if current_state is None and 'state' in obs:
            current_state = obs['state']
        actions = self.postprocess_action(actions, current_state=current_state,
                                          cold_first_frame=(frame_st_id == 0))
        torch.cuda.empty_cache()
        return actions, latents

    def _execute_grounding_transaction(
            self, obs, *, rgb_motion_enabled, initial_rgb_motion,
            initial_latent, request_frame_st_id, grounding_frame_start,
            seed_is_already_cached, prepared_current_rgb_motion):
        """Advance encoders and write both grounding experts inside one transaction."""
        rgb_motion = None
        # Real observations append new KV. Existing predicted KV and its index
        # remain unchanged; only normal capacity eviction or episode reset can
        # remove them, not the arrival of an observation.
        save_async(obs['obs'], os.path.join(
            self.exp_save_root, f'obs_data_{request_frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        dense_index = getattr(self, '_current_observed_kv_index', None)

        if request_frame_st_id == 0 and not seed_is_already_cached:
            # Legacy dense path and defensive direct-grounding path also encode
            # the initial real condition. No existing prediction is removed.
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent
            if self._global_index_enabled():
                from n0_twam.preprocessing.kv_index import concat_indices
                dense_index = concat_indices(self._init_kv_index, dense_index)

        if rgb_motion_enabled:
            p_t = self.job_config.patch_size[0]
            if latent_model_input.shape[2] % p_t:
                raise ValueError(
                    "grounding latent frames must be divisible by temporal patch_size")
            total_motion_frames = latent_model_input.shape[2] // p_t
            initial_motion_frames = (
                0 if initial_latent is None else initial_latent.shape[2] // p_t)
            current_motion_frames = total_motion_frames - initial_motion_frames
            if prepared_current_rgb_motion is None:
                raise RuntimeError("RGB-motion observation was not prepared")
            if prepared_current_rgb_motion['num_frames'] != current_motion_frames:
                raise ValueError(
                    "grounding RGB-motion/WAN latent frame mismatch: prepared "
                    f"F={prepared_current_rgb_motion['num_frames']}, encoded "
                    f"F={current_motion_frames}")
            current_rgb_motion = prepared_current_rgb_motion['sidecar']
            rgb_motion = self._concat_rgb_motion(
                initial_rgb_motion, current_rgb_motion)
            if rgb_motion is None:
                raise RuntimeError(
                    "prepared RGB-motion observation produced no canonical sidecar")

        action_anchor_state = obs.get(
            'action_anchor_state', obs.get('current_state'))
        action_format = obs.get(
            'state_action_format', obs.get('action_format'))
        action_model_input = self.preprocess_action(
            obs['state'],
            action_anchor_state=action_anchor_state,
            action_format=action_format,
            cold_first_frame=(request_frame_st_id == 0),
        )
        if seed_is_already_cached:
            # The client does not execute cold action frame 0: it is the
            # current-pose seed. Since its paired video seed is already in the
            # semantic cache, ground only the continuation at t=1... .
            if action_model_input.shape[2] <= 1:
                raise ValueError(
                    "cold RGB-motion grounding needs at least one executed "
                    "action frame after the seed")
            action_model_input = action_model_input[:, :, 1:]
        action_model_input = action_model_input.to(latent_model_input)
        tactile_latents = self._encode_tactile_obs(obs)

        # Diagnostic only: compare the last generated tactile chunk with the
        # newly observed tactile, without changing the grounding inputs.
        if self._last_gen_tactile is not None and tactile_latents is not None:
            _gen = self._last_gen_tactile.float()
            _real = tactile_latents['tactile_global_latent'].float()
            if _gen.shape == _real.shape:
                _err = (_gen - _real).abs().mean().item()
                _rstd = _real.std().item()
                logger.info(
                    "[tactile-pred-eval] predicted(fsid=%s) vs real-observed-after-exec: "
                    "|pred-real|mean=%.4f  pred_std=%.4f  real_std=%.4f  rel=%.3f",
                    self._last_gen_tactile_fsid, _err, _gen.std().item(), _rstd,
                    _err / (_rstd + 1e-6))
            else:
                logger.info(
                    "[tactile-pred-eval] shape mismatch pred=%s real=%s (skip)",
                    tuple(_gen.shape), tuple(_real.shape))

        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(
            latent_model_input,
            action_model_input,
            frame_st_id=grounding_frame_start,
            tactile_latents=tactile_latents,
            rgb_motion=rgb_motion)
        if self._global_index_enabled():
            input_dict['latent_res_lst']['kv_index'] = dense_index
            # Optional pre-aligned NeoForce/DINO for the separate tactile tail.
            # They never enter the tactile latent/content projection.
            if 'tactile_kv_index' in obs:
                input_dict['latent_res_lst']['tactile_kv_index'] = obs['tactile_kv_index']
                input_dict['action_res_lst']['tactile_kv_index'] = obs['tactile_kv_index']
        last_observed_video_latent = (
            latent_model_input[:, :, -1:].detach().clone())

        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False)

            self.transformer(
                self._repeat_input_for_cfg(input_dict['action_res_lst']),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True)
        return latent_model_input, tactile_latents, last_observed_video_latent

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        rgb_motion = None
        rgb_motion_enabled = bool(getattr(
            self.job_config, 'use_rgb_motion_tokens', False))
        initial_rgb_motion = None
        initial_latent = None
        request_frame_st_id = int(getattr(self, 'frame_st_id', 0))
        grounding_frame_start = request_frame_st_id
        seed_is_already_cached = False
        if request_frame_st_id == 0:
            initial_latent = getattr(self, 'init_latent', None)
            seed_is_already_cached = (
                (rgb_motion_enabled and self._last_rgb_motion is not None)
                or (self._global_index_enabled()
                    and getattr(self, '_init_kv_index', None) is not None))
            if seed_is_already_cached:
                # The cold generation pass committed its clean-clamped first
                # frame as an observed semantic token, so do not append the
                # seed a second time. Start the
                # newly grounded continuation immediately after it.
                grounding_frame_start = (
                    self.init_latent.shape[2]
                    // self.job_config.patch_size[0])
                initial_latent = None
            if (rgb_motion_enabled and initial_latent is not None
                    and self._last_rgb_motion is not None):
                initial_rgb_motion = {
                    name: value[None].detach().clone()
                    for name, value in self._last_rgb_motion.items()
                }

        # DINO loading/inference, camera geometry, raw RGB binding and canonical
        # sidecar validation are deliberately completed before the streaming
        # WAN VAE advances or any new KV is appended. Keep the candidate episode
        # state private until the encoded temporal length is known to agree.
        prepared_current_rgb_motion = None
        if rgb_motion_enabled:
            initial_motion_frames = (
                0 if initial_latent is None
                else initial_latent.shape[2] // self.job_config.patch_size[0])
            # This must come from the causal encoder cache, not frame_st_id.
            # Cold imagination leaves a warm seed cache while the first
            # grounding request still has frame_st_id == 0.
            raw_payload = self._rgb_motion_precomputed_payload(obs)
            streaming_vae_warm = (
                None if raw_payload is not None
                else self._rgb_motion_streaming_vae_is_warm()
            )
            prepared_current_rgb_motion = self._prepare_observed_rgb_motion(
                obs,
                grounding_frame_start + initial_motion_frames,
                streaming_vae_warm=streaming_vae_warm,
            )

        grounding_snapshot = self._snapshot_grounding_state()
        transaction_factory = getattr(
            self.transformer, 'cache_transaction', None)
        cache_transaction = (
            transaction_factory(self.cache_name)
            if callable(transaction_factory) else nullcontext()
        )
        try:
            # The transaction starts before encoding and appending. Encoder/cache
            # snapshots make every later failure retry-safe, including a WAN
            # temporal-length mismatch or an action pass that fails after video.
            with cache_transaction:
                (
                    latent_model_input,
                    tactile_latents,
                    last_observed_video_latent,
                ) = self._execute_grounding_transaction(
                    obs,
                    rgb_motion_enabled=rgb_motion_enabled,
                    initial_rgb_motion=initial_rgb_motion,
                    initial_latent=initial_latent,
                    request_frame_st_id=request_frame_st_id,
                    grounding_frame_start=grounding_frame_start,
                    seed_is_already_cached=seed_is_already_cached,
                    prepared_current_rgb_motion=prepared_current_rgb_motion,
                )
        except BaseException:
            self._restore_grounding_state(grounding_snapshot)
            raise

        # Publish episode-level values only after both transformer passes and
        # their cache transaction have committed successfully.
        self.last_tactile_latents = tactile_latents
        self._last_observed_video_latent = last_observed_video_latent
        self._commit_prepared_rgb_motion(prepared_current_rgb_motion)
        torch.cuda.empty_cache()
        self.frame_st_id = grounding_frame_start + latent_model_input.shape[2]

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
    
    config = TWAM_CONFIGS[args.config_name]
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
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
