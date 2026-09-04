# Copyright 2025-2026 NeoteAI Team. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import json
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.video_utils import decode_video_frames
import logging

def recursive_find_file(directory, filename='info.json'):
    result = []
    ignored_dirs = {"data", "videos", "latents", ".cache", "__pycache__"}
    try:
        for root, dirs, files in os.walk(directory, followlinks=True):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    # Tolerate broken/in-flight repos (conversion interrupted mid-episode,
    # parquet missing, meta inconsistent): skip with a warning instead of
    # killing the whole multi-dataset init. Essential for train-while-convert.
    try:
        return LatentLeRobotDataset(
            repo_id=repo_id,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Skipping unusable repo %s: %s", repo_id, exc)
        return None

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    repo_list = sorted(repo_list)
    if not repo_list:
        return []
    num_init_worker = max(1, min(int(num_init_worker), len(repo_list)))
    if num_init_worker <= 1 or len(repo_list) <= 1:
        datasets_out_lst = [construct_func(repo_id) for repo_id in repo_list]
    else:
        with Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)
    skipped = sum(1 for d in datasets_out_lst if d is None)
    if skipped:
        logging.warning("construct_lerobot: skipped %d unusable repos", skipped)
    return [d for d in datasets_out_lst if d is not None]

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        num_init_worker = int(getattr(config, "num_init_worker", num_init_worker))
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        try:
            self.meta = LeRobotDatasetMetadata(
                self.repo_id, self.root, self.revision, force_cache_sync=False
            )
        except Exception:
            # LeRobot v3.0 (+ lerobot>=0.3) rejects a full-path repo_id ("must be
            # 'repo_name'") and tries to resolve a pinned revision via the hub.
            # Retry with the basename and no revision -> uses the local snapshot.
            # v2.1 repos succeed on the first call above, so they are untouched.
            self.meta = LeRobotDatasetMetadata(
                Path(self.repo_id).name, self.root, None, force_cache_sync=False
            )

        self.repo_name = Path(self.repo_id).name
        self.config = config
        per_repo_obs_cam_keys = getattr(config, 'per_repo_obs_cam_keys', None) or {}
        self.used_video_keys = list(per_repo_obs_cam_keys.get(self.repo_name, config.obs_cam_keys))
        # Mixed-robot pretraining: tactile sensor sets differ per repo (4/2/0
        # streams). per_repo_tactile_keys overrides the global list; an empty
        # list means this repo has no tactile at all.
        per_repo_tactile_keys = getattr(config, 'per_repo_tactile_keys', None) or {}
        self.used_tactile_keys = list(
            per_repo_tactile_keys.get(self.repo_name, getattr(config, 'tactile_keys', []))
        )
        self.has_tactile_condition = bool(self.used_tactile_keys)
        # When True, episodes whose tactile latents are missing fall back to the
        # CFG tactile-drop path (model's zero-anchor keeps grads sane) instead of
        # raising — required when tactile-less repos mix into pretraining.
        self.tactile_optional = bool(getattr(config, 'tactile_optional', False))
        self.synthetic_tactile_data = bool(getattr(config, 'synthetic_tactile_data', False))
        self.tactile_channels = int(getattr(config, 'tactile_in_channels', 3))
        # RGB-motion sidecars are deliberately opt-in.  In particular, do not
        # probe the filesystem or add sample keys while this is False: the old
        # dense-latent data path must remain byte-for-byte compatible.
        self.use_rgb_motion_tokens = bool(
            getattr(config, 'use_rgb_motion_tokens', False)
        )
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(f"Incomplete local LeRobot dataset under {self.root}") from exc
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False).detach()
        self.cfg_prob = config.cfg_prob
        per_repo_used_action_channel_ids = getattr(config, 'per_repo_used_action_channel_ids', None) or {}
        self.used_action_channel_ids = list(
            per_repo_used_action_channel_ids.get(
                self.repo_name,
                getattr(config, 'used_action_channel_ids', []),
            )
        )
        if self.used_action_channel_ids:
            action_dim = int(getattr(config, 'action_dim', 30))
            inverse_ids = [len(self.used_action_channel_ids)] * action_dim
            for i, j in enumerate(self.used_action_channel_ids):
                inverse_ids[j] = i
            self.inverse_used_action_channel_ids = inverse_ids
        else:
            self.inverse_used_action_channel_ids = list(config.inverse_used_action_channel_ids)
        # per-robot (embodiment) norm: per_repo_norm_stat maps repo basename
        # ("arx5") or its robot base ("ur" for "ur_3cam") to {q01,q99}; falls
        # back to the global norm_stat when absent.
        #
        # robot_base must find the robot token ANYWHERE in repo_name, not just the
        # first token: some repos are named "<task>_<robot>" (e.g.
        # "<task>_<robot>_<n>cam" variants) where split("_")[0] is
        # the TASK ("stack"/"scoop") and would miss the per-robot stat -> silent
        # global fallback (wrong scale, esp. grippers).
        # So we match any known-robot key (from per_repo_norm) appearing as a token.
        _norm_stat = config.norm_stat
        _per_repo_norm = getattr(config, 'per_repo_norm_stat', None) or {}
        if _per_repo_norm:
            _tokens = set(self.repo_name.split("_"))
            _known_robots = sorted(k for k in _per_repo_norm if k)  # drop '' key
            _robot_base = next((k for k in _known_robots if k in _tokens),
                               self.repo_name.split("_")[0])
            _norm_stat = _per_repo_norm.get(
                self.repo_name, _per_repo_norm.get(_robot_base, _norm_stat))
        self.q01 = np.array(_norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(_norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self._hf_tactile_view = None
        available_columns = set(getattr(self.hf_dataset, "column_names", []))
        tactile_columns = [key for key in self.used_tactile_keys if key in available_columns]
        if self.has_tactile_condition and tactile_columns:
            self._hf_tactile_view = self.hf_dataset.with_format(
                type='torch',
                columns=tactile_columns,
                output_all_columns=False,
            )
        if self.has_tactile_condition and self._hf_tactile_view is None and self.synthetic_tactile_data:
            logging.warning(
                "Using synthetic tactile videos for training. "
                "Fake tactile streams are enabled because tactile columns were not found in dataset %s.",
                self.repo_id,
            )
        self.filter_mismatched_latents = bool(
            getattr(config, 'filter_mismatched_latents', True)
        )
        self._latent_frame_count_cache = {}
        self._rgb_motion_meta_cache = {}
        self._meta_filter_counts = {}
        self.parse_meta()

    def _episode_chunk_candidates(self, episode_index: int) -> list[int]:
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        candidates = [episode_chunk]
        if episode_chunk != 0:
            candidates.append(0)
        return candidates

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = []
        for ep_idx in episodes:
            data_path = self.meta.get_data_file_path(ep_idx)
            full_path = self.root / data_path
            if not full_path.is_file():
                fallback = self.root / "data" / "chunk-000" / Path(data_path).name
                if fallback.is_file():
                    data_path = fallback.relative_to(self.root)
            fpaths.append(str(data_path))
        return fpaths

    def _resolve_latent_file(self, episode_index: int, start_frame: int, end_frame: int, key: str) -> Path:
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        for chunk_index in self._episode_chunk_candidates(episode_index):
            latent_file = self.latent_path / f"chunk-{chunk_index:03d}" / key / filename
            if latent_file.exists():
                return latent_file
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return self.latent_path / f"chunk-{episode_chunk:03d}" / key / filename

    def _resolve_rgb_motion_file(
        self,
        episode_index: int,
        start_frame: int,
        end_frame: int,
    ) -> Path:
        """Resolve the segment-level RGB-motion sidecar.

        Unlike video latents, the sidecar is intentionally *not* stored below
        a camera key: its ``motion_indices`` already address the transformer
        grid formed after all configured cameras are concatenated along width.
        Keeping one file per segment prevents ambiguous per-camera index
        conversion in the dataset loader.
        """
        root_name = getattr(self.config, 'rgb_motion_root_name', 'rgb_motion')
        motion_root = self.root / root_name
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"

        # Match the tolerant chunk lookup used for latent/tactile files.  Some
        # converted v2 datasets put every episode in chunk-000 even when their
        # metadata reports the canonical episode chunk.
        chunks_size = int(self.meta.info.get('chunks_size', 1000))
        chunk_candidates = [episode_index // chunks_size]
        for chunk_index in self._episode_chunk_candidates(episode_index):
            if chunk_index not in chunk_candidates:
                chunk_candidates.append(chunk_index)
        for chunk_index in chunk_candidates:
            path = motion_root / f"chunk-{chunk_index:03d}" / filename
            if path.is_file():
                return path

        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return motion_root / f"chunk-{episode_chunk:03d}" / filename

    @staticmethod
    def _rgb_motion_tensor(value, field_name: str) -> torch.Tensor:
        """Convert a sidecar field to a detached CPU tensor."""
        try:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            return torch.as_tensor(value)
        except Exception as exc:
            raise ValueError(
                f"RGB-motion field {field_name!r} is not tensor-like"
            ) from exc

    @classmethod
    def _rgb_motion_integer_tensor(cls, value, field_name: str) -> torch.Tensor:
        """Convert an integer-valued field without silently truncating floats."""
        tensor = cls._rgb_motion_tensor(value, field_name)
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise ValueError(f"RGB-motion field {field_name!r} must be integer-valued")
        if torch.is_floating_point(tensor) and tensor.numel():
            if not torch.isfinite(tensor).all() or not torch.equal(
                tensor, tensor.round()
            ):
                raise ValueError(
                    f"RGB-motion field {field_name!r} must be integer-valued; "
                    "fractional/NaN values cannot be used as indices"
                )
        return tensor.long()

    @classmethod
    def _rgb_motion_binary_tensor(cls, value, field_name: str) -> torch.Tensor:
        """Validate a tensor whose only legal values are 0 and 1."""
        tensor = cls._rgb_motion_tensor(value, field_name)
        if tensor.is_complex() or (
            tensor.numel() and not torch.all((tensor == 0) | (tensor == 1))
        ):
            raise ValueError(
                f"RGB-motion field {field_name!r} must contain only 0 or 1"
            )
        return tensor

    @classmethod
    def _rgb_motion_valid_mask(
        cls,
        value,
        *,
        source_layout: str,
        num_frames: int,
        tokens_per_frame: int,
        source_count: int,
    ) -> torch.Tensor:
        """Normalize a validity mask in the sidecar's source layout."""
        if value is None:
            if source_layout == 'frame':
                return torch.ones(
                    (num_frames, tokens_per_frame), dtype=torch.bool
                )
            return torch.ones((source_count,), dtype=torch.bool)

        mask = cls._rgb_motion_binary_tensor(
            value, 'motion_valid_mask'
        ).bool()
        if source_layout == 'frame':
            if mask.shape == (num_frames, tokens_per_frame):
                return mask
            if mask.numel() == num_frames * tokens_per_frame:
                return mask.reshape(num_frames, tokens_per_frame)
            raise ValueError(
                "RGB-motion motion_valid_mask must match motion_indices "
                f"shape {(num_frames, tokens_per_frame)}, got {tuple(mask.shape)}"
            )

        if mask.numel() != source_count:
            raise ValueError(
                "RGB-motion motion_valid_mask must have one entry per global "
                f"index ({source_count}), got {tuple(mask.shape)}"
            )
        return mask.reshape(source_count)

    @classmethod
    def _rgb_motion_scalar_field(
        cls,
        value,
        field_name: str,
        *,
        source_layout: str,
        num_frames: int,
        tokens_per_frame: int,
        source_count: int,
        source_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Map a scalar-per-token field into canonical ``[F, K]`` form."""
        tensor = cls._rgb_motion_tensor(value, field_name)
        target_shape = (num_frames, tokens_per_frame)

        # A scalar has unambiguous broadcast semantics.  [F] is accepted as a
        # per-frame value only for the canonical frame layout; in the global
        # layout N may equal F, so [N] must remain token-aligned.
        if tensor.numel() == 1:
            return tensor.reshape(1, 1).expand(target_shape).clone()

        if source_layout == 'frame':
            if tensor.ndim == 1 and tensor.shape[0] == num_frames:
                return tensor[:, None].expand(target_shape).clone()
            if tensor.shape == target_shape:
                return tensor
            if tensor.shape == (*target_shape, 1):
                return tensor[..., 0]
            if tensor.numel() == num_frames * tokens_per_frame:
                return tensor.reshape(target_shape)
            raise ValueError(
                f"RGB-motion field {field_name!r} must be scalar, [F], or "
                f"[F,K]={target_shape}; got {tuple(tensor.shape)}"
            )

        if tensor.numel() != source_count:
            raise ValueError(
                f"RGB-motion field {field_name!r} must contain one value per "
                f"global index ({source_count}); got {tuple(tensor.shape)}"
            )
        source = tensor.reshape(source_count)
        out = torch.zeros(target_shape, dtype=source.dtype)
        occupied = source_slots >= 0
        if occupied.any():
            out[occupied] = source[source_slots[occupied]]
        return out

    @classmethod
    def _rgb_motion_feature_field(
        cls,
        value,
        field_name: str,
        *,
        source_layout: str,
        num_frames: int,
        tokens_per_frame: int,
        source_count: int,
        source_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Map a vector-per-token field into canonical ``[F, K, D]`` form."""
        tensor = cls._rgb_motion_tensor(value, field_name)
        if source_layout == 'frame':
            if tensor.ndim >= 3 and tensor.shape[:2] == (
                num_frames, tokens_per_frame
            ):
                feature_width = 1
                for size in tensor.shape[2:]:
                    feature_width *= int(size)
                return tensor.reshape(
                    num_frames, tokens_per_frame, feature_width
                )
            if tensor.ndim >= 2 and tensor.shape[0] == (
                num_frames * tokens_per_frame
            ):
                feature_width = 1
                for size in tensor.shape[1:]:
                    feature_width *= int(size)
                return tensor.reshape(
                    num_frames, tokens_per_frame, feature_width
                )
            raise ValueError(
                f"RGB-motion field {field_name!r} must have shape [F,K,D] "
                f"with F={num_frames}, K={tokens_per_frame}; got "
                f"{tuple(tensor.shape)}"
            )

        if tensor.ndim < 2 or tensor.shape[0] != source_count:
            raise ValueError(
                f"RGB-motion field {field_name!r} must have shape [N,D] "
                f"with N={source_count}; got {tuple(tensor.shape)}"
            )
        feature_width = 1
        for size in tensor.shape[1:]:
            feature_width *= int(size)
        source = tensor.reshape(source_count, feature_width)
        out = torch.zeros(
            (num_frames, tokens_per_frame, source.shape[-1]),
            dtype=source.dtype,
        )
        occupied = source_slots >= 0
        if occupied.any():
            out[occupied] = source[source_slots[occupied]]
        return out

    @staticmethod
    def _latent_world_time_ids(
        latent_frame_ids, num_latent_frames: int
    ) -> torch.Tensor:
        """Return the full episode/stream clip's WAN-step ordinals.

        ``world_time_id`` is deliberately measured on the model's temporal
        axis: one increment means one WAN latent frame / transformer temporal
        token.  ``latent_frame_ids`` contains raw source-frame anchors and is
        retained in the signature for callers that already have it, but those
        values must not leak into the semantic index (nor be confused with the
        diffusion timestep).

        The implicit fallback assumes the loaded full clip begins at WAN step
        zero.  The caller constructs this vector before any random temporal
        crop, and the sidecar normalizer applies the same crop.  Thus a crop
        ``[s:e]`` keeps ordinals ``s..e-1`` instead of renumbering the crop
        from zero.  A sidecar for a mid-stream clip must provide an explicit
        ``world_time_id``; explicit values always take precedence in
        ``_normalize_rgb_motion_payload``.
        """
        num_latent_frames = int(num_latent_frames)
        if num_latent_frames < 0:
            raise ValueError("num_latent_frames must be non-negative")
        del latent_frame_ids  # raw source-frame ids are not semantic world time
        return torch.arange(num_latent_frames, dtype=torch.long)

    @staticmethod
    def _resize_rgb_motion_tokens(
        fields: dict,
        max_tokens_per_frame: int,
        *,
        source: str,
    ) -> dict:
        """Sort by score and pad/truncate every frame to a fixed token count."""
        target_k = int(max_tokens_per_frame)
        if target_k < 0:
            raise ValueError(
                f"rgb_motion_max_tokens must be >= 0, got {target_k}"
            )
        if target_k == 0:
            # Expert-only escape hatch: sidecars must then already share K for
            # batch>1, because default_collate cannot stack ragged dimensions.
            return fields

        scores = fields['motion_scores']
        valid = fields['motion_valid_mask']
        num_frames, current_k = valid.shape
        keep = min(current_k, target_k)
        if keep:
            rank_scores = scores.masked_fill(~valid, float('-inf'))
            order = torch.argsort(
                rank_scores, dim=1, descending=True, stable=True
            )[:, :keep]
        else:
            order = torch.empty((num_frames, 0), dtype=torch.long)

        fill_values = {
            'motion_indices': -1,
            'world_time_id': -1,
        }
        resized = {}
        for field_name, value in fields.items():
            if value.shape[:2] != (num_frames, current_k):
                raise ValueError(
                    f"Internal RGB-motion shape mismatch for {field_name!r} "
                    f"in {source}: expected leading [F,K]="
                    f"{(num_frames, current_k)}, got {tuple(value.shape)}"
                )
            if keep:
                gather_index = order
                for _ in value.shape[2:]:
                    gather_index = gather_index.unsqueeze(-1)
                gather_index = gather_index.expand(
                    (num_frames, keep, *value.shape[2:])
                )
                selected = torch.gather(value, 1, gather_index)
            else:
                selected = value[:, :0]

            if target_k > keep:
                pad_shape = (num_frames, target_k - keep, *value.shape[2:])
                pad = torch.full(
                    pad_shape,
                    fill_values.get(field_name, 0),
                    dtype=value.dtype,
                )
                selected = torch.cat((selected, pad), dim=1)
            resized[field_name] = selected.contiguous()
        return resized

    @classmethod
    def _normalize_rgb_motion_payload(
        cls,
        payload: dict,
        *,
        expected_full_frames: int,
        spatial_tokens_per_frame: int,
        latent_world_time_ids: torch.Tensor,
        max_tokens_per_frame: int = 0,
        truncate_start: int | None = None,
        truncate_end: int | None = None,
        source: str = '<in-memory>',
    ) -> dict:
        """Validate and normalize one RGB-motion sidecar.

        Canonical input uses ``motion_indices[F,K]``.  Each non-negative value
        is a *spatial* index within that frame after camera latents have been
        concatenated along width.  ``-1`` is padding.  For offline pipelines
        that already flattened the full FHW grid, the unambiguous alternative
        key ``rgb_motion_indices[N]`` is accepted and converted to [F,K].

        The returned dictionary always has the fixed, dense/collatable schema
        consumed by the model.  Padding is represented by -1 indices and the
        accompanying boolean mask.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                f"RGB-motion sidecar {source} must contain a dict, got "
                f"{type(payload).__name__}"
            )
        num_frames = int(expected_full_frames)
        spatial_size = int(spatial_tokens_per_frame)
        if num_frames <= 0 or spatial_size <= 0:
            raise ValueError(
                f"Invalid RGB-motion target grid F={num_frames}, "
                f"spatial_tokens_per_frame={spatial_size}"
            )

        has_frame_indices = 'motion_indices' in payload
        has_global_indices = 'rgb_motion_indices' in payload
        if has_frame_indices == has_global_indices:
            raise KeyError(
                f"RGB-motion sidecar {source} must contain exactly one of "
                "'motion_indices' ([F,K] frame-local) or "
                "'rgb_motion_indices' ([N] globally flattened)"
            )

        if has_frame_indices:
            source_layout = 'frame'
            indices = cls._rgb_motion_integer_tensor(
                payload['motion_indices'], 'motion_indices'
            )
            if indices.ndim != 2:
                raise ValueError(
                    f"RGB-motion motion_indices in {source} must be [F,K], "
                    f"got {tuple(indices.shape)}"
                )
            if indices.shape[0] != num_frames:
                raise ValueError(
                    f"RGB-motion/video latent frame mismatch in {source}: "
                    f"sidecar F={indices.shape[0]}, video latent F={num_frames}"
                )
            tokens_per_frame = int(indices.shape[1])
            source_count = num_frames * tokens_per_frame
            source_slots = torch.arange(source_count, dtype=torch.long).reshape(
                num_frames, tokens_per_frame
            )
            source_valid = cls._rgb_motion_valid_mask(
                payload.get('motion_valid_mask'),
                source_layout=source_layout,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                source_count=source_count,
            )
            if (indices < -1).any():
                raise ValueError(
                    f"RGB-motion motion_indices in {source} may only use -1 "
                    "for padding"
                )
            motion_valid = source_valid & (indices >= 0)
            indices = indices.masked_fill(~motion_valid, -1)
        else:
            source_layout = 'global'
            global_indices = cls._rgb_motion_integer_tensor(
                payload['rgb_motion_indices'], 'rgb_motion_indices'
            ).flatten()
            source_count = int(global_indices.numel())
            source_valid = cls._rgb_motion_valid_mask(
                payload.get('motion_valid_mask'),
                source_layout=source_layout,
                num_frames=num_frames,
                tokens_per_frame=0,
                source_count=source_count,
            )
            if (global_indices < -1).any():
                raise ValueError(
                    f"RGB-motion rgb_motion_indices in {source} may only use "
                    "-1 for padding"
                )
            source_valid = source_valid & (global_indices >= 0)
            max_global_index = num_frames * spatial_size
            if source_valid.any() and (
                global_indices[source_valid] >= max_global_index
            ).any():
                bad = int(global_indices[source_valid].max().item())
                raise ValueError(
                    f"RGB-motion global index {bad} in {source} is outside "
                    f"the flattened FHW token grid [0,{max_global_index})"
                )

            frame_for_source = torch.div(
                global_indices.clamp_min(0), spatial_size, rounding_mode='floor'
            )
            counts = torch.bincount(
                frame_for_source[source_valid], minlength=num_frames
            )
            tokens_per_frame = int(counts.max().item()) if counts.numel() else 0
            indices = torch.full(
                (num_frames, tokens_per_frame), -1, dtype=torch.long
            )
            source_slots = torch.full_like(indices, -1)
            next_slot = torch.zeros((num_frames,), dtype=torch.long)
            for source_index in torch.nonzero(source_valid, as_tuple=False).flatten():
                frame_index = int(frame_for_source[source_index].item())
                slot = int(next_slot[frame_index].item())
                indices[frame_index, slot] = (
                    global_indices[source_index] % spatial_size
                )
                source_slots[frame_index, slot] = source_index
                next_slot[frame_index] += 1
            motion_valid = indices >= 0

        if motion_valid.any() and (indices[motion_valid] >= spatial_size).any():
            bad = int(indices[motion_valid].max().item())
            raise ValueError(
                f"RGB-motion frame-local index {bad} in {source} is outside "
                f"the concatenated spatial token grid [0,{spatial_size})"
            )
        for frame_index in range(num_frames):
            selected = indices[frame_index, motion_valid[frame_index]]
            if selected.numel() != torch.unique(selected).numel():
                raise ValueError(
                    f"RGB-motion motion_indices in {source} contains duplicate "
                    f"valid addresses in frame {frame_index}"
                )

        field_kwargs = dict(
            source_layout=source_layout,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            source_count=source_count,
            source_slots=source_slots,
        )

        if 'dino_features' not in payload:
            raise KeyError(
                f"RGB-motion sidecar {source} is missing required "
                "'dino_features' (the visual semantic index)"
            )
        dino = cls._rgb_motion_feature_field(
            payload['dino_features'], 'dino_features', **field_kwargs
        )
        if dino.is_complex():
            raise ValueError(
                f"RGB-motion dino_features in {source} must be real-valued"
            )
        if dino.shape[-1] == 0:
            raise ValueError(
                f"RGB-motion dino_features in {source} has zero feature width"
            )
        if not torch.is_floating_point(dino):
            dino = dino.float()

        if 'neoforce_features' in payload:
            neoforce = cls._rgb_motion_feature_field(
                payload['neoforce_features'], 'neoforce_features', **field_kwargs
            )
            if neoforce.is_complex():
                raise ValueError(
                    f"RGB-motion neoforce_features in {source} must be real-valued"
                )
            if not torch.is_floating_point(neoforce):
                neoforce = neoforce.float()
        else:
            neoforce = torch.empty(
                (num_frames, tokens_per_frame, 0), dtype=dino.dtype
            )

        if 'motion_scores' in payload:
            motion_scores = cls._rgb_motion_scalar_field(
                payload['motion_scores'], 'motion_scores', **field_kwargs
            ).float()
        else:
            motion_scores = motion_valid.float()
        if motion_valid.any() and not torch.isfinite(
            motion_scores[motion_valid]
        ).all():
            raise ValueError(
                f"RGB-motion motion_scores in {source} must be finite for "
                "valid tokens"
            )

        if 'world_time_id' in payload:
            world_time_id = cls._rgb_motion_scalar_field(
                cls._rgb_motion_integer_tensor(
                    payload['world_time_id'], 'world_time_id'
                ),
                'world_time_id',
                **field_kwargs,
            )
        else:
            latent_world_time_ids = torch.as_tensor(
                latent_world_time_ids, dtype=torch.long
            ).flatten()
            if latent_world_time_ids.numel() != num_frames:
                raise ValueError(
                    "latent_world_time_ids must have one entry per video "
                    f"latent frame ({num_frames}), got "
                    f"{latent_world_time_ids.numel()}"
                )
            world_time_id = latent_world_time_ids[:, None].expand(
                num_frames, tokens_per_frame
            ).clone()

        if 'observation_flag' in payload:
            observation_flag = cls._rgb_motion_scalar_field(
                cls._rgb_motion_binary_tensor(
                    payload['observation_flag'], 'observation_flag'
                ),
                'observation_flag',
                **field_kwargs,
            ).long()
        else:
            observation_flag = motion_valid.long()
        if motion_valid.any() and not torch.all(
            (observation_flag[motion_valid] == 0)
            | (observation_flag[motion_valid] == 1)
        ):
            raise ValueError(
                f"RGB-motion observation_flag in {source} must contain only "
                "0 (predicted) or 1 (observed)"
            )

        if 'visual_valid' in payload:
            visual_valid = cls._rgb_motion_scalar_field(
                cls._rgb_motion_binary_tensor(
                    payload['visual_valid'], 'visual_valid'
                ),
                'visual_valid',
                **field_kwargs,
            ).bool()
        else:
            visual_valid = motion_valid.clone()

        if neoforce.shape[-1] > 0:
            if 'tactile_valid' not in payload:
                raise KeyError(
                    f"RGB-motion sidecar {source} supplies NeoForce features "
                    "but is missing tactile_valid; numeric zero cannot encode "
                    "whether tactile data is absent"
                )
            tactile_valid = cls._rgb_motion_scalar_field(
                cls._rgb_motion_binary_tensor(
                    payload['tactile_valid'], 'tactile_valid'
                ),
                'tactile_valid',
                **field_kwargs,
            ).bool()
        else:
            if 'tactile_valid' in payload:
                supplied_tactile_valid = cls._rgb_motion_scalar_field(
                    cls._rgb_motion_binary_tensor(
                        payload['tactile_valid'], 'tactile_valid'
                    ),
                    'tactile_valid',
                    **field_kwargs,
                ).bool()
                if supplied_tactile_valid.any():
                    raise ValueError(
                        f"RGB-motion tactile_valid in {source} cannot be true "
                        "when NeoForce is absent or has zero feature width"
                    )
            tactile_valid = torch.zeros_like(motion_valid)

        # A present-but-zero-width NeoForce tensor has the same semantics as a
        # missing NeoForce field.  It must never advertise tactile validity.
        if neoforce.shape[-1] == 0:
            tactile_valid = torch.zeros_like(motion_valid)

        # Padding carries no semantic data.  Keeping deterministic zero/-1
        # values makes serialized batches inspectable and avoids NaNs in cosine
        # similarity if a downstream component accidentally sees padding.
        indices = indices.masked_fill(~motion_valid, -1)
        motion_scores = motion_scores.masked_fill(~motion_valid, 0)
        world_time_id = world_time_id.masked_fill(~motion_valid, -1)
        observation_flag = observation_flag.masked_fill(~motion_valid, 0)
        visual_valid &= motion_valid
        tactile_valid &= motion_valid
        if visual_valid.any() and not torch.isfinite(dino[visual_valid]).all():
            raise ValueError(
                f"RGB-motion dino_features in {source} must be finite wherever "
                "visual_valid is true"
            )
        if tactile_valid.any() and not torch.isfinite(
            neoforce[tactile_valid]
        ).all():
            raise ValueError(
                f"RGB-motion neoforce_features in {source} must be finite "
                "wherever tactile_valid is true"
            )
        dino = dino.masked_fill(~motion_valid[..., None], 0)
        neoforce = neoforce.masked_fill(~motion_valid[..., None], 0)

        if motion_valid.any() and (~(visual_valid | tactile_valid) & motion_valid).any():
            raise ValueError(
                f"Every valid RGB-motion token in {source} must have DINO or "
                "NeoForce marked valid"
            )

        normalized = {
            'motion_indices': indices.contiguous(),
            'motion_valid_mask': motion_valid.contiguous(),
            'motion_scores': motion_scores.contiguous(),
            'world_time_id': world_time_id.contiguous(),
            'dino_features': dino.contiguous(),
            'neoforce_features': neoforce.contiguous(),
            'observation_flag': observation_flag.contiguous(),
            'visual_valid': visual_valid.contiguous(),
            'tactile_valid': tactile_valid.contiguous(),
        }
        normalized = cls._resize_rgb_motion_tokens(
            normalized, max_tokens_per_frame, source=source
        )

        start = 0 if truncate_start is None else int(truncate_start)
        end = num_frames if truncate_end is None else int(truncate_end)
        if not (0 <= start <= end <= num_frames):
            raise ValueError(
                f"Invalid RGB-motion truncation [{start}:{end}] for "
                f"F={num_frames} in {source}"
            )
        return {key: value[start:end] for key, value in normalized.items()}

    def _load_rgb_motion_sidecar(
        self,
        *,
        episode_index: int,
        local_start_frame: int,
        local_end_frame: int,
        expected_full_frames: int,
        spatial_tokens_per_frame: int,
        spatial_grid_shape: tuple[int, int] | None,
        camera_wan_grid_shapes: dict[str, tuple[int, int]] | None,
        latent_world_time_ids: torch.Tensor,
        latent_frame_ids,
        truncate_start: int | None = None,
        truncate_end: int | None = None,
    ) -> dict:
        sidecar_file = self._resolve_rgb_motion_file(
            episode_index, local_start_frame, local_end_frame
        )
        if not sidecar_file.is_file():
            raise FileNotFoundError(
                "RGB-motion tokens are enabled but the segment sidecar is "
                f"missing: {sidecar_file}. Expected one segment-level file "
                "whose motion_indices address the multi-camera WAN token grid."
            )
        try:
            payload = torch.load(
                sidecar_file, map_location='cpu', weights_only=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load RGB-motion sidecar {sidecar_file}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise TypeError(
                f"RGB-motion sidecar {sidecar_file} must contain a dict"
            )

        # A frame-local spatial index addresses the width-concatenated camera
        # grid.  With multiple cameras, provenance is therefore part of the
        # address contract rather than optional documentation: without it the
        # same integer can silently select a patch from a different camera.
        payload_camera_keys = payload.get(
            'camera_keys', payload.get('obs_cam_keys')
        )
        provenance = payload.get('provenance')
        payload_camera_grids = payload.get('camera_wan_grid_shapes')
        if payload_camera_grids is None and isinstance(provenance, dict):
            payload_camera_grids = provenance.get('camera_wan_grid_shapes')
        if len(self.used_video_keys) > 1:
            missing_provenance = [
                name for name, value in (
                    ('camera_keys', payload_camera_keys),
                    ('patch_size', payload.get('patch_size')),
                    ('spatial_grid_shape', payload.get('spatial_grid_shape')),
                    ('camera_wan_grid_shapes', payload_camera_grids),
                ) if value is None
            ]
            if missing_provenance:
                raise KeyError(
                    f"Multi-camera RGB-motion sidecar {sidecar_file} is missing "
                    f"address provenance {missing_provenance}"
                )
        if payload_camera_keys is not None and list(payload_camera_keys) != list(
            self.used_video_keys
        ):
            raise ValueError(
                f"RGB-motion camera order mismatch in {sidecar_file}: "
                f"sidecar={list(payload_camera_keys)}, "
                f"dataset={list(self.used_video_keys)}"
            )
        if payload_camera_grids is not None:
            if not isinstance(payload_camera_grids, dict):
                raise TypeError(
                    "RGB-motion camera_wan_grid_shapes must be a mapping from "
                    "camera key to [height, width]"
                )
            payload_grid_keys = list(payload_camera_grids)
            if payload_grid_keys != list(self.used_video_keys):
                raise ValueError(
                    f"RGB-motion per-camera grid order mismatch in {sidecar_file}: "
                    f"sidecar={payload_grid_keys}, "
                    f"dataset={list(self.used_video_keys)}"
                )
            normalized_payload_grids = {}
            for key in self.used_video_keys:
                shape = payload_camera_grids[key]
                if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                    raise ValueError(
                        "RGB-motion camera_wan_grid_shapes entries must be "
                        f"[height, width]; got {key}={shape!r}"
                    )
                normalized_shape = tuple(int(v) for v in shape)
                if any(v <= 0 for v in normalized_shape):
                    raise ValueError(
                        "RGB-motion camera_wan_grid_shapes must be positive; "
                        f"got {key}={normalized_shape}"
                    )
                normalized_payload_grids[key] = normalized_shape
            if camera_wan_grid_shapes is not None:
                normalized_current_grids = {
                    key: tuple(int(v) for v in camera_wan_grid_shapes[key])
                    for key in self.used_video_keys
                }
                if normalized_payload_grids != normalized_current_grids:
                    raise ValueError(
                        f"RGB-motion per-camera WAN grid mismatch in {sidecar_file}: "
                        f"sidecar={normalized_payload_grids}, "
                        f"current={normalized_current_grids}"
                    )
        payload_patch_size = payload.get('patch_size')
        expected_patch_size = tuple(getattr(self.config, 'patch_size', (1, 2, 2)))
        if (
            payload_patch_size is not None
            and tuple(payload_patch_size) != expected_patch_size
        ):
            raise ValueError(
                f"RGB-motion patch_size mismatch in {sidecar_file}: "
                f"sidecar={tuple(payload_patch_size)}, model={expected_patch_size}"
            )
        payload_grid_shape = payload.get('spatial_grid_shape')
        if (
            payload_grid_shape is not None
            and spatial_grid_shape is not None
            and tuple(int(v) for v in payload_grid_shape) != spatial_grid_shape
        ):
            raise ValueError(
                f"RGB-motion spatial grid mismatch in {sidecar_file}: "
                f"sidecar={tuple(payload_grid_shape)}, "
                f"current={spatial_grid_shape}"
            )
        bundle_frame_ids = (
            provenance.get('bundle_frame_ids')
            if isinstance(provenance, dict)
            else None
        )
        if bundle_frame_ids is not None:
            declared_frame_ids = self._rgb_motion_integer_tensor(
                bundle_frame_ids, 'provenance.bundle_frame_ids'
            ).flatten()
            current_frame_ids = self._rgb_motion_integer_tensor(
                latent_frame_ids, 'video latent frame_ids'
            ).flatten()
            if not torch.equal(declared_frame_ids, current_frame_ids):
                raise ValueError(
                    f"RGB-motion/raw frame alignment mismatch in {sidecar_file}: "
                    "provenance.bundle_frame_ids does not equal the current "
                    "WAN latent frame_ids"
                )
        return self._normalize_rgb_motion_payload(
            payload,
            expected_full_frames=expected_full_frames,
            spatial_tokens_per_frame=spatial_tokens_per_frame,
            latent_world_time_ids=latent_world_time_ids,
            max_tokens_per_frame=int(
                getattr(self.config, 'rgb_motion_max_tokens', 32)
            ),
            truncate_start=truncate_start,
            truncate_end=truncate_end,
            source=str(sidecar_file),
        )

    def _maybe_load_rgb_motion(
        self,
        out_dict: dict,
        *,
        episode_index: int,
        local_start_frame: int,
        local_end_frame: int,
        expected_full_frames: int,
        latent_world_time_ids: torch.Tensor | None,
        latent_frame_ids=None,
        camera_wan_grid_shapes: dict[str, tuple[int, int]] | None = None,
        truncate_start: int | None = None,
        truncate_end: int | None = None,
    ) -> None:
        """Attach canonical RGB-motion fields to a sample when opted in.

        Both the base dataset and the pi0.5 action-delta variant assemble their
        own samples. Keeping the grid calculation and sidecar loading here gives
        those entry points one identical opt-in contract.
        """
        if not self.use_rgb_motion_tokens:
            return
        if latent_world_time_ids is None:
            raise ValueError(
                "latent_world_time_ids are required when RGB-motion is enabled"
            )
        if latent_frame_ids is None:
            raise ValueError(
                "latent_frame_ids are required to verify RGB-motion/WAN alignment"
            )

        # ``_cat_video_latents`` returns [F,H,W_total,C], with all cameras
        # already concatenated along W. motion_indices are local spatial
        # addresses in the corresponding transformer-patch grid.
        _, latent_height, latent_width, _ = out_dict['latents'].shape
        patch_size = tuple(getattr(self.config, 'patch_size', (1, 2, 2)))
        if len(patch_size) != 3:
            raise ValueError(
                "patch_size must be a (temporal,height,width) triple, got "
                f"{patch_size}"
            )
        patch_time, patch_height, patch_width = (int(v) for v in patch_size)
        if patch_time != 1:
            raise ValueError(
                "RGB-motion sidecars currently require temporal patch_size=1; "
                f"got {patch_time}"
            )
        if (
            patch_height <= 0
            or patch_width <= 0
            or latent_height % patch_height != 0
            or latent_width % patch_width != 0
        ):
            raise ValueError(
                "RGB-motion indices require a divisible WAN spatial grid: "
                f"latent HxW={latent_height}x{latent_width}, "
                f"patch HxW={patch_height}x{patch_width}"
            )
        spatial_tokens_per_frame = (
            latent_height // patch_height
        ) * (latent_width // patch_width)
        spatial_grid_shape = (
            latent_height // patch_height,
            latent_width // patch_width,
        )
        if len(self.used_video_keys) > 1 and camera_wan_grid_shapes is None:
            raise ValueError(
                "camera_wan_grid_shapes are required for multi-camera "
                "RGB-motion address validation"
            )
        if camera_wan_grid_shapes is not None:
            if list(camera_wan_grid_shapes) != list(self.used_video_keys):
                raise ValueError(
                    "Current per-camera WAN grid order must match used_video_keys: "
                    f"grids={list(camera_wan_grid_shapes)}, "
                    f"cameras={list(self.used_video_keys)}"
                )
            normalized_camera_grids = {
                key: tuple(int(v) for v in camera_wan_grid_shapes[key])
                for key in self.used_video_keys
            }
            if any(
                len(shape) != 2 or any(v <= 0 for v in shape)
                for shape in normalized_camera_grids.values()
            ):
                raise ValueError(
                    "Every camera WAN grid must be a positive (height, width) pair"
                )
            camera_heights = {
                shape[0] for shape in normalized_camera_grids.values()
            }
            summed_width = sum(
                shape[1] for shape in normalized_camera_grids.values()
            )
            if camera_heights != {spatial_grid_shape[0]} or summed_width != spatial_grid_shape[1]:
                raise ValueError(
                    "Per-camera WAN grids do not reconstruct the concatenated "
                    f"grid: cameras={normalized_camera_grids}, "
                    f"concatenated={spatial_grid_shape}"
                )
            camera_wan_grid_shapes = normalized_camera_grids
        out_dict.update(
            self._load_rgb_motion_sidecar(
                episode_index=episode_index,
                local_start_frame=local_start_frame,
                local_end_frame=local_end_frame,
                expected_full_frames=int(expected_full_frames),
                spatial_tokens_per_frame=spatial_tokens_per_frame,
                spatial_grid_shape=spatial_grid_shape,
                camera_wan_grid_shapes=camera_wan_grid_shapes,
                latent_world_time_ids=latent_world_time_ids,
                latent_frame_ids=latent_frame_ids,
                truncate_start=truncate_start,
                truncate_end=truncate_end,
            )
        )

    def _resolve_tactile_latent_file(
        self,
        episode_index: int,
        start_frame: int,
        end_frame: int,
        key: str,
        mode: str,
        raise_on_ambiguous: bool = True,
    ) -> Path | None:
        tactile_root_name = getattr(self.config, 'tactile_latent_root_name', 'latents_tactile')
        tactile_root = self.root / tactile_root_name
        if not tactile_root.exists():
            return None
        filename = f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        glob_name = f"episode_{episode_index:06d}_*.pth"
        chunks_size = int(self.meta.info.get('chunks_size', 1000))
        chunk_candidates = [episode_index // chunks_size]
        for chunk_index in self._episode_chunk_candidates(episode_index):
            if chunk_index not in chunk_candidates:
                chunk_candidates.append(chunk_index)

        for chunk_index in chunk_candidates:
            directory = tactile_root / mode / f'chunk-{chunk_index:03d}' / key
            if not directory.exists():
                continue
            exact = directory / filename
            if exact.exists():
                return exact
            candidates = list(directory.glob(glob_name))
            if not candidates:
                continue
            if len(candidates) > 1:
                if raise_on_ambiguous:
                    raise FileNotFoundError(
                        f"Tactile latent segment {filename} not found in "
                        f"{directory}, and the episode glob matches "
                        f"{len(candidates)} files — cannot pick one safely. "
                        "Re-encode tactile latents per segment "
                        "(script/encode_tactile_latent.py) so names match the "
                        "video latents."
                    )
                return None
            return candidates[0]
        return None

    def _latent_frame_count(self, latent_file: Path) -> int:
        latent_file = Path(latent_file)
        cached = self._latent_frame_count_cache.get(latent_file)
        if cached is not None:
            return cached
        # mmap=True reads the tensor lazily, so reading only the small
        # 'latent_num_frames' metadata field doesn't pull the whole latent off
        # shared storage — ~4x faster for the validation pass. Fall back if unsupported.
        try:
            payload = torch.load(latent_file, map_location='cpu',
                                 weights_only=False, mmap=True)
        except Exception:
            payload = torch.load(latent_file, map_location='cpu', weights_only=False)
        count = int(payload['latent_num_frames'])
        self._latent_frame_count_cache[latent_file] = count
        return count

    def _rgb_motion_sidecar_summary(self, sidecar_file: Path) -> dict:
        """Read only enough sidecar state for segment validation."""
        sidecar_file = Path(sidecar_file)
        cached = self._rgb_motion_meta_cache.get(sidecar_file)
        if cached is not None:
            return cached
        try:
            payload = torch.load(
                sidecar_file,
                map_location='cpu',
                weights_only=False,
                mmap=True,
            )
        except Exception:
            # ``mmap`` is unavailable for some legacy torch serialization
            # formats. Fall back so readable older sidecars remain supported.
            payload = torch.load(
                sidecar_file, map_location='cpu', weights_only=False
            )
        if not isinstance(payload, dict):
            raise TypeError(
                f"RGB-motion sidecar {sidecar_file} must contain a dict"
            )
        if len(self.used_video_keys) > 1:
            camera_keys = payload.get(
                'camera_keys', payload.get('obs_cam_keys')
            )
            provenance = payload.get('provenance')
            camera_grids = payload.get('camera_wan_grid_shapes')
            if camera_grids is None and isinstance(provenance, dict):
                camera_grids = provenance.get('camera_wan_grid_shapes')
            required = {
                'camera_keys': camera_keys,
                'patch_size': payload.get('patch_size'),
                'spatial_grid_shape': payload.get('spatial_grid_shape'),
                'camera_wan_grid_shapes': camera_grids,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise KeyError(
                    f"Multi-camera RGB-motion sidecar {sidecar_file} is missing "
                    f"address provenance {missing}"
                )
            if list(camera_keys) != list(self.used_video_keys):
                raise ValueError(
                    f"RGB-motion camera order mismatch in {sidecar_file}: "
                    f"sidecar={list(camera_keys)}, "
                    f"dataset={list(self.used_video_keys)}"
                )
            expected_patch_size = tuple(
                getattr(self.config, 'patch_size', (1, 2, 2))
            )
            if tuple(payload['patch_size']) != expected_patch_size:
                raise ValueError(
                    f"RGB-motion patch_size mismatch in {sidecar_file}: "
                    f"sidecar={tuple(payload['patch_size'])}, "
                    f"model={expected_patch_size}"
                )
            if not isinstance(camera_grids, dict):
                raise TypeError(
                    "RGB-motion camera_wan_grid_shapes must be a mapping"
                )
            if list(camera_grids) != list(self.used_video_keys):
                raise ValueError(
                    f"RGB-motion per-camera grid order mismatch in {sidecar_file}: "
                    f"sidecar={list(camera_grids)}, "
                    f"dataset={list(self.used_video_keys)}"
                )
            normalized_camera_grids = {}
            for key in self.used_video_keys:
                shape = camera_grids[key]
                if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                    raise ValueError(
                        "RGB-motion camera_wan_grid_shapes entries must be "
                        f"[height, width]; got {key}={shape!r}"
                    )
                normalized_shape = tuple(int(v) for v in shape)
                if any(v <= 0 for v in normalized_shape):
                    raise ValueError(
                        "RGB-motion camera_wan_grid_shapes must be positive; "
                        f"got {key}={normalized_shape}"
                    )
                normalized_camera_grids[key] = normalized_shape
            global_shape = tuple(int(v) for v in payload['spatial_grid_shape'])
            heights = {shape[0] for shape in normalized_camera_grids.values()}
            total_width = sum(
                shape[1] for shape in normalized_camera_grids.values()
            )
            if heights != {global_shape[0]} or total_width != global_shape[1]:
                raise ValueError(
                    "RGB-motion per-camera WAN grids do not reconstruct "
                    f"spatial_grid_shape: cameras={normalized_camera_grids}, "
                    f"global={global_shape}"
                )
        if 'dino_features' not in payload:
            raise KeyError(
                f"RGB-motion sidecar {sidecar_file} is missing dino_features"
            )
        has_local = 'motion_indices' in payload
        has_global = 'rgb_motion_indices' in payload
        if has_local == has_global:
            raise KeyError(
                f"RGB-motion sidecar {sidecar_file} must contain exactly one "
                "index layout"
            )

        dino = self._rgb_motion_tensor(
            payload['dino_features'], 'dino_features'
        )
        if has_local:
            indices = self._rgb_motion_tensor(
                payload['motion_indices'], 'motion_indices'
            )
            if indices.ndim != 2:
                raise ValueError(
                    f"motion_indices must be [F,K], got {tuple(indices.shape)}"
                )
            if dino.ndim < 3 or dino.shape[:2] != indices.shape:
                raise ValueError(
                    "dino_features must share the [F,K] leading shape of "
                    f"motion_indices; got {tuple(dino.shape)} versus "
                    f"{tuple(indices.shape)}"
                )
            frame_count = int(indices.shape[0])
        else:
            indices = self._rgb_motion_tensor(
                payload['rgb_motion_indices'], 'rgb_motion_indices'
            )
            if indices.ndim != 1:
                raise ValueError(
                    f"rgb_motion_indices must be [N], got {tuple(indices.shape)}"
                )
            if dino.ndim < 2 or dino.shape[0] != indices.shape[0]:
                raise ValueError(
                    "dino_features must have one row per rgb_motion_indices "
                    f"entry; got {tuple(dino.shape)} versus {tuple(indices.shape)}"
                )
            declared_frames = payload.get(
                'latent_num_frames', payload.get('num_frames')
            )
            frame_count = (
                None if declared_frames is None else int(declared_frames)
            )

        summary = {'frame_count': frame_count}
        self._rgb_motion_meta_cache[sidecar_file] = summary
        return summary

    def _reject_meta(self, reason: str) -> bool:
        self._meta_filter_counts[reason] = self._meta_filter_counts.get(reason, 0) + 1
        return False

    def _valid_seg_cache_path(self):
        return Path(self.root) / ".valid_seg_cache.json"

    def _repo_validation_signature(self):
        """Cheap (no torch.load) fingerprint of the repo's data + filter config.
        The valid-segment cache is reused only when this matches, so any latent
        add/remove (re-encode / gapfill), parquet change, camera/tactile-key
        change, or filter-flag flip forces a fresh validation -> never a stale
        cache. Counting latent files is os-level (fast), unlike _check_meta's
        per-segment torch.load."""
        root = Path(self.root)
        vp = root / "latents" / "chunk-000"
        nvid = 0
        if vp.is_dir():
            for cam in vp.iterdir():
                if cam.is_dir():
                    nvid += sum(1 for _ in cam.glob("*.pth"))
        tname = getattr(self.config, 'tactile_latent_root_name', 'latents_tactile')
        tp = root / tname
        ntac = sum(1 for _ in tp.rglob("*.pth")) if tp.is_dir() else 0
        rgb_motion_enabled = bool(
            getattr(self.config, 'use_rgb_motion_tokens', False)
        )
        rgb_motion_root_name = getattr(
            self.config, 'rgb_motion_root_name', 'rgb_motion'
        )
        rp = root / rgb_motion_root_name
        nrgb = (
            sum(1 for _ in rp.rglob("*.pth"))
            if rgb_motion_enabled and rp.is_dir()
            else 0
        )
        pp = root / "data" / "chunk-000"
        npq = sum(1 for _ in pp.glob("*.parquet")) if pp.is_dir() else 0
        return {
            "v": 3,
            "parquet": npq,
            "video_latents": nvid,
            "tactile_latents": ntac,
            "rgb_motion_enabled": rgb_motion_enabled,
            "rgb_motion_root_name": str(rgb_motion_root_name),
            "rgb_motion_sidecars": nrgb,
            "rgb_motion_max_tokens": int(
                getattr(self.config, 'rgb_motion_max_tokens', 32)
            ),
            # Camera order determines width-concatenated RGB-motion addresses.
            "video_keys": list(self.used_video_keys),
            "tactile_keys": sorted(self.used_tactile_keys),
            "filter": bool(self.filter_mismatched_latents),
        }

    def _load_valid_seg_cache(self):
        """Cached set of valid (episode, start, end) keys iff the on-disk cache
        matches the current data signature; else None (-> full validation)."""
        if not bool(getattr(self.config, 'use_valid_seg_cache', True)):
            return None
        p = self._valid_seg_cache_path()
        if not p.is_file():
            return None
        try:
            blob = json.loads(p.read_text())
            if blob.get("signature") != self._repo_validation_signature():
                return None
            return {tuple(int(v) for v in k) for k in blob.get("valid", [])}
        except Exception:
            return None

    def _save_valid_seg_cache(self, valid_set):
        if not bool(getattr(self.config, 'use_valid_seg_cache', True)):
            return
        try:
            p = self._valid_seg_cache_path()
            tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps({
                "signature": self._repo_validation_signature(),
                "valid": [[int(k[0]), int(k[1]), int(k[2])] for k in valid_set],
            }))
            os.replace(tmp, p)  # atomic; concurrent ranks write identical content
        except Exception as exc:
            logging.warning("valid-seg cache write failed for %s: %s", self.repo_id, exc)

    def parse_meta(self):
        # One-time per data state: validating every segment (existence + frame
        # consistency of each latent via torch.load) is the slow part of init,
        # and all ranks repeat it. Cache the validated valid-segment set keyed by
        # a data signature so subsequent launches/resumes skip the torch.load
        # pass. Build it once up front.
        cached_valid = self._load_valid_seg_cache()
        out = []
        total = 0
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                total += 1
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                if cached_valid is not None:
                    check_statu = (int(episode_index),
                                   int(cur_meta["start_frame"]),
                                   int(cur_meta["end_frame"])) in cached_valid
                else:
                    check_statu = self._check_meta(
                        cur_meta["start_frame"],
                        cur_meta["end_frame"],
                        cur_meta["episode_index"],
                    )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out
        if cached_valid is None:
            self._save_valid_seg_cache({
                (int(m["episode_index"]), int(m["start_frame"]), int(m["end_frame"]))
                for m in out})
            if self._meta_filter_counts:
                logging.warning(
                    "Filtered %d/%d latent segments for repo=%s: %s",
                    total - len(out),
                    total,
                    self.repo_id,
                    self._meta_filter_counts,
                )
        else:
            logging.info(
                "valid-seg cache hit: %d/%d segments for repo=%s",
                len(out), total, self.repo_id)

    def _check_meta(self, start_frame, end_frame, episode_index):
        expected_frames = None
        for key in self.used_video_keys:
            latent_file = self._resolve_latent_file(episode_index, start_frame, end_frame, key)
            if not os.path.exists(latent_file):
                return self._reject_meta('missing_video_latent')
            if self.filter_mismatched_latents:
                try:
                    frame_count = self._latent_frame_count(latent_file)
                except Exception as exc:
                    logging.warning("Failed to read video latent metadata %s: %s", latent_file, exc)
                    return self._reject_meta('bad_video_latent')
                if expected_frames is None:
                    expected_frames = frame_count
                elif frame_count != expected_frames:
                    return self._reject_meta('video_frame_mismatch')

        if self.use_rgb_motion_tokens:
            sidecar_file = self._resolve_rgb_motion_file(
                episode_index, start_frame, end_frame
            )
            if not sidecar_file.is_file():
                return self._reject_meta('missing_rgb_motion_sidecar')
            try:
                summary = self._rgb_motion_sidecar_summary(sidecar_file)
            except Exception as exc:
                logging.warning(
                    "Failed to read RGB-motion sidecar metadata %s: %s",
                    sidecar_file,
                    exc,
                )
                return self._reject_meta('bad_rgb_motion_sidecar')
            sidecar_frames = summary['frame_count']
            if (
                expected_frames is not None
                and sidecar_frames is not None
                and sidecar_frames != expected_frames
            ):
                return self._reject_meta('rgb_motion_video_frame_mismatch')

        if (
            self.filter_mismatched_latents
            and self.has_tactile_condition
            and self.used_tactile_keys
            and not self.synthetic_tactile_data
        ):
            for key in self.used_tactile_keys:
                global_file = self._resolve_tactile_latent_file(
                    episode_index, start_frame, end_frame, key, 'global',
                    raise_on_ambiguous=False)
                local_file = self._resolve_tactile_latent_file(
                    episode_index, start_frame, end_frame, key, 'local',
                    raise_on_ambiguous=False)
                if global_file is None or local_file is None:
                    return self._reject_meta('missing_tactile_latent')
                try:
                    global_frames = self._latent_frame_count(global_file)
                    local_frames = self._latent_frame_count(local_file)
                except Exception as exc:
                    logging.warning(
                        "Failed to read tactile latent metadata repo=%s episode=%s key=%s: %s",
                        self.repo_id, episode_index, key, exc,
                    )
                    return self._reject_meta('bad_tactile_latent')
                if global_frames != local_frames:
                    return self._reject_meta('tactile_local_global_mismatch')
                if expected_frames is not None and global_frames != expected_frames:
                    return self._reject_meta('tactile_video_frame_mismatch')
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        out = {}
        for key in self.used_video_keys:
            latent_file = self._resolve_latent_file(episode_index, start_frame, end_frame, key)
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        reference_key = self.used_video_keys[0]
        reference_frame_ids = self._rgb_motion_integer_tensor(
            data_dict[f"{reference_key}.frame_ids"],
            f"{reference_key}.frame_ids",
        ).flatten()
        reference_latent_frames = int(
            data_dict[f"{reference_key}.latent_num_frames"]
        )
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            if int(latent_num_frames) != reference_latent_frames:
                raise ValueError(
                    "Video cameras must have identical WAN latent frame counts: "
                    f"{reference_key}={reference_latent_frames}, "
                    f"{key}={int(latent_num_frames)}"
                )
            camera_frame_ids = self._rgb_motion_integer_tensor(
                data_dict[f"{key}.frame_ids"], f"{key}.frame_ids"
            ).flatten()
            if not torch.equal(camera_frame_ids, reference_frame_ids):
                raise ValueError(
                    "Video camera latent frame_ids are not aligned: "
                    f"{key!r} differs from {reference_key!r}; width-concatenating "
                    "them would pair different source times"
                )
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"].detach()
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict

    def _camera_wan_grid_shapes(
        self,
        data_dict: dict,
    ) -> dict[str, tuple[int, int]]:
        """Return the exact per-camera transformer-patch address partition.

        A global motion index addresses cameras concatenated along latent width.
        Validating only the total width cannot detect a stale sidecar when two
        cameras change widths while preserving their sum, so each camera is an
        explicit part of the address contract.
        """
        patch_size = tuple(getattr(self.config, 'patch_size', (1, 2, 2)))
        if len(patch_size) != 3:
            raise ValueError(
                "patch_size must be a (temporal,height,width) triple, got "
                f"{patch_size}"
            )
        patch_time, patch_height, patch_width = (int(v) for v in patch_size)
        if patch_time != 1:
            raise ValueError(
                "RGB-motion sidecars currently require temporal patch_size=1; "
                f"got {patch_time}"
            )
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError(
                f"RGB-motion patch dimensions must be positive, got {patch_size}"
            )

        grids: dict[str, tuple[int, int]] = {}
        for key in self.used_video_keys:
            height = int(data_dict[f"{key}.latent_height"])
            width = int(data_dict[f"{key}.latent_width"])
            if (
                height <= 0
                or width <= 0
                or height % patch_height != 0
                or width % patch_width != 0
            ):
                raise ValueError(
                    "RGB-motion indices require each camera WAN latent grid "
                    f"to be divisible independently: camera={key!r}, "
                    f"latent HxW={height}x{width}, "
                    f"patch HxW={patch_height}x{patch_width}"
                )
            grids[key] = (height // patch_height, width // patch_width)
        if len({shape[0] for shape in grids.values()}) != 1:
            raise ValueError(
                "All camera WAN patch grids must have the same height before "
                f"width concatenation; got {grids}"
            )
        return grids
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='edge')

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _load_tactile_latents(
        self,
        episode_index: int,
        local_start_frame: int,
        local_end_frame: int,
        latent_frame_ids,
        truncate_start: int | None = None,
        truncate_end: int | None = None,
        expected_full_frames: int | None = None,
    ) -> dict | None:
        """Load pre-computed GlobalTactile + LocalTactile latents for an episode.

        Expects files produced by script/encode_tactile_latent.py at:
            <dataset_root>/latents_tactile/{global,local}/chunk-XXX/<tactile_key>/episode_*_*.pth

        Returns dict {
            'global':      torch.Tensor (S, C=48, F_lat_truncated, H_lat, W_lat),
            'local':       torch.Tensor (S, C=48, F_lat_truncated, H_lat, W_lat),
            'sensor_ids':  torch.LongTensor (S,)  — sensor index per slot,
        } or None if any sensor's latent is missing.

        truncate_start/truncate_end: indices into F_lat to slice (matches the
        truncation applied to video latents to keep temporal alignment).
        expected_full_frames: the VIDEO latent's full (pre-truncation) frame
        count. Tactile is frame-aligned with video in the attention mask, so a
        mismatched tactile F would silently misalign every tactile frame —
        assert instead of training on wrong semantics.
        """
        sensor_id_map = getattr(self.config, 'tactile_sensor_id_map', None)
        tactile_root_name = getattr(self.config, 'tactile_latent_root_name', 'latents_tactile')
        tactile_root = self.root / tactile_root_name
        if not tactile_root.exists():
            return None

        globals_list = []
        locals_list = []
        sensor_ids_list = []
        for key in self.used_tactile_keys:
            global_file = self._resolve_tactile_latent_file(
                episode_index, local_start_frame, local_end_frame, key, 'global')
            local_file = self._resolve_tactile_latent_file(
                episode_index, local_start_frame, local_end_frame, key, 'local')
            if global_file is None or local_file is None:
                # one sensor missing — skip whole tactile for this batch
                return None

            g_payload = torch.load(global_file, map_location='cpu', weights_only=False)
            l_payload = torch.load(local_file, map_location='cpu', weights_only=False)
            g_flat = g_payload['latent']                            # (F*H*W, C)
            l_flat = l_payload['latent']
            F_lat = int(g_payload['latent_num_frames'])
            H_lat = int(g_payload['latent_height'])
            W_lat = int(g_payload['latent_width'])
            F_lat_local = int(l_payload.get('latent_num_frames', F_lat))
            if F_lat_local != F_lat:
                raise ValueError(
                    f"Tactile local/global frame mismatch for {key} "
                    f"episode {episode_index}: local={F_lat_local}, "
                    f"global={F_lat}."
                )
            # Tactile is frame-aligned with video tokens (same frame_id in the
            # attention mask), so the FULL tactile F must equal the video's
            # full latent F — otherwise the shared truncate window slices a
            # shifted/short range and every tactile frame silently pairs with
            # the wrong video frame.
            if expected_full_frames is not None and F_lat != expected_full_frames:
                raise ValueError(
                    f"Tactile/video latent frame mismatch for {key} episode "
                    f"{episode_index}: tactile F={F_lat}, video F="
                    f"{expected_full_frames}. Re-encode tactile latents with "
                    "the same --target-fps/frame policy as the video latents."
                )
            # Reshape (F*H*W, C) → (F, H, W, C) → (C, F, H, W)
            g_5d = g_flat.reshape(F_lat, H_lat, W_lat, -1).permute(3, 0, 1, 2).contiguous()
            l_5d = l_flat.reshape(F_lat, H_lat, W_lat, -1).permute(3, 0, 1, 2).contiguous()

            # Slice along F to match the truncated video latent window
            if truncate_start is not None and truncate_end is not None:
                g_5d = g_5d[:, truncate_start:truncate_end]
                l_5d = l_5d[:, truncate_start:truncate_end]

            globals_list.append(g_5d)
            locals_list.append(l_5d)

            # Resolve sensor_id from cfg map, default to enumeration order
            if sensor_id_map and key in sensor_id_map:
                sensor_ids_list.append(int(sensor_id_map[key]))
            else:
                sensor_ids_list.append(len(sensor_ids_list))

        return {
            'global': torch.stack(globals_list, dim=0).contiguous(),       # (S, C, F, H, W)
            'local': torch.stack(locals_list, dim=0).contiguous(),
            'sensor_ids': torch.tensor(sensor_ids_list, dtype=torch.long),  # (S,)
        }

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)
        camera_wan_grid_shapes = (
            self._camera_wan_grid_shapes(ori_data_dict)
            if self.use_rgb_motion_tokens
            else None
        )

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        num_latent_frames = ori_data_dict[f"{self.used_video_keys[0]}.latent_num_frames"]
        # One stable world-time ordinal per *full* WAN latent frame. Compute it
        # before random temporal cropping so the RGB-motion sidecar applies the
        # identical [start_lat:end_lat] slice instead of renumbering the crop.
        latent_world_time_ids = (
            self._latent_world_time_ids(latent_frame_ids, int(num_latent_frames))
            if self.use_rgb_motion_tokens
            else None
        )

        # Truncate long episodes to avoid CUDA OOM
        start_lat = None        # init so tactile loader can reference outside the if
        end_lat = None
        max_latent_frames = int(getattr(self.config, 'max_latent_frames', 0))
        if max_latent_frames > 0 and num_latent_frames > max_latent_frames:
            start_lat = torch.randint(0, num_latent_frames - max_latent_frames + 1, (1,)).item()
            end_lat = start_lat + max_latent_frames
            # Truncate latent tensors for each video key
            for key in self.used_video_keys:
                F = ori_data_dict[f"{key}.latent_num_frames"]
                H = ori_data_dict[f"{key}.latent_height"]
                W = ori_data_dict[f"{key}.latent_width"]
                latent = ori_data_dict[f"{key}.latent"].reshape(F, H * W, -1)
                ori_data_dict[f"{key}.latent"] = latent[start_lat:end_lat].reshape(-1, latent.shape[-1])
                ori_data_dict[f"{key}.latent_num_frames"] = max_latent_frames
            # Truncate frame_ids: action code uses (len(frame_ids)-1)//4+1 as latent_frame_num
            # so we need exactly (max_latent_frames-1)*4+1 video frame entries
            needed_vid_frames = (max_latent_frames - 1) * 4 + 1
            vid_start = start_lat * 4
            vid_start = min(vid_start, max(0, len(latent_frame_ids) - needed_vid_frames))
            vid_end = vid_start + needed_vid_frames
            latent_frame_ids = latent_frame_ids[vid_start:vid_end]

        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)
        self._maybe_load_rgb_motion(
            out_dict,
            episode_index=episode_index,
            local_start_frame=local_start_frame,
            local_end_frame=local_end_frame,
            expected_full_frames=int(num_latent_frames),
            latent_world_time_ids=latent_world_time_ids,
            latent_frame_ids=ori_data_dict[
                f"{self.used_video_keys[0]}.frame_ids"
            ],
            camera_wan_grid_shapes=camera_wan_grid_shapes,
            truncate_start=start_lat,
            truncate_end=end_lat,
        )
        # ─── NEW: load pre-computed GlobalTactile / LocalTactile latents ───
        # (replaces the old RGB-video → CNN tactile pipeline)
        if self.has_tactile_condition and self.used_tactile_keys:
            tactile_payload = self._load_tactile_latents(
                episode_index=episode_index,
                local_start_frame=local_start_frame,
                local_end_frame=local_end_frame,
                latent_frame_ids=latent_frame_ids,
                truncate_start=start_lat,
                truncate_end=end_lat,
                # video's FULL latent frame count (num_latent_frames is read
                # before truncation) — tactile must match it frame-for-frame.
                expected_full_frames=int(num_latent_frames),
            )
            if tactile_payload is None and not self.tactile_optional:
                raise FileNotFoundError(
                    "Missing tactile latents for cond branch: "
                    f"repo={self.repo_id} episode={episode_index} "
                    f"frames={local_start_frame}:{local_end_frame} "
                    f"keys={self.used_tactile_keys}"
                )

            if tactile_payload is None:
                # tactile_optional: missing tactile falls back to the CFG-drop
                # path (model's zero-anchor keeps gradients FSDP-safe).
                out_dict['tactile_cond_drop'] = torch.tensor(True, dtype=torch.bool)
            else:
                # Keep the real tensors in the sample so missing files are still
                # caught above. The model uses this explicit flag to skip tactile
                # tokens entirely for CFG drop, so tactile modules get no gradient.
                tactile_cfg_prob = float(getattr(self.config, 'tactile_cfg_prob', 0.1))
                tactile_cond_drop = torch.rand(1).item() < tactile_cfg_prob

                out_dict['tactile_global_latent'] = tactile_payload['global']
                out_dict['tactile_local_latent'] = tactile_payload['local']
                out_dict['tactile_sensor_ids'] = tactile_payload['sensor_ids']
                out_dict['tactile_cond_drop'] = torch.tensor(
                    tactile_cond_drop, dtype=torch.bool)
        else:
            # Repo with no tactile sensors at all (per_repo_tactile_keys = []):
            # explicit drop flag so the model takes the zero-anchor path.
            out_dict['tactile_cond_drop'] = torch.tensor(True, dtype=torch.bool)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from n0_twam.configs import TWAM_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        TWAM_CONFIGS['base']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
