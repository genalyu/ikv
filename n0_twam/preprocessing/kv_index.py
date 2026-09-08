"""Dense RGB index helpers: no motion filtering and no prediction feedback."""
from __future__ import annotations

import torch

from n0_twam.models.rgb_motion import pool_dino_to_grid


def observed_index(payload, count, device):
    """Validate externally aligned features. Exact zero denotes unavailable.

    Rows follow WAN's (frame, height, concatenated-camera-width) token order.
    There are intentionally no persisted visual_valid/tactile_valid fields.
    """
    payload = {} if payload is None else payload
    unsupported = set(payload) - {"dino", "neoforce", "duration", "observation_flag"}
    if unsupported:
        raise ValueError(f"unknown kv_index fields: {sorted(unsupported)}")
    result = {}
    for name in ("dino", "neoforce"):
        value = torch.as_tensor(payload.get(name, torch.empty(count, 0)), device=device)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or len(value) != count or not torch.isfinite(value).all():
            raise ValueError(f"observed kv_index.{name} must be finite [N,D] or [1,N,D], N={count}")
        result[name] = value.detach().float()
    duration = torch.as_tensor(payload.get("duration", 1.0), device=device).reshape(-1)
    duration = torch.broadcast_to(duration, (count,))
    if not torch.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("kv_index.duration must be finite and nonnegative")
    flag = torch.as_tensor(payload.get("observation_flag", 1), device=device)
    if not (flag == 1).all():
        raise ValueError("observed RGB index cannot be labelled predicted")
    result["duration"] = duration.detach().float()
    result["observation_flag"] = torch.ones(count, dtype=torch.bool, device=device)
    return result


@torch.no_grad()
def encode_dense_dino(videos, anchors, target_size, encoder):
    """[cameras,3,raw_T,H,W] in [0,1] -> [WAN_tokens,D].

    DINO describes each causal latent's endpoint RGB (not its entire receptive
    field). Spatial pooling is approximate alignment, not object tracking.
    """
    cameras = []
    for video in videos:
        rgb = video[:, anchors].permute(1, 0, 2, 3)
        features = encoder(rgb).tokens
        cameras.append(pool_dino_to_grid(features, target_size))
    return torch.cat(cameras, dim=2).flatten(0, 2).detach()


def prediction_index(count, device, seed=None):
    """Unknown predicted features stay zero; only the real cold seed is copied."""
    seed_count = 0 if seed is None else len(seed["observation_flag"])
    if seed_count > count:
        raise ValueError("real seed index exceeds prediction chunk")
    result = {
        "observation_flag": torch.zeros(count, dtype=torch.bool, device=device),
        "duration": torch.ones(count, device=device),
    }
    for name in ("dino", "neoforce"):
        width = 0 if seed is None else seed[name].shape[-1]
        result[name] = torch.zeros(count, width, device=device)
    if seed is not None:
        for name in result:
            result[name][:seed_count] = seed[name].to(device)
    return result


def concat_indices(first, second):
    if first is None:
        return second
    if second is None:
        return first
    result = {}
    for name in first:
        a, b = first[name], second[name].to(first[name].device)
        if name in ("dino", "neoforce") and a.shape[-1] != b.shape[-1]:
            width = max(a.shape[-1], b.shape[-1])
            if a.shape[-1] and b.shape[-1]:
                raise ValueError(f"{name} width changed between observations")
            if not a.shape[-1]:
                a = a.new_zeros(len(a), width)
            if not b.shape[-1]:
                b = b.new_zeros(len(b), width)
        result[name] = torch.cat((a, b))
    return result
