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


@torch.no_grad()
def route_contact_index(visual_index, contact_features, response, visual_rows,
                        visual_camera_ids, paired_camera_ids):
    """Attach confirmed contacts to existing third-person visual rows.

    visual_rows[M] is a caller-confirmed 2D correspondence, -1 for unknown.
    Camera choice is NOT a correspondence estimator. A confirmed pair is indexed
    symmetrically: the visual row receives the tactile NeoForce feature, while
    the tactile row keeps its NeoForce feature and inherits the paired visual
    row's DINO feature. Wrist/unknown contacts remain tactile-only metadata with
    DINO exactly zero. No Q/K/V is created or changed here; the caller aligns
    the returned tactile rows to its stream.
    """
    device = visual_index['dino'].device
    neo = torch.as_tensor(contact_features, device=device).float()
    response = torch.as_tensor(response, device=device).float()
    rows = torch.as_tensor(visual_rows, device=device)
    cameras = torch.as_tensor(visual_camera_ids, device=device)
    n = len(visual_index['dino'])
    if neo.ndim != 2 or neo.shape[-1] == 0 or not torch.isfinite(neo).all():
        raise ValueError('contact_features must be finite [M,D] with D>0')
    m = len(neo)
    if response.shape != (m,) or not torch.isfinite(response).all() or (response < 0).any():
        raise ValueError('response must be finite nonnegative [M]')
    if rows.shape != (m,) or rows.dtype != torch.long or (rows < -1).any() or (rows >= n).any():
        raise ValueError('visual_rows must be int64 [M], -1 or an existing visual row')
    if cameras.shape != (n,) or cameras.dtype != torch.long:
        raise ValueError('visual_camera_ids must be int64 [N]')
    if len(visual_index['neoforce']) != n:
        raise ValueError('visual index fields have different row counts')
    if visual_index['neoforce'].numel() and visual_index['neoforce'].any():
        raise ValueError('refuse to overwrite an already populated NeoForce index')
    visible = rows >= 0
    eligible = torch.zeros(n, dtype=torch.bool, device=device)
    for camera_id in paired_camera_ids:
        eligible |= cameras == int(camera_id)
    paired = torch.zeros(m, dtype=torch.bool, device=device)
    paired[visible] = eligible[rows[visible]]
    paired &= response > 0
    visual = {k: v.clone() for k, v in visual_index.items()}
    visual['neoforce'] = neo.new_zeros(n, neo.shape[-1])
    weight = neo.new_zeros(n)
    visual['neoforce'].index_add_(0, rows[paired], neo[paired] * response[paired, None])
    weight.index_add_(0, rows[paired], response[paired])
    visual['neoforce'] /= weight.clamp_min(torch.finfo(weight.dtype).tiny)[:, None]
    tactile_dino = visual_index['dino'].new_zeros(
        m, visual_index['dino'].shape[-1])
    if paired.any():
        tactile_dino[paired] = visual_index['dino'][rows[paired]]
    tactile_neoforce = neo.clone()
    tactile_neoforce[response == 0] = 0
    return visual, {
        'dino': tactile_dino,
        'neoforce': tactile_neoforce,
    }, paired
