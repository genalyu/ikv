"""Global KV retention metadata; none of these features enter attention Q/K/V.

A logical token occupies one slot in every layer. One policy owns its statistics
and draws one eviction plan for all layers. Missing DINO/NeoForce uses exact zero.
Contact duration is the nonzero interval represented by a token, in world-time
units (WAN steps by default), never a count of diffusion iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetentionConfig:
    top_k: int = 128
    time_scale: float = 8.0
    contact_weight: float = 1.0
    visual_weight: float = 1.0
    time_weight: float = 1.0
    query_weight: float = 1.0
    repetition_weight: float = 1.0
    action_scale: float = 0.1
    query_samples: int = 16
    seed: int = 0

    def __post_init__(self):
        for name in ("top_k", "query_samples", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("time_scale", "action_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("contact_weight", "visual_weight", "time_weight",
                     "query_weight", "repetition_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _max_cosine(left, right):
    """Bounded-memory visual relevance, with zero meaning absent."""
    result = torch.zeros(len(left), device=left.device)
    if left.shape[-1] == 0 or right.numel() == 0:
        return result
    present = (left != 0).any(-1)
    right = right[(right != 0).any(-1)]
    if not len(right):
        return result
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    for start in range(0, len(left), 256):
        best = result[start:start + 256]
        for block in right.split(256):
            best = torch.maximum(best, (left[start:start + 256] @ block.T).amax(-1))
        result[start:start + 256] = best
    return result.clamp(0, 1) * present


class GlobalKVRetention:
    """Statistics shared across all experts and layers of one streaming cache."""

    def __init__(self, capacity, device, config=None):
        self.config = config or RetentionConfig()
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.t0 = None
        self.next_uid = 0
        self.revision = 0
        self.reference_dino = torch.empty(0, 0, device=self.device)
        self.data = {
            "token_uid": torch.full((capacity,), -1, dtype=torch.long, device=device),
            "world_time_id": torch.zeros(capacity, device=device),
            "grid_position": torch.zeros(capacity, 3, device=device),
            "observation_flag": torch.zeros(capacity, dtype=torch.bool, device=device),
            "kind": torch.full((capacity,), -1, dtype=torch.long, device=device),
            "contact_time": torch.full((capacity,), -torch.inf, device=device),
            "contact_duration": torch.zeros(capacity, device=device),
            "query_mass": torch.zeros(capacity, device=device),
            "query_exposure": torch.zeros(capacity, device=device),
            "action_repetition": torch.zeros(capacity, device=device),
            "dino": torch.empty(capacity, 0, device=device),
            "neoforce": torch.empty(capacity, 0, device=device),
            "action": torch.empty(capacity, 0, device=device),
        }

    def snapshot(self):
        return (self.t0, self.next_uid, self.revision, self.reference_dino.clone(),
                {k: v.clone() for k, v in self.data.items()})

    def restore(self, snapshot):
        self.t0, self.next_uid, self.revision, self.reference_dino, self.data = snapshot

    def video_handle(self, mask, since_uid):
        """Capture this forward's video rows in input order, not physical order."""
        slots = (mask & (self.data['kind'] == 0)
                 & (self.data['token_uid'] >= since_uid)).nonzero().flatten()
        slots = slots[torch.argsort(self.data['token_uid'][slots])]
        return {'owner': self, 'slots': slots.clone(),
                **{name: self.data[name][slots].clone() for name in
                   ('token_uid', 'world_time_id', 'grid_position', 'observation_flag')}}

    @torch.no_grad()
    def annotate_video_dino(self, mask, handle, features):
        """Back-label existing predicted rows; never modify K/V or the real clock.

        Owner + UID checks reject reset/reused/evicted slots. The complete
        packet is validated before mutation, and observed seed rows stay intact.
        """
        if handle['owner'] is not self:
            raise ValueError('video index handle belongs to a different cache generation')
        slots = handle['slots']
        if (not mask[slots].all()
                or not torch.equal(self.data['token_uid'][slots], handle['token_uid'])
                or not (self.data['kind'][slots] == 0).all()):
            raise ValueError('video index handle is stale: KV slots were removed or reused')
        features = torch.as_tensor(features, device=self.device).detach().float()
        if (features.ndim != 2 or len(features) != len(slots)
                or features.shape[-1] == 0 or not torch.isfinite(features).all()):
            raise ValueError('predicted DINO must be finite [video_tokens, feature_dim]')
        width = self.data['dino'].shape[-1]
        if width not in (0, features.shape[-1]):
            raise ValueError('predicted DINO width differs from cached observation DINO')
        predicted = ~self.data['observation_flag'][slots]
        if not predicted.any():
            return 0
        if width == 0:
            self.data['dino'] = torch.zeros(self.capacity, features.shape[-1], device=self.device)
        self.data['dino'][slots[predicted]] = features[predicted]
        return int(predicted.sum())

    @staticmethod
    def _unit_scale(values):
        return values / values.amax().clamp_min(1e-12) if len(values) else values

    def components(self, slots, *, t0=None, reference=None):
        d = self.data
        anchor = self.t0 if t0 is None else t0
        time = torch.zeros(len(slots), device=self.device)
        if anchor is not None:
            time = torch.exp(-(d["world_time_id"][slots] - anchor).abs()
                             / self.config.time_scale)
        reference = self.reference_dino if reference is None else reference
        visual = _max_cosine(d["dino"][slots], reference)
        usage = d["query_mass"][slots] / d["query_exposure"][slots].clamp_min(1)
        return {
            "contact": self._unit_scale(d["contact_duration"][slots]),
            "visual": visual,
            "time": time,
            "query": self._unit_scale(usage),
            "repetition": d["action_repetition"][slots],
        }

    def scores(self, slots, **kwargs):
        terms = self.components(slots, **kwargs)
        c = self.config
        return (c.contact_weight * terms["contact"] + c.visual_weight * terms["visual"]
                + c.time_weight * terms["time"] + c.query_weight * terms["query"]
                - c.repetition_weight * terms["repetition"])

    def anchor_for(self, rows):
        """Only real VIDEO observations advance the global observation clock."""
        real = rows["observation_flag"] & (rows["kind"] == 0)
        if not real.any():
            return self.t0, self.reference_dino
        newest = float(rows["world_time_id"][real].max())
        if self.t0 is not None and newest < self.t0:
            return self.t0, self.reference_dino
        at_anchor = real & (rows["world_time_id"] == newest)
        return newest, rows["dino"][at_anchor].detach()

    def plan(self, mask, count, rows):
        """Protect global top-k OLD tokens, then randomly free the exact deficit.

        Incoming tokens must fit. If necessary k is capped by capacity-count;
        this explicit bound prevents impossible protection guarantees. Plans do
        not advance RNG/state, so temporary forwards and failed retries repeat
        the same draw. The caller applies this same plan to every layer.
        """
        if count > self.capacity:
            raise ValueError("current KV update exceeds global cache capacity")
        free = (~mask).nonzero().flatten()
        victims = free[:0]
        if len(free) < count:
            used = mask.nonzero().flatten()
            anchor, reference = self.anchor_for(rows)
            # A first DINO observation may arrive after featureless predictions.
            if reference.shape[-1] != self.data["dino"].shape[-1]:
                reference = self.reference_dino
            scores = self.scores(used, t0=anchor, reference=reference)
            protected_count = min(self.config.top_k, self.capacity - count, len(used))
            order = torch.argsort(scores, descending=True, stable=True)
            candidates = used[order[protected_count:]]
            rng = torch.Generator(device="cpu")
            rng.manual_seed(self.config.seed + self.revision)
            draw = torch.randperm(len(candidates), generator=rng).to(mask.device)
            victims = candidates[draw[:count - len(free)]]
            free = torch.cat((free, victims))
        return free[:count], victims

    @torch.no_grad()
    def commit(self, slots, rows, old_mask):
        """Store features once, independent of layer count and denoising steps."""
        if not len(slots):
            return
        for name in ("dino", "neoforce", "action"):
            width = rows[name].shape[-1]
            old_width = self.data[name].shape[-1]
            if width and old_width not in (0, width):
                raise ValueError(f"{name} width changed within one KV cache")
            if width and old_width == 0:
                self.data[name] = torch.zeros(self.capacity, width, device=self.device)
        # Compare committed action vectors, not Q/K features or vector magnitude.
        action_rows = (rows["kind"] == 1).nonzero().flatten()
        repetition = torch.zeros(len(slots), device=self.device)
        previous = self.data["action"][old_mask & (self.data["kind"] == 1)]
        for pos in action_rows.tolist():
            current = rows["action"][pos:pos + 1]
            if not current.shape[-1]:
                continue
            if current.shape[-1] and len(previous):
                distances = (previous - current).square().mean(-1)
                repetition[pos] = torch.exp(-distances.min() / self.config.action_scale**2)
            previous = torch.cat((previous, current), dim=0)
        self.t0, self.reference_dino = self.anchor_for(rows)
        for name in ("world_time_id", "grid_position", "observation_flag", "kind"):
            self.data[name][slots] = rows[name]
        for name in ("dino", "neoforce", "action"):
            self.data[name][slots] = 0
            if rows[name].shape[-1]:
                self.data[name][slots] = rows[name].detach().float()
        contact = (rows["neoforce"] != 0).any(-1)
        self.data["contact_time"][slots] = rows["world_time_id"].masked_fill(~contact, -torch.inf)
        self.data["contact_duration"][slots] = rows["duration"] * contact
        self.data["action_repetition"][slots] = repetition
        self.data["query_mass"][slots] = 0
        self.data["query_exposure"][slots] = 0
        self.data["token_uid"][slots] = torch.arange(
            self.next_uid, self.next_uid + len(slots), device=self.device)
        self.next_uid += len(slots)
        self.revision += 1

    @torch.no_grad()
    def measure_usage(self, q, k, slots, query_valid=None):
        """Sample conditional-batch queries; never materialize full Q x K."""
        q = q[0]
        if query_valid is not None:
            q = q[query_valid[0]]
        count = min(len(q), self.config.query_samples)
        if not count or not len(slots):
            return None
        positions = torch.linspace(0, len(q) - 1, count, device=q.device).long()
        q = q[positions].float()
        logits = torch.einsum("qhd,khd->hqk", q, k[0].float()) / math.sqrt(q.shape[-1])
        mass = logits.softmax(-1).sum(1).mean(0)
        return slots.detach(), mass, count

    def add_usage(self, measurements):
        measurements = [m for m in measurements if m is not None]
        for slots, mass, count in measurements:
            self.data["query_mass"][slots] += mass / len(measurements)
            self.data["query_exposure"][slots] += count / len(measurements)


def token_rows(context, *, batch_size, length, main_count, action_mode,
               update_cache, device):
    """Build token-major metadata from the SAME grid used for positional RoPE.

    kv_index features address the main video/action prefix as [B,N,D] or
    [N,D]. Optional tail_index addresses tactile tokens. CFG metadata must agree.
    Physical padding is filtered by the caller separately from feature presence.
    """
    grid = context["grid_id"]
    if grid.ndim != 3 or grid.shape[:2] != (batch_size, 4):
        raise ValueError("cache grid must have shape [B,4,N]")
    if not torch.equal(grid[:, 0], grid[:1, 0].expand_as(grid[:, 0])):
        raise ValueError("cache world times differ across CFG batch")
    if not torch.isfinite(grid).all() or not torch.equal(grid, grid[:1].expand_as(grid)):
        raise ValueError("cache positions must be finite and agree across CFG batch")
    times = grid[0, 0].float()
    if len(times) != length or not torch.isfinite(times).all():
        raise ValueError("cache world times must match the complete token sequence")
    kind = torch.full((length,), 2, dtype=torch.long, device=device)
    kind[:main_count] = 1 if action_mode else 0
    rows = {"world_time_id": times, "grid_position": grid[0, 1:].T.float(), "kind": kind,
            "observation_flag": torch.full((length,), update_cache == 2,
                                            dtype=torch.bool, device=device),
            "duration": torch.ones(length, device=device)}
    indices = [(context.get("index") or {}, 0, main_count),
               (context.get("tail_index") or {}, main_count, length)]
    for name in ("dino", "neoforce"):
        blocks = []
        width = 0
        for index, start, end in indices:
            value = torch.as_tensor(index.get(name, torch.empty(end-start, 0)), device=device)
            if value.ndim == 3:
                if value.shape[0] not in (1, batch_size):
                    raise ValueError(f"kv_index.{name} batch mismatch")
                if not torch.equal(value, value[:1].expand_as(value)):
                    raise ValueError(f"kv_index.{name} differs across CFG batch")
                value = value[0]
            if value.ndim != 2 or len(value) != end-start or not torch.isfinite(value).all():
                raise ValueError(f"kv_index.{name} must be finite [N,D] or [B,N,D]")
            width = max(width, value.shape[-1])
            blocks.append(value.float())
        if any(b.shape[-1] not in (0, width) for b in blocks):
            raise ValueError(f"main/tail {name} feature widths differ")
        rows[name] = torch.cat([b if b.shape[-1] else b.new_zeros(len(b), width)
                                for b in blocks])
    for index, start, end in indices:
        for name in ("duration", "observation_flag"):
            if name not in index:
                continue
            value = torch.as_tensor(index[name], device=device)
            if value.ndim == 2:
                if value.shape[0] not in (1, batch_size) or not torch.equal(value, value[:1].expand_as(value)):
                    raise ValueError(f"kv_index.{name} differs across batch")
                value = value[0]
            value = torch.broadcast_to(value, (end-start,))
            if not torch.isfinite(value).all():
                raise ValueError(f"kv_index.{name} must be finite")
            if name == "duration" and (value < 0).any():
                raise ValueError("contact duration must be nonnegative")
            if name == "observation_flag" and not ((value == 0) | (value == 1)).all():
                raise ValueError("observation_flag must be binary")
            rows[name][start:end] = value
    actions = context.get("actions")
    rows["action"] = torch.empty(length, 0, device=device)
    if action_mode and actions is not None:
        values = actions[0].flatten(1).T.float()
        if len(values) != main_count or not torch.isfinite(values).all():
            raise ValueError("action retention vectors must match action tokens")
        rows["action"] = torch.zeros(length, values.shape[-1], device=device)
        rows["action"][:main_count] = values
    return rows
