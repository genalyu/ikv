"""Versioned multimodal IKV. Content indices contain no spatial coordinates."""
from __future__ import annotations
from array import array
import torch
import torch.nn.functional as F
from .global_kv_retention import GlobalKVRetention, _max_cosine


class ContentHistory:
    """Bounded content prototypes; observation support, never cache residence."""
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.features = torch.empty(0, 0, device=device)
        self.duration = torch.empty(0, device=device)
        self.last_time = torch.empty(0, device=device)
        self.clock = None
        self.decision_cache = None
        self.cache_hits = 0

    def snapshot(self):
        return (self.features.clone(), self.duration.clone(), self.last_time.clone(), self.clock)

    def restore(self, state):
        self.features, self.duration, self.last_time, self.clock = state

    @torch.no_grad()
    def observe(self, features, times, durations):
        if not features.shape[-1]:
            return
        if not (torch.isfinite(features).all() & torch.isfinite(times).all()):
            raise ValueError("content observations must be finite")
        if self.features.shape[-1] not in (0, features.shape[-1]):
            raise ValueError("DINO width changed within an episode")
        for t in torch.unique(times, sorted=True).tolist():
            if self.clock is not None and t <= self.clock:
                continue  # duplicate commit / grounded overlap
            frame = times == t
            values = F.normalize(features[frame].float(), dim=-1)
            spans = durations[frame].float()
            present = values.abs().sum(-1) > 0
            values, spans = values[present], spans[present]
            # Greedy deterministic matching is position independent. Prototypes
            # are fixed to avoid chaining distinct objects through centroid drift.
            seen = {}
            cached = (None if self.decision_cache is None
                      else self.decision_cache.get(float(t)))
            if cached is not None:
                if len(cached) != len(values):
                    raise ValueError("cached content decisions do not match this frame")
                self.cache_hits += 1
                for feature, span, code in zip(values, spans.tolist(), cached):
                    if code < 0:
                        continue
                    best, action = code >> 2, code & 3
                    if action == 1:  # append a new prototype
                        if best != len(self.features):
                            raise ValueError("cached prototype append order changed")
                        if not self.features.numel():
                            self.features = feature[None].clone()
                        else:
                            self.features = torch.cat((self.features, feature[None]))
                        self.duration = torch.cat((self.duration, spans.new_zeros(1)))
                        self.last_time = torch.cat((self.last_time, spans.new_tensor([t])))
                    elif action == 2:  # replace an evicted prototype
                        if best >= len(self.features):
                            raise ValueError("cached prototype replacement is out of range")
                        self.features[best] = feature
                        self.duration[best] = 0
                        self.last_time[best] = t
                    elif action != 0 or best >= len(self.features):
                        raise ValueError("invalid cached content decision")
                    seen[best] = max(seen.get(best, 0.), float(span))
            else:
                # The first visit records the exact greedy decisions. Subsequent
                # visits to this trajectory replay them without any similarity
                # kernels or per-feature device synchronizations.
                decisions = array("i") if self.decision_cache is not None else None
                last_times = self.last_time.tolist()
                for feature, span in zip(values, spans.tolist()):
                    similarities = self.features @ feature if self.features.numel() else feature.new_empty(0)
                    scores = similarities.cpu()
                    best = int(scores.argmax()) if len(scores) else -1
                    action = 0
                    if best < 0 or float(scores[best]) < self.config.content_threshold:
                        if len(self.features) >= self.config.content_capacity:
                            candidates = [i for i in range(len(self.features)) if i not in seen]
                            if not candidates:
                                if decisions is not None:
                                    decisions.append(-1)
                                continue
                            best = min(candidates, key=last_times.__getitem__)
                            last_times[best] = t
                            self.features[best] = feature
                            self.duration[best] = 0
                            self.last_time[best] = t
                            action = 2
                        else:
                            best = len(self.features)
                            last_times.append(t)
                            if not self.features.numel():
                                self.features = feature[None].clone()
                            else:
                                self.features = torch.cat((self.features, feature[None]))
                            self.duration = torch.cat((self.duration, spans.new_zeros(1)))
                            self.last_time = torch.cat((self.last_time, spans.new_tensor([t])))
                            action = 1
                    if decisions is not None:
                        decisions.append((best << 2) | action)
                    seen[best] = max(seen.get(best, 0.), float(span))
                if decisions is not None:
                    self.decision_cache[float(t)] = decisions
            if len(seen) > 1:
                # Each slot appears once: batched updates preserve one addition
                # per duration while avoiding several launches per prototype.
                ids = torch.tensor(list(seen), dtype=torch.long, device=self.device)
                increments = self.duration.new_tensor(list(seen.values()))
                self.duration.index_add_(0, ids, increments)
                self.last_time.index_fill_(0, ids, t)
            else:
                for group, span in seen.items():
                    self.duration[group] += span
                    self.last_time[group] = t
            self.clock = t

    def persistence(self, features):
        result = features.new_zeros(len(features))
        if not features.shape[-1] or not self.features.numel():
            return result
        for start in range(0, len(features), 256):
            block = features[start:start+256]
            similarity, group = (F.normalize(block.float(), dim=-1) @ self.features.T).max(-1)
            support = 1 - torch.exp(-self.duration[group] / self.config.persistence_scale)
            result[start:start+256] = support * (similarity >= self.config.content_threshold) * block.ne(0).any(-1)
        return result

    def class_recency(self, features, times):
        """Age a visual token against its class's last real observation."""
        result = torch.zeros(len(features), device=features.device, dtype=torch.float32)
        if not features.shape[-1] or not self.features.numel():
            return result
        if features.shape[-1] != self.features.shape[-1] or len(times) != len(features):
            raise ValueError("class recency requires aligned DINO features and times")
        for start in range(0, len(features), 256):
            block = features[start:start+256]
            similarity, group = (F.normalize(block.float(), dim=-1) @ self.features.T).max(-1)
            age = (self.last_time[group] - times[start:start+256]).clamp_min(0)
            value = torch.exp(-age / self.config.class_recency_scale)
            result[start:start+256] = value * (similarity >= self.config.content_threshold) * block.ne(0).any(-1)
        return result


class MultimodalKVRetention(GlobalKVRetention):
    """Separate modality budgets, shared attention, detached version-2 indices."""
    def __init__(self, capacity, device, config):
        super().__init__(capacity, device, config)
        self.budgets = (config.video_capacity, config.action_capacity, config.tactile_capacity)
        if sum(self.budgets) != capacity:
            raise ValueError("v2 capacity must equal video + action + tactile capacities")
        # These belong to positional encoding / legacy scoring, never to i.
        for name in ("grid_position", "action", "action_repetition"):
            del self.data[name]
        self.data["contact_present"] = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.history = ContentHistory(config, self.device)
        self.tactile_scorer = None

    def snapshot(self):
        return super().snapshot(), self.history.snapshot()

    def restore(self, state):
        super().restore(state[0])
        self.history.restore(state[1])

    def video_handle(self, mask, since_uid):
        slots = (mask & (self.data["kind"] == 0) & (self.data["token_uid"] >= since_uid)).nonzero().flatten()
        slots = slots[torch.argsort(self.data["token_uid"][slots])]
        return {"owner": self, "slots": slots.clone(),
                **{name: self.data[name][slots].clone() for name in
                   ("token_uid", "world_time_id", "observation_flag")}}

    def anchor_for(self, rows):
        anchor, reference = super().anchor_for(rows)
        real = rows["observation_flag"]
        if real.any():
            latest = float(rows["world_time_id"][real].floor().max())
            anchor = latest if anchor is None else max(anchor, latest)
        return anchor, reference

    def components(self, slots, *, t0=None, reference=None):
        d, c = self.data, self.config
        anchor = self.t0 if t0 is None else t0
        age = torch.zeros(len(slots), device=self.device) if anchor is None else (
            anchor - d["world_time_id"][slots]).clamp_min(0)
        reference = self.reference_dino if reference is None else reference
        if reference.shape[-1] != d["dino"].shape[-1]:
            reference = self.reference_dino
        video = d["kind"][slots] == 0
        real_video = video & d["observation_flag"][slots]
        class_recency = torch.zeros(len(slots), device=self.device)
        if c.class_recency_weight and real_video.any():
            class_recency[real_video] = self.history.class_recency(
                d["dino"][slots[real_video]], d["world_time_id"][slots[real_video]])
        visual = torch.zeros(len(slots), device=self.device)
        persistence = torch.zeros(len(slots), device=self.device)
        if c.visual_weight and video.any():
            visual[video] = _max_cosine(d["dino"][slots[video]], reference)
        if c.persistence_weight and video.any():
            persistence[video] = self.history.persistence(d["dino"][slots[video]])
        return {
            "time": torch.exp(-age / c.time_scale),
            "query": d["query_mass"][slots] / d["query_exposure"][slots].clamp_min(1),
            "contact": 1 - torch.exp(-d["contact_duration"][slots] / c.contact_scale),
            "visual": visual,
            "persistence": persistence,
            "class_recency": class_recency,
        }

    @torch.no_grad()
    def scores(self, slots, **kwargs):
        terms, c = self.components(slots, **kwargs), self.config
        kind = self.data["kind"][slots]
        qw = torch.where(kind == 0, c.query_weight,
                        torch.where(kind == 1, c.action_query_weight, c.tactile_query_weight))
        result = c.time_weight * terms["time"] + qw * terms["query"]
        if self.tactile_scorer is not None:
            touch = kind == 2
            with torch.no_grad():
                extra = self.tactile_scorer(self.data["world_time_id"][slots[touch]].clone(),
                                           self.data["neoforce"][slots[touch]].clone())
                extra = torch.as_tensor(extra, device=self.device).detach()
                if extra.shape != result[touch].shape or not torch.isfinite(extra).all():
                    raise ValueError("tactile scorer must return one finite score per token")
                result[touch] += extra
        return result + (kind == 0) * (
            c.contact_weight * terms["contact"] + c.visual_weight * terms["visual"]
            + c.persistence_weight * terms["persistence"]
            + c.class_recency_weight * terms["class_recency"])

    def evidence(self, slots, mask, incoming=None):
        d = self.data
        valid = mask & ((d["kind"] == 0) | ((d["kind"] == 2) & d["contact_present"]))
        times = d["world_time_id"][valid]
        if incoming is not None:
            valid = (incoming["kind"] == 0) | (
                (incoming["kind"] == 2) & incoming.get("contact_present", incoming["neoforce"].ne(0).any(-1)))
            times = torch.cat((times, incoming["world_time_id"][valid]))
        return torch.isin(d["world_time_id"][slots].floor(), times.floor())

    def plan(self, mask, count, rows):
        if count != len(rows["kind"]) or not ((rows["kind"] >= 0) & (rows["kind"] <= 2)).all():
            raise ValueError("v2 plans require one valid modality per incoming token")
        chosen = torch.empty(count, dtype=torch.long, device=self.device)
        victims = []
        retained = mask.clone()
        anchor, reference = self.anchor_for(rows)
        # Determine evidence eviction before ranking actions.
        for kind in (0, 2, 1):
            incoming = (rows["kind"] == kind).nonzero().flatten()
            n = len(incoming)
            if n > self.budgets[kind]:
                raise ValueError(f"modality {kind} incoming {n} exceeds budget {self.budgets[kind]}")
            used = (mask & (self.data["kind"] == kind)).nonzero().flatten()
            need = max(0, len(used) + n - self.budgets[kind])
            if need:
                # Tie order: oldest time, then oldest token. Stable score sort.
                order = torch.argsort(self.data["token_uid"][used], stable=True)
                order = order[torch.argsort(self.data["world_time_id"][used[order]], stable=True)]
                score = self.scores(used, t0=anchor, reference=reference)
                order = order[torch.argsort(score[order], stable=True)]
                if kind == 1:
                    support = self.evidence(used, retained, rows)
                    order = order[torch.argsort(support[order].long(), stable=True)]
                evict = used[order[:need]]
                retained[evict] = False
                victims.append(evict)
        victims = torch.cat(victims) if victims else chosen[:0]
        free = (~retained).nonzero().flatten()
        if len(free) < count:
            raise RuntimeError("modality allocation exceeds physical capacity")
        return free[:count], victims

    @torch.no_grad()
    def observe_dense(self, features, times, durations):
        """Caller supplies real pre-motion observations inside its transaction."""
        self.history.observe(features, times, durations)

    @torch.no_grad()
    def commit(self, slots, rows, old_mask):
        for name in ("dino", "neoforce"):
            width, old_width = rows[name].shape[-1], self.data[name].shape[-1]
            if width and old_width not in (0, width):
                raise ValueError(f"{name} width changed within an episode")
            if width and not old_width:
                self.data[name] = torch.zeros(self.capacity, width, device=self.device)
        self.t0, self.reference_dino = self.anchor_for(rows)
        for name in ("world_time_id", "observation_flag", "kind"):
            self.data[name][slots] = rows[name]
        for name in ("dino", "neoforce"):
            self.data[name][slots] = 0
            if rows[name].shape[-1]:
                self.data[name][slots] = rows[name].detach().float()
        contact = rows.get("contact_present", rows["neoforce"].ne(0).any(-1))
        self.data["contact_present"][slots] = contact
        self.data["contact_time"][slots] = rows["world_time_id"].masked_fill(~contact, -torch.inf)
        self.data["contact_duration"][slots] = rows["duration"] * contact
        self.data["query_mass"][slots] = 0
        self.data["query_exposure"][slots] = 0
        self.data["token_uid"][slots] = torch.arange(self.next_uid, self.next_uid+len(slots), device=self.device)
        self.next_uid += len(slots)
        self.revision += 1


def make_retention_policy(capacity, device, config):
    if config.version == 2:
        return MultimodalKVRetention(capacity, device, config)
    return GlobalKVRetention(capacity, device, config)

