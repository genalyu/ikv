"""Inference-only CPU history archive, local to each pipeline stage.
Records are immutable for transactional snapshots; only missing selected KV move.
"""
from __future__ import annotations
from copy import copy
from pathlib import Path
import shutil
import uuid
import torch


class CPUKVArchive:
    def __init__(self, config):
        self.config, self.records, self.files = config, [], set()
        self.spill_dir = None
        self.stats = dict(archived_tokens=0, resident_bytes=0, spilled_bytes=0,
                          recalled_tokens=0, h2d_bytes=0, d2h_bytes=0)

    def snapshot(self):
        return list(self.records), dict(self.stats), set(self.files)

    def restore(self, state):
        records, stats, files = state
        for file in self.files - files:
            Path(file).unlink(missing_ok=True)
        self.records, self.stats, self.files = list(records), dict(stats), set(files)

    def close(self):
        if self.spill_dir is not None:
            shutil.rmtree(self.spill_dir, ignore_errors=True)
        self.records.clear()
        self.files.clear()
        self.stats.update({key: 0 for key in self.stats})

    @staticmethod
    def _bytes(kv):
        return sum(t.numel() * t.element_size() for pair in kv.values() for t in pair)

    @staticmethod
    def _host(t):
        out = t.detach().to("cpu").contiguous()
        return out.pin_memory() if t.is_cuda else out.clone()

    def append(self, policy, attentions, cache_name, slots):
        slots = slots[policy.data["observation_flag"][slots]]
        if not len(slots):
            return
        metadata = {name: value[slots].detach().cpu().clone() for name, value in policy.data.items()}
        kv = {layer: tuple(self._host(attention.attn_caches[cache_name][name][:, slots])
                          for name in ("k", "v")) for layer, attention in attentions}
        first = attentions[0][1].attn_caches[cache_name]
        sidecar = first.get("semantic")
        semantic = None if sidecar is None else {
            name: value[slots].detach().cpu().clone() for name, value in sidecar.items()}
        size = self._bytes(kv)
        if (self.config.archive_overflow == "error" and
                self.stats["resident_bytes"] + size > self.config.archive_max_bytes):
            raise MemoryError("CPU KV archive limit exceeded; request rolled back. "
                              "Increase archive_max_bytes or configure disk overflow.")
        self.records = self.records + [dict(metadata=metadata, semantic=semantic, kv=kv, path=None, size=size)]
        self.stats["archived_tokens"] += len(slots)
        self.stats["resident_bytes"] += size
        self.stats["d2h_bytes"] += size
        self._spill()

    def _spill(self):
        if self.stats["resident_bytes"] <= self.config.archive_max_bytes:
            return
        if self.spill_dir is None:
            self.spill_dir = Path(self.config.archive_directory) / ("episode-" + uuid.uuid4().hex)
            self.spill_dir.mkdir(parents=True, exist_ok=False)
        for index, record in enumerate(self.records):
            if record["kv"] is None:
                continue
            target = self.spill_dir / (uuid.uuid4().hex + ".pt")
            try:
                torch.save(record["kv"], target)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            self.files.add(str(target))
            self.records = self.records[:index] + [dict(record, kv=None, path=str(target))] + self.records[index+1:]
            self.stats["resident_bytes"] -= record["size"]
            self.stats["spilled_bytes"] += record["size"]
            if self.stats["resident_bytes"] <= self.config.archive_max_bytes:
                break

    @staticmethod
    def _cat(records):
        result = {}
        for name in records[-1]["metadata"]:
            parts = [record["metadata"][name] for record in records]
            width = max((x.shape[-1] for x in parts if x.ndim == 2), default=0)
            if width:
                parts = [x if x.shape[-1] == width else x.new_zeros(len(x), width) for x in parts]
            result[name] = torch.cat(parts)
        return result

    def select(self, policy, rows):
        if not self.records:
            if policy.config.score_backend == "qwen_sys2":
                empty = {name: value[:0].cpu() for name,value in policy.data.items()}
                policy.sys2_scorer(empty, rows, policy.next_uid)
            return None, torch.empty(0, dtype=torch.long)
        data = self._cat(self.records)
        scorer = copy(policy)
        scorer.device, scorer.data = torch.device("cpu"), data
        scorer.reference_dino = policy.reference_dino.cpu()
        scorer.history = copy(policy.history)
        for name in ("features", "duration", "last_time"):
            setattr(scorer.history, name, getattr(policy.history, name).cpu())
        scorer.tactile_scorer = None
        anchor, reference = policy.anchor_for(rows)
        if policy.config.score_backend == "qwen_sys2":
            scores = policy.sys2_scorer(data, rows, policy.next_uid).float()
            if scores.shape != data["kind"].shape or not torch.isfinite(scores).all():
                raise ValueError("Sys2 archive score shape or values invalid")
            action = data["kind"] == 1
            age = (float(anchor) - data["world_time_id"][action]).clamp_min(0)
            scores[action] = policy.config.time_weight * torch.exp(-age / policy.config.time_scale)
            data["sys2_score"] = scores.clone()
        else:
            scores = scorer.scores(torch.arange(len(data["kind"])), t0=anchor, reference=reference.cpu())
        incoming = {name: value.detach().cpu() for name, value in rows.items()}
        selected = []
        for kind in (0, 2, 1):
            available = policy.budgets[kind] - int((incoming["kind"] == kind).sum())
            if available < 0:
                raise ValueError("incoming modality exceeds GPU budget")
            ids = (data["kind"] == kind).nonzero().flatten()
            order = torch.argsort(data["token_uid"][ids], descending=True, stable=True)
            order = order[torch.argsort(data["world_time_id"][ids[order]], descending=True, stable=True)]
            order = order[torch.argsort(scores[ids[order]], descending=True, stable=True)]
            if kind == 1 and len(ids):
                evidence = torch.cat(selected) if selected else ids[:0]
                contact = incoming.get("contact_present", incoming["neoforce"].ne(0).any(-1))
                valid = (incoming["kind"] == 0) | ((incoming["kind"] == 2) & contact)
                valid_evidence = (data["kind"][evidence] == 0) | data["contact_present"][evidence]
                times = torch.cat((data["world_time_id"][evidence[valid_evidence]], incoming["world_time_id"][valid]))
                support = torch.isin(data["world_time_id"][ids].floor(), times.floor())
                order = order[torch.argsort(support[order].long(), descending=True, stable=True)]
            selected.append(ids[order[:available]])
        return data, torch.cat(selected)

    def prepare(self, backbone, attentions, cache_name, policy, rows, entries):
        data, selected = self.select(policy, rows)
        if data is None:
            return
        first = attentions[0][1].attn_caches[cache_name]
        wanted = data["token_uid"][selected].to(policy.device)
        used = first["mask"].nonzero().flatten()
        keep = used[torch.isin(policy.data["token_uid"][used], wanted)]
        missing = selected[~torch.isin(data["token_uid"][selected], policy.data["token_uid"][keep].cpu())]
        removed = used[~torch.isin(used, keep)]
        free_mask = ~first["mask"].clone()
        free_mask[removed] = True
        target = free_mask.nonzero().flatten()[:len(missing)]
        changed = torch.unique(torch.cat((removed, target)))
        if policy.config.score_backend == "qwen_sys2":
            lookup = {int(uid): float(score) for uid, score in zip(data["token_uid"], data["sys2_score"])}
            policy.data["sys2_score"][keep] = torch.tensor(
                [lookup[int(uid)] for uid in policy.data["token_uid"][keep]], device=policy.device)
        if not len(changed):
            return
        for _, attention in attentions:
            cache = attention.attn_caches[cache_name]
            entries.append((attention, cache_name, attention._snapshot_cache_slots(cache, changed)))
            cache["mask"][removed], cache["id"][removed], cache["is_pred"][removed] = False, -1, False
        semantic = first.get("semantic")
        if semantic is not None:
            semantic["valid"][removed] = False
        offset = 0
        for record in self.records:
            n = len(record["metadata"]["kind"])
            match = ((missing >= offset) & (missing < offset+n)).nonzero().flatten()
            if len(match):
                local, dest = missing[match] - offset, target[match.to(target.device)]
                kv = record["kv"]
                if kv is None:
                    kv = torch.load(record["path"], map_location="cpu", weights_only=True)
                for layer, attention in attentions:
                    cache = attention.attn_caches[cache_name]
                    for name, value in zip(("k", "v"), kv[layer]):
                        cache[name][:, dest] = value[:, local].to(cache[name].device)
                    cache["mask"][dest] = True
                    cache["id"][dest] = record["metadata"]["token_uid"][local].to(dest.device)
                    cache["is_pred"][dest] = False
                if semantic is not None and record["semantic"] is not None:
                    for name, value in record["semantic"].items():
                        semantic[name][dest] = value[local].to(semantic[name].device)
                self.stats["h2d_bytes"] += record["size"] * len(local) // n
            offset += n
        for name, values in data.items():
            old = policy.data[name]
            if old.ndim == 2 and old.shape[-1] != values.shape[-1]:
                policy.data[name] = old.new_zeros(policy.capacity, values.shape[-1])
            policy.data[name][target] = values[missing].to(policy.device)
        self.stats["recalled_tokens"] += len(missing)
        backbone._share_semantic_sidecar(cache_name)


def archive_prepare(backbone, attentions, cache_name, policy, rows):
    if policy.archive is not None:
        policy.archive.prepare(backbone, attentions, cache_name, policy, rows,
                               backbone._active_cache_transactions[cache_name]["entries"])


def archive_append(attentions, cache_name, policy, slots):
    if policy.archive is not None:
        policy.archive.append(policy, attentions, cache_name, slots)
