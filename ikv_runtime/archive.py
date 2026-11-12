"""Transactional CPU archive of real, pre-RoPE keys and native values.

GPU residency is bounded by the requested budget. The CPU archive contains all
committed candidates, including ones not selected this round. There is no
silent eviction and no heuristic score fallback.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
import torch
from .types import Candidate


def _order(candidate):
    return (candidate.time, candidate.uid)


def _storage_bytes(layers):
    storages = {}
    for payload in layers.values():
        for pair in payload.values():
            for tensor in pair:
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class ArchiveBank:
    def __init__(self, expected_layers: int, max_bytes: int = 64 * 1024**3):
        if type(expected_layers) is not int or expected_layers < 1:
            raise ValueError("expected_layers must be positive")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.expected_layers = expected_layers
        self.max_bytes = max_bytes
        self._metadata = {}
        self._layers = {}
        self._layout = {}
        self.generation = 0
        self.bytes = 0

    @property
    def candidates(self):
        return tuple(sorted(self._metadata.values(), key=_order))

    @property
    def candidate_ids(self):
        return tuple(c.uid for c in self.candidates)

    def reset(self):
        self._metadata = {}
        self._layers = {}
        self._layout = {}
        self.bytes = 0
        self.generation += 1

    def begin(self, fresh_candidates, budget: int, scores=None, *, score_fn=None,
              selected_ids=None):
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        fresh = tuple(fresh_candidates)
        if any(not isinstance(c, Candidate) for c in fresh):
            raise TypeError("fresh_candidates must contain Candidate records")
        if len({c.uid for c in fresh}) != len(fresh):
            raise ValueError("Duplicate fresh candidate IDs")
        metadata = dict(self._metadata)
        for c in fresh:
            if c.uid in metadata and metadata[c.uid] != c:
                raise ValueError("A candidate ID cannot change its observation provenance")
            metadata[c.uid] = c
        candidates = tuple(sorted(metadata.values(), key=_order))
        if scores is not None and score_fn is not None:
            raise ValueError("Specify scores or score_fn, not both")
        # A plan may be broadcast by one PP rank. Each layer owner materializes
        # exactly these IDs from its own archive; KV need not cross the network.
        if selected_ids is not None:
            selected = tuple(selected_ids)
            if len(set(selected)) != len(selected) or len(selected) > budget:
                raise ValueError("Invalid explicit selection")
            if any(uid not in metadata for uid in selected):
                raise ValueError("Unknown explicit selection ID")
            if len(selected) != min(budget, len(candidates)):
                raise ValueError("Explicit selection must fill the available budget")
        elif budget == 0:
            selected = ()
        elif len(candidates) <= budget:
            selected = tuple(c.uid for c in candidates)
        else:
            provider = score_fn if score_fn is not None else scores
            values = provider(candidates) if callable(provider) else provider
            if not isinstance(values, Mapping):
                raise ValueError("A learned scorer or explicit test scores are required")
            if any(c.uid not in values for c in candidates):
                raise ValueError("Scores must cover the entire archive, not only residents")
            numeric = {c.uid: float(values[c.uid]) for c in candidates}
            if not all(math.isfinite(v) for v in numeric.values()):
                raise ValueError("Nonfinite candidate scores")
            ranked = sorted(candidates, key=lambda c: (-numeric[c.uid], c.time, c.uid))
            selected = tuple(c.uid for c in ranked[:budget])
        selected = tuple(sorted(selected, key=lambda uid: _order(metadata[uid])))
        return KVTransaction(self, fresh, metadata, candidates, selected, budget)


class KVTransaction:
    def __init__(self, bank, fresh, metadata, candidates, selected_ids, budget):
        self.bank = bank
        self.generation = bank.generation
        self.fresh_candidates = fresh
        self.candidates = candidates
        self.selected_ids = selected_ids
        self.budget = budget
        self.valid_mask = torch.arange(budget) < len(selected_ids)
        self._metadata = metadata
        self._staged = {}
        self._layouts = {}
        self._closed = False

    def _check(self):
        if self._closed or self.bank.generation != self.generation:
            raise RuntimeError("Closed or stale KV transaction")

    @torch.no_grad()
    def layer(self, layer_id, k_raw, v, seq_dim=1):
        """Stage fresh rows and gather the selected budget on the caller device.

        Returned padding is zero with valid_mask=False and is never archived.
        Both native 3-D and 4-D layouts are accepted; seq_dim is explicit.
        """
        self._check()
        if type(layer_id) is not int or not 0 <= layer_id < self.bank.expected_layers:
            raise ValueError("Unexpected layer ID")
        if layer_id in self._staged:
            raise RuntimeError("Each layer may be staged once per transaction")
        if not isinstance(k_raw, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError("KV payloads must be tensors")
        if k_raw.ndim < 2 or k_raw.shape != v.shape or k_raw.device != v.device:
            raise ValueError("Key/value shape or device mismatch")
        if not k_raw.is_floating_point() or not v.is_floating_point():
            raise ValueError("KV payloads must be floating point")
        if not -k_raw.ndim <= seq_dim < k_raw.ndim:
            raise ValueError("Invalid sequence dimension")
        dim = seq_dim % k_raw.ndim
        if k_raw.shape[dim] != len(self.fresh_candidates):
            raise ValueError("Fresh payload length must match fresh candidates")
        if not torch.isfinite(k_raw).all().item() or not torch.isfinite(v).all().item():
            raise ValueError("Nonfinite native KV")
        k_rows, v_rows = k_raw.movedim(dim, 0), v.movedim(dim, 0)
        layout = (tuple(k_rows.shape[1:]), k_raw.dtype, v.dtype, dim, k_raw.ndim)
        old_layout = self.bank._layout.get(layer_id)
        if old_layout is not None and old_layout != layout:
            raise ValueError("KV layout/dtype changed without an episode reset")
        # Clone once per fresh block. Per-ID views keep its CPU storage alive.
        # Commit accounts for actual retained storage, including partial blocks.
        cpu_k = k_rows.detach().to("cpu").contiguous().clone()
        cpu_v = v_rows.detach().to("cpu").contiguous().clone()
        staged = {c.uid: (cpu_k[i], cpu_v[i])
                  for i, c in enumerate(self.fresh_candidates)}
        old = self.bank._layers.get(layer_id, {})
        shape = (self.budget, *k_rows.shape[1:])
        out_k, out_v = k_raw.new_zeros(shape), v.new_zeros(shape)
        fresh_index = {c.uid: i for i, c in enumerate(self.fresh_candidates)}
        fresh_dest, fresh_src, old_dest, old_pairs = [], [], [], []
        for dest, uid in enumerate(self.selected_ids):
            if uid in fresh_index:
                fresh_dest.append(dest)
                fresh_src.append(fresh_index[uid])
            elif uid in old:
                old_dest.append(dest)
                old_pairs.append(old[uid])
            else:
                raise RuntimeError("Selected candidate has no payload for this layer")
        if fresh_dest:
            src = torch.tensor(fresh_src, device=k_raw.device)
            dst = torch.tensor(fresh_dest, device=k_raw.device)
            out_k.index_copy_(0, dst, k_rows.index_select(0, src))
            out_v.index_copy_(0, dst, v_rows.index_select(0, src))
        if old_dest:
            dst = torch.tensor(old_dest, device=k_raw.device)
            out_k.index_copy_(0, dst, torch.stack([x[0] for x in old_pairs]).to(k_raw))
            out_v.index_copy_(0, dst, torch.stack([x[1] for x in old_pairs]).to(v))
        self._staged[layer_id] = staged
        self._layouts[layer_id] = layout
        return (out_k.movedim(0, dim), out_v.movedim(0, dim),
                self.valid_mask.to(device=k_raw.device))

    def commit(self):
        self._check()
        if set(self._staged) != set(range(self.bank.expected_layers)):
            raise RuntimeError("Cannot commit an incomplete set of layers")
        layers = {}
        for layer in range(self.bank.expected_layers):
            payload = dict(self.bank._layers.get(layer, {}))
            payload.update(self._staged[layer])
            if set(payload) != set(self._metadata):
                raise RuntimeError("Archive metadata and layer payloads disagree")
            layers[layer] = payload
        size = _storage_bytes(layers)
        if size > self.bank.max_bytes:
            raise MemoryError("CPU KV archive limit exceeded; no candidates were evicted")
        # Publication follows all validations/allocation and is atomic to readers.
        self.bank._metadata = self._metadata
        self.bank._layers = layers
        self.bank._layout = self._layouts
        self.bank.bytes = size
        self.bank.generation += 1
        self._closed = True
        self._staged = {}

    def abort(self):
        self._staged = {}
        self._closed = True
