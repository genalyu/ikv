# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Mixture-of-Transformers (MoT) backbone for the TWAM model.

Replaces the single shared `WanTransformerBlock` stack with one *expert* stack
per modality (video / action / tactile). At every layer each expert computes its
own q/k/v from its own weights (norm1 + AdaLN + q/k/v proj + RoPE), the q/k/v are
CONCATENATED across experts, a SINGLE shared self-attention runs over the union
(the "Multimodal Shared Attention" of FastWAM, with the existing joint
causal+noise mask), then the output is split back and each expert applies its own
output proj + gate + (optional) text cross-attention + FFN.

Design choices (see discussion / repo docs):
  * Token order in the concatenated sequence is the SAME as the legacy model:
    [video(noisy,clean) | action(noisy,clean) | tactile(noisy,clean) | pad].
    Each modality block is contiguous, so the legacy flex self-attn mask applies
    verbatim — the cascade ordering "predict visual+tactile, then action" is
    already encoded in that mask's frame-id causality and is preserved unchanged.
  * Experts share num_layers / num_heads / head_dim / hidden dim (required to
    concatenate q/k/v for the shared attention); per-expert `ffn_dim` may differ.
  * Tactile expert skips text cross-attention (legacy behaviour: the cross mask
    excluded tactile queries).

Correctness: the per-expert block split reproduces the legacy block.forward, and
the MoT with weights tied to a single stack matches that shared stack run over the
whole sequence (routing / concat / split are transparent).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from .model import WanTransformerBlock, FlexAttnFunc, custom_sdpa, WanTransformer3DModel
from .global_kv_retention import GlobalKVRetention, RetentionConfig, token_rows


class SharedSelfAttention(nn.Module):
    """The one cross-expert attention per layer.

    Two interchangeable backends:
      * flex (default, GPU/train): a sparse `BlockMask` set via `set_block_mask`,
        identical to the legacy self-attention path.
      * dense (CPU / no-GPU tests): an explicit boolean mask [S, S] via
        `set_dense_mask`, routed through `custom_sdpa` (no triton/compile needed).
    When a dense mask is set it takes precedence; otherwise the flex op is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flex = FlexAttnFunc(is_cross=False)
        self._dense_mask: Optional[torch.Tensor] = None
        self.attn_caches = {}  # streaming KV-cache pools per cache_name

    def set_block_mask(self, block_mask) -> None:
        self.flex.set_block_mask(block_mask)

    def set_dense_mask(self, dense_mask: Optional[torch.Tensor]) -> None:
        self._dense_mask = dense_mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        update_cache: int = 0,
        cache_name=None,
        semantic_index=None,
        token_valid_mask=None,
        cache_transaction=None,
        cache_plan=None,
        usage_collector=None,
        cache_observation_flags=None,
    ) -> torch.Tensor:
        # q/k/v: [B(=1), S, heads, head_dim]
        # Streaming KV-cache path: when a cache pool exists for cache_name,
        # append this chunk's K/V to the rolling window and attend q over all valid
        # (committed past + current) K/V via plain SDPA — temporal causality is
        # encoded by what is committed in the pool (mirrors WanAttention cache path).
        cache = self.attn_caches.get(cache_name) if cache_name is not None else None
        if cache is not None:
            temporary = update_cache == 0
            transaction = None
            try:
                slots, current_valid, transaction = self._update_cache_with_validity(
                    cache_name,
                    k,
                    v,
                    is_pred=(update_cache == 1),
                    semantic_index=semantic_index,
                    token_valid_mask=token_valid_mask,
                    # Even a committed append is provisional until attention
                    # itself succeeds.  A caller-owned transaction keeps this
                    # compact, slot-level snapshot alive through later layers
                    # (and, in serving, through the paired action pass).
                    transactional=True,
                    cache_plan=cache_plan,
                    cache_observation_flags=cache_observation_flags,
                )
                valid = cache["mask"].nonzero(as_tuple=False).squeeze(-1)
                if valid.numel() == 0:
                    out = torch.zeros_like(q)
                else:
                    k_w = cache["k"][:, valid]
                    v_w = cache["v"][:, valid]
                    out = custom_sdpa(q, k_w, v_w)
                result = self._zero_invalid_queries(out, current_valid)
                if usage_collector is not None and not temporary and valid.numel():
                    policy, measurements = usage_collector
                    measurements.append(policy.measure_usage(q, k_w, valid, current_valid))
            except BaseException:
                # Cache writes happen before the backend call.  Restore them on
                # OOM/backend failures as well as ordinary validation errors so
                # one failed request cannot leave a committed-looking row.
                if transaction is not None:
                    self.restore_cache(cache_name, transaction)
                raise

            if temporary:
                # Denoising iterations lend slots only for this attention call.
                self.restore_cache(cache_name, transaction)
            elif cache_transaction is not None:
                # The outer MoT/request boundary owns the snapshot from here.
                cache_transaction.append((self, cache_name, transaction))
            return result

        _semantic, current_valid = self._normalise_semantic_sequence(
            semantic_index,
            k.shape[0],
            k.shape[1],
            k.device,
            token_valid_mask=token_valid_mask,
        )
        semantic_mask = None
        if current_valid is not None:
            if q.shape[:2] != current_valid.shape:
                raise ValueError(
                    "semantic validity must address the current self-attention "
                    f"queries: valid={tuple(current_valid.shape)}, "
                    f"q={tuple(q.shape[:2])}"
                )
            # True means an attention edge is allowed.  The semantic prefix is
            # sparse video; every token after it is a real tactile/action tail.
            semantic_mask = (
                current_valid[:, None, :, None] & current_valid[:, None, None, :]
            )
        if self._dense_mask is not None:
            attn_mask = self._dense_mask.to(q.device)
            if semantic_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask & semantic_mask
                else:
                    attn_mask = attn_mask.masked_fill(~semantic_mask, float("-inf"))
            out = custom_sdpa(q, k, v, attn_mask=attn_mask)
            return self._zero_invalid_queries(out, current_valid)
        if self.flex.block_mask is None:
            out = custom_sdpa(q, k, v, attn_mask=semantic_mask)
            return self._zero_invalid_queries(out, current_valid)
        out = self.flex(q, k, v)
        return self._zero_invalid_queries(out, current_valid)

    # ───── streaming KV-cache (copied from WanAttention, operates on q/k/v heads) ─────
    def init_kv_cache(
        self, cache_name, total_tolen, num_head, head_dim, device, dtype, batch_size
    ):
        self.attn_caches[cache_name] = {
            "k": torch.empty(
                [batch_size, total_tolen, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "v": torch.empty(
                [batch_size, total_tolen, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "id": torch.full((total_tolen,), -1, device=device),
            "mask": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_tolen,), dtype=torch.bool, device=device),
            # Optional per-video-token semantic sidecar.  It is allocated lazily
            # on the first RGB-motion update because DINO/NeoForce dimensions
            # belong to the data contract, not to the attention K/V width.
            "semantic": None,
        }

    def clear_cache(self, cache_name):
        self.attn_caches[cache_name] = None

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]["id"]
        mask = self.attn_caches[cache_name]["mask"]
        if mask.any():
            return ids[mask].max() + 1
        return torch.tensor(0, device=ids.device)

    def _plan_slot_allocation(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        to_free = free.new_empty(0)
        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(ids[used])
            need = key_size - free.numel()
            to_free = used[order[:need]]
            planned_free = ~mask.clone()
            planned_free[to_free] = True
            free = planned_free.nonzero(as_tuple=False).squeeze(-1)
        if free.numel() < key_size:
            raise ValueError(
                f"cache capacity {mask.numel()} is smaller than update size "
                f"{key_size}"
            )
        return free[:key_size], to_free

    @staticmethod
    def _snapshot_cache_slots(cache, slots):
        """Capture every cache field that a slot update can mutate.

        Only the selected slots are copied, rather than cloning the complete K/V
        pool on every denoising iteration.  The semantic object itself is kept so
        a lazily-created temporary sidecar can be removed on rollback and an
        existing sidecar retains its identity.
        """

        semantic = cache.get("semantic")
        return {
            "slots": slots.clone(),
            "k": cache["k"].index_select(1, slots).clone(),
            "v": cache["v"].index_select(1, slots).clone(),
            "id": cache["id"].index_select(0, slots).clone(),
            "mask": cache["mask"].index_select(0, slots).clone(),
            "is_pred": cache["is_pred"].index_select(0, slots).clone(),
            "semantic_object": semantic,
            "semantic": (
                None
                if semantic is None
                else {
                    name: value.index_select(0, slots).clone()
                    for name, value in semantic.items()
                }
            ),
        }

    @staticmethod
    def _apply_slot_allocation(cache, to_free):
        cache["mask"][to_free] = False
        cache["id"][to_free] = -1

    def allocate_slots(self, cache_name, key_size):
        slots, to_free = self._plan_slot_allocation(cache_name, key_size)
        self._apply_slot_allocation(self.attn_caches[cache_name], to_free)
        return slots

    @staticmethod
    def _normalise_semantic_index(semantic_index, batch_size, token_count, device):
        if semantic_index is None:
            return None
        required = (
            "world_time_id",
            "dino",
            "neoforce",
            "observation_flag",
            "visual_valid",
            "tactile_valid",
            "valid_mask",
        )
        missing = [name for name in required if name not in semantic_index]
        if missing:
            raise KeyError(f"semantic_index missing fields: {missing}")

        out = {}
        for name in (
            "world_time_id",
            "observation_flag",
            "visual_valid",
            "tactile_valid",
            "valid_mask",
        ):
            value = semantic_index[name]
            if value.dim() == 1:
                value = value.unsqueeze(0)
            if value.shape != (batch_size, token_count):
                raise ValueError(
                    f"semantic_index[{name!r}] must have shape "
                    f"({batch_size},{token_count}), got {tuple(value.shape)}"
                )
            out[name] = value.to(device)
        for name in ("dino", "neoforce"):
            value = semantic_index[name]
            if value.dim() == 2:
                value = value.unsqueeze(0)
            if value.dim() != 3 or value.shape[:2] != (batch_size, token_count):
                raise ValueError(
                    f"semantic_index[{name!r}] must have shape "
                    f"({batch_size},{token_count},D), got {tuple(value.shape)}"
                )
            out[name] = value.to(device)

        # Streaming CFG duplicates the same observation across batch.  Cache
        # metadata is token-major (one index per physical slot), so require all
        # copies to agree rather than silently choosing conflicting metadata.
        if batch_size > 1:
            for name, value in out.items():
                reference = value[0:1].expand_as(value)
                if torch.is_floating_point(value):
                    equal = torch.allclose(value, reference, rtol=0, atol=0)
                else:
                    equal = torch.equal(value, reference)
                if not equal:
                    raise ValueError(
                        f"semantic index differs across cache batch for {name}"
                    )
        return {name: value[0] for name, value in out.items()}

    @classmethod
    def _normalise_semantic_sequence(
        cls,
        semantic_index,
        batch_size,
        sequence_length,
        device,
        token_valid_mask=None,
    ):
        """Return semantic metadata and physical validity for one Q/K/V chunk.

        Semantic metadata covers only the leading sparse-video token block.
        Tokens after that prefix belong to tactile/action experts and remain
        physically valid even though they do not carry the RGB semantic index.
        Physical validity is a separate concern: sparse padding must still be
        omitted when semantic metadata is optional/absent.
        """
        sequence_valid = None
        physical_prefix_length = None
        if token_valid_mask is not None:
            physical = torch.as_tensor(token_valid_mask, device=device)
            if physical.dim() == 1 and batch_size == 1:
                physical = physical.unsqueeze(0)
            if physical.dim() != 2 or physical.shape[0] != batch_size:
                raise ValueError(
                    "token_valid_mask must have shape [B,N], got "
                    f"{tuple(physical.shape)} for B={batch_size}"
                )
            if physical.shape[1] > sequence_length:
                raise ValueError(
                    f"token_valid_mask length {physical.shape[1]} exceeds "
                    f"K/V length {sequence_length}"
                )
            if physical.is_complex():
                raise TypeError("token_valid_mask must be bool or binary 0/1")
            if physical.dtype != torch.bool:
                if torch.is_floating_point(physical) and not torch.isfinite(
                    physical
                ).all():
                    raise ValueError("token_valid_mask must be finite")
                if not ((physical == 0) | (physical == 1)).all():
                    raise ValueError("token_valid_mask must contain only 0 or 1")
                physical = physical.bool()
            physical_prefix_length = int(physical.shape[1])
            if batch_size > 1 and not torch.equal(
                physical, physical[0:1].expand_as(physical)
            ):
                raise ValueError(
                    "token_valid_mask differs across cache batch copies"
                )
            sequence_valid = torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=device
            )
            sequence_valid[:, :physical_prefix_length] = physical

        if semantic_index is None:
            return None, sequence_valid
        world_time = semantic_index.get("world_time_id")
        if world_time is None or world_time.dim() == 0:
            raise ValueError(
                "semantic_index['world_time_id'] must have a token dimension"
            )
        token_count = world_time.shape[-1]
        normalised = cls._normalise_semantic_index(
            semantic_index, batch_size, token_count, device
        )
        if token_count > sequence_length:
            raise ValueError(
                f"semantic token count {token_count} exceeds K/V length "
                f"{sequence_length}"
            )

        semantic_valid = normalised["valid_mask"][None].bool().expand(
            batch_size, -1
        )
        if sequence_valid is None:
            sequence_valid = torch.ones(
                batch_size, sequence_length, dtype=torch.bool, device=device
            )
            sequence_valid[:, :token_count] = semantic_valid
        else:
            if physical_prefix_length != token_count:
                raise ValueError(
                    "token_valid_mask and semantic index must address the same "
                    f"video prefix, got {physical_prefix_length} and {token_count}"
                )
            if not torch.equal(sequence_valid[:, :token_count], semantic_valid):
                raise ValueError(
                    "token_valid_mask disagrees with semantic_index.valid_mask"
                )
        return normalised, sequence_valid

    @staticmethod
    def _zero_invalid_queries(output, sequence_valid):
        if sequence_valid is None:
            return output
        if output.shape[:2] != sequence_valid.shape:
            raise ValueError(
                "semantic validity/output shape mismatch: "
                f"valid={tuple(sequence_valid.shape)}, "
                f"output={tuple(output.shape[:2])}"
            )
        return output.masked_fill(~sequence_valid[..., None, None], 0)

    @staticmethod
    def _write_semantic_sidecar(
        cache, slots, semantic_index, sequence_positions
    ):
        if semantic_index is None:
            return
        n = semantic_index["world_time_id"].shape[0]
        # `slots` contains only physically valid K/V rows.  Map the subset that
        # came from the leading semantic-video prefix back to their source rows;
        # invalid sparse padding has no physical slot at all.
        semantic_slot_mask = sequence_positions < n
        source = sequence_positions[semantic_slot_mask]
        target = slots[semantic_slot_mask]
        capacity = cache["mask"].numel()
        dino_dim = semantic_index["dino"].shape[-1]
        neo_dim = semantic_index["neoforce"].shape[-1]
        sidecar = cache.get("semantic")
        if sidecar is None:
            sidecar = {
                "valid": torch.zeros(capacity, dtype=torch.bool, device=target.device),
                "world_time_id": torch.full(
                    (capacity,), -1, dtype=torch.long, device=target.device
                ),
                "dino": torch.zeros(
                    capacity,
                    dino_dim,
                    dtype=semantic_index["dino"].dtype,
                    device=target.device,
                ),
                "neoforce": torch.zeros(
                    capacity,
                    neo_dim,
                    dtype=semantic_index["neoforce"].dtype,
                    device=target.device,
                ),
                "observation_flag": torch.zeros(
                    capacity, dtype=torch.bool, device=target.device
                ),
                "visual_valid": torch.zeros(
                    capacity, dtype=torch.bool, device=target.device
                ),
                "tactile_valid": torch.zeros(
                    capacity, dtype=torch.bool, device=target.device
                ),
            }
            cache["semantic"] = sidecar
        elif (
            sidecar["dino"].shape[-1] != dino_dim
            or sidecar["neoforce"].shape[-1] != neo_dim
        ):
            raise ValueError("semantic feature dimensions changed within one KV cache")

        valid = semantic_index["valid_mask"].index_select(0, source).bool()
        sidecar["valid"][target] = valid
        sidecar["world_time_id"][target] = semantic_index[
            "world_time_id"
        ].index_select(0, source).long()
        sidecar["dino"][target] = semantic_index["dino"].index_select(0, source)
        sidecar["neoforce"][target] = semantic_index["neoforce"].index_select(
            0, source
        )
        sidecar["observation_flag"][target] = semantic_index[
            "observation_flag"
        ].index_select(0, source).bool()
        sidecar["visual_valid"][target] = semantic_index[
            "visual_valid"
        ].index_select(0, source).bool()
        sidecar["tactile_valid"][target] = semantic_index[
            "tactile_valid"
        ].index_select(0, source).bool()

    def _update_cache_with_validity(
        self,
        cache_name,
        key,
        value,
        is_pred,
        semantic_index=None,
        token_valid_mask=None,
        transactional=False,
        cache_plan=None,
        cache_observation_flags=None,
    ):
        cache = self.attn_caches[cache_name]
        semantic_index, sequence_valid = self._normalise_semantic_sequence(
            semantic_index,
            key.shape[0],
            key.shape[1],
            key.device,
            token_valid_mask=token_valid_mask,
        )
        physical_valid = (
            torch.ones(key.shape[1], dtype=torch.bool, device=key.device)
            if sequence_valid is None
            else sequence_valid[0]
        )
        sequence_positions = physical_valid.nonzero(as_tuple=False).flatten()
        # Sparse padding must not merely be masked after allocation: asking the
        # allocator for padded rows can evict committed history when the pool is
        # full.  Pack only real current K/V rows into physical cache slots while
        # retaining the full query sequence (invalid query outputs are zeroed by
        # `_zero_invalid_queries`).
        slots, to_free = (self._plan_slot_allocation(
            cache_name, sequence_positions.numel()
        ) if cache_plan is None else cache_plan)
        if len(slots) != sequence_positions.numel():
            raise ValueError("global eviction plan/current token count mismatch")
        rollback = self._snapshot_cache_slots(cache, slots)
        transaction = rollback if transactional else None
        try:
            self._apply_slot_allocation(cache, to_free)
            if cache.get("semantic") is not None:
                cache["semantic"]["valid"][slots] = False
            new_id = self._next_cache_id(cache_name)
            cache["k"][:, slots] = key.index_select(1, sequence_positions)
            cache["v"][:, slots] = value.index_select(1, sequence_positions)
            cache["mask"][slots] = True
            cache["id"][slots] = new_id
            physical_is_pred = torch.full(
                (key.shape[1],), bool(is_pred), dtype=torch.bool, device=key.device
            )
            if semantic_index is not None:
                # The indexed video prefix may deliberately mix grounding and
                # imagination in one chunk (for example cold frame 0 observed,
                # later frames predicted).  Its source flag is authoritative per
                # token; only the unindexed tactile/action tail inherits the coarse
                # update mode supplied by the caller.
                semantic_count = semantic_index["observation_flag"].shape[0]
                physical_is_pred[:semantic_count] = ~semantic_index[
                    "observation_flag"
                ].bool()
            if cache_observation_flags is not None:
                if cache_observation_flags.shape != physical_is_pred.shape:
                    raise ValueError("cache source flags must match all tokens")
                physical_is_pred = ~cache_observation_flags.bool()
            cache["is_pred"][slots] = physical_is_pred.index_select(
                0, sequence_positions
            )
            self._write_semantic_sidecar(
                cache, slots, semantic_index, sequence_positions
            )
        except Exception:
            self.restore_cache(cache_name, rollback)
            raise
        return slots, sequence_valid, transaction

    def update_cache(
        self,
        cache_name,
        key,
        value,
        is_pred,
        semantic_index=None,
        token_valid_mask=None,
    ):
        """Append valid K/V rows and return their allocated physical slots."""
        slots, _, _ = self._update_cache_with_validity(
            cache_name,
            key,
            value,
            is_pred,
            semantic_index=semantic_index,
            token_valid_mask=token_valid_mask,
        )
        return slots

    def restore_cache(self, cache_name, transaction):
        """Roll back a temporary cache update exactly, including evictions."""

        cache = self.attn_caches[cache_name]
        slots = transaction["slots"]
        cache["k"][:, slots] = transaction["k"]
        cache["v"][:, slots] = transaction["v"]
        cache["id"][slots] = transaction["id"]
        cache["mask"][slots] = transaction["mask"]
        cache["is_pred"][slots] = transaction["is_pred"]

        original_semantic = transaction["semantic_object"]
        if original_semantic is None:
            # The temporary append may have lazily allocated a sidecar.
            cache["semantic"] = None
        else:
            cache["semantic"] = original_semantic
            for name, value in transaction["semantic"].items():
                original_semantic[name][slots] = value

    def semantic_cache(self, cache_name):
        """Return the independent semantic slot metadata for inspection."""
        cache = self.attn_caches.get(cache_name)
        return None if cache is None else cache.get("semantic")


class MoTExpert(nn.Module):
    """One modality's expert stack, optionally with a NARROW residual width.

    The residual/FFN width `hidden_dim` may be smaller than the shared
    attention/interface width `shared_dim` (= num_heads * attn_head_dim). The
    block's q/k/v still project hidden_dim -> shared_dim so they concatenate with
    the other experts in the shared attention; FFN/norm run at hidden_dim. For a
    narrow expert we add four thin projections at the expert boundary so the rest
    of the model (token embeddings, output norm, heads) stays at `shared_dim`:
        in_proj  : shared_dim -> hidden_dim       (narrow the incoming embedding)
        time_proj: shared_dim -> 6*hidden_dim     (AdaLN modulation from time vec)
        text_proj: shared_dim -> hidden_dim       (text KV for the cross-attention)
        out_proj : hidden_dim -> shared_dim       (widen the output back)
    A full expert (hidden_dim == shared_dim) keeps identities and uses the shared
    timestep_proj/text directly, so it is byte-identical to the legacy stack.
    """

    def __init__(
        self,
        shared_dim,
        hidden_dim,
        num_heads,
        attn_head_dim,
        ffn_dim,
        num_layers,
        cross_attn_norm,
        eps,
        attn_mode,
        do_cross_attn,
    ):
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.narrow = self.hidden_dim != self.shared_dim
        self.do_cross_attn = bool(do_cross_attn)
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    self.hidden_dim,
                    ffn_dim,
                    num_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode=attn_mode,
                    attn_head_dim=attn_head_dim,
                )
                for _ in range(num_layers)
            ]
        )
        if self.narrow:
            self.in_proj = nn.Linear(self.shared_dim, self.hidden_dim)
            self.out_proj = nn.Linear(self.hidden_dim, self.shared_dim)
            self.time_proj = nn.Linear(self.shared_dim, 6 * self.hidden_dim)
            self.text_proj = (
                nn.Linear(self.shared_dim, self.hidden_dim)
                if self.do_cross_attn
                else None
            )

    def embed_in(self, h):
        return self.in_proj(h) if self.narrow else h

    def embed_out(self, h):
        return self.out_proj(h) if self.narrow else h

    def modulation(self, timestep_proj_slice, time_vec_slice):
        """Per-token AdaLN modulation at hidden_dim: full experts reuse the shared
        6*shared_dim `timestep_proj`; narrow experts derive 6*hidden_dim from the
        time vector (shared_dim)."""
        if not self.narrow:
            return timestep_proj_slice
        return self.time_proj(time_vec_slice).unflatten(-1, (6, self.hidden_dim))

    def text_kv(self, text):
        if not self.do_cross_attn:
            return None
        return self.text_proj(text) if self.narrow else text


class MoTBackbone(nn.Module):
    """Per-modality experts (each a MoTExpert, possibly narrow) + one shared
    attention per layer.

    Args:
        num_layers/dim/num_heads/eps/cross_attn_norm/attn_mode: block hyper-params.
            `dim` is the SHARED interface/attention width; attn head dim = dim//num_heads.
        ffn_dim: default FFN width.
        expert_names: ordered modality names = concatenation/slice order.
        expert_ffn_dim: optional {name: ffn_dim} overrides.
        expert_hidden_dim: optional {name: hidden_dim} overrides for narrow experts
            (e.g. {"action": 1024, "tactile": 1024}); default = dim (full).
        cross_attn_experts: experts that run text cross-attention.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        attn_mode: str = "torch",
        expert_names=("video", "action", "tactile"),
        expert_ffn_dim: Optional[dict] = None,
        expert_hidden_dim: Optional[dict] = None,
        cross_attn_experts=("video", "action"),
    ) -> None:
        super().__init__()
        self.expert_names = list(expert_names)
        self.cross_attn_experts = set(cross_attn_experts)
        self.num_layers = int(num_layers)
        self.shared_dim = int(dim)
        self.attn_head_dim = dim // num_heads  # shared attention head dim
        ffn_over = dict(expert_ffn_dim or {})
        hdim_over = dict(expert_hidden_dim or {})

        self.experts = nn.ModuleDict()
        for name in self.expert_names:
            self.experts[name] = MoTExpert(
                shared_dim=dim,
                hidden_dim=int(hdim_over.get(name, dim)),
                num_heads=num_heads,
                attn_head_dim=self.attn_head_dim,
                ffn_dim=int(ffn_over.get(name, ffn_dim)),
                num_layers=self.num_layers,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
                attn_mode=attn_mode,
                do_cross_attn=(name in self.cross_attn_experts),
            )
        self.shared_attn = nn.ModuleList(
            [SharedSelfAttention() for _ in range(self.num_layers)]
        )
        self._cross_masks: dict = {}
        # Layer-level activation checkpointing. The legacy model checkpoints a WHOLE
        # block (incl. attention) so q/k/v/attn are recomputed, not stored. The MoT
        # splits each block into pre/post around the shared attention, so a per-block
        # wrapper would leave q/k/v/attn (~10GB at S~15k) resident. Checkpointing the
        # ENTIRE layer body here restores legacy-granularity AC (fixes 80GB OOM).
        self.gradient_checkpointing = True
        # Slot-level undo logs for streaming cache transactions.  Logs contain
        # only rows touched by the current request, avoiding a full clone of the
        # (potentially very large) rolling K/V pools.
        self._active_cache_transactions = {}
        self.retention_policies = {}

    @staticmethod
    def _rollback_cache_entries(entries, start=0):
        """Restore and remove transaction entries from ``start`` onward."""

        for attention, cache_name, snapshot in reversed(entries[start:]):
            attention.restore_cache(cache_name, snapshot)
        del entries[start:]

    def has_active_cache_transaction(self, cache_name):
        return cache_name in self._active_cache_transactions

    def active_cache_transaction_entries(self, cache_name):
        transaction = self._active_cache_transactions.get(cache_name)
        return None if transaction is None else transaction["entries"]

    def begin_cache_transaction(self, cache_name):
        if cache_name is None:
            raise ValueError("cache_name is required for a cache transaction")
        if self.has_active_cache_transaction(cache_name):
            raise RuntimeError(
                f"cache transaction for {cache_name!r} is already active"
            )
        policy = self.retention_policies.get(cache_name)
        transaction = {"cache_name": cache_name, "entries": [],
                       "retention_snapshot": None if policy is None else policy.snapshot()}
        self._active_cache_transactions[cache_name] = transaction
        return transaction

    def _require_active_cache_transaction(self, transaction):
        cache_name = transaction.get("cache_name")
        if self._active_cache_transactions.get(cache_name) is not transaction:
            raise RuntimeError("cache transaction is not active")
        return cache_name

    def commit_cache_transaction(self, transaction):
        cache_name = self._require_active_cache_transaction(transaction)
        transaction["entries"].clear()
        del self._active_cache_transactions[cache_name]

    def rollback_cache_transaction(self, transaction):
        cache_name = self._require_active_cache_transaction(transaction)
        try:
            self._rollback_cache_entries(transaction["entries"])
        finally:
            if transaction.get("retention_snapshot") is not None:
                self.retention_policies[cache_name].restore(transaction["retention_snapshot"])
            del self._active_cache_transactions[cache_name]

    @contextmanager
    def cache_transaction(self, cache_name):
        """Atomically commit every cache append performed inside the context."""

        transaction = self.begin_cache_transaction(cache_name)
        try:
            yield transaction
        except BaseException:
            self.rollback_cache_transaction(transaction)
            raise
        else:
            self.commit_cache_transaction(transaction)

    # ───────────────────────── mask wiring ─────────────────────────
    def set_masks(self, self_block_mask=None, dense_self_mask=None, cross_masks=None):
        """Set the shared self-attn mask (all layers) and per-expert cross masks.
        `self_block_mask` (flex, GPU) OR `dense_self_mask` (bool [S,S], CPU);
        `cross_masks` maps expert name -> dense bool (1,1,S_q,S_text)."""
        for layer in range(self.num_layers):
            if dense_self_mask is not None:
                self.shared_attn[layer].set_dense_mask(dense_self_mask)
            else:
                self.shared_attn[layer].set_dense_mask(None)
                self.shared_attn[layer].set_block_mask(self_block_mask)
        self._cross_masks = dict(cross_masks or {})

    # ───────────────────────── forward ─────────────────────────
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        temb,
        rotary_emb,
        slices,
        collect_cache=False,
        update_cache=0,
        cache_name=None,
        semantic_index=None,
        token_valid_mask=None,
        cache_metadata=None,
    ):
        """Run the MoT stack.

        Args:
            hidden_states: [1, S_total, shared_dim] — modality blocks concatenated
                in `slices` order (pad folded into the last expert's slice).
            encoder_hidden_states: [1, L_text, shared_dim] text KV.
            timestep_proj: [1, S_total, 6, shared_dim] AdaLN modulation (full experts).
            temb: [1, S_total, shared_dim] time VECTOR (narrow experts' time_proj
                source); may be None when all experts are full.
            rotary_emb: [1, S_total, 1, head_dim//2] complex RoPE freqs.
            slices: ordered list of (expert_name, start, end) spanning S_total.

        Returns:
            [1, S_total, shared_dim] updated, same layout as input.
        """
        names = [s[0] for s in slices]
        if names != self.expert_names:
            raise ValueError(
                f"slice order {names} must equal expert order {self.expert_names}"
            )

        # 1. split + per-expert narrow projection / modulation / text / rope
        streams, mod, text, rope_e, seg_len = {}, {}, {}, {}, {}
        for name, s, e in slices:
            ex = self.experts[name]
            streams[name] = ex.embed_in(hidden_states[:, s:e])
            mod[name] = ex.modulation(
                timestep_proj[:, s:e], temb[:, s:e] if temb is not None else None
            )
            text[name] = ex.text_kv(encoder_hidden_states)
            rope_e[name] = rotary_emb[:, s:e]
            seg_len[name] = e - s

        # 2. layers: per-expert pre -> concat q/k/v -> shared attn -> per-expert post.
        # The ENTIRE layer body is one activation-checkpoint unit (legacy granularity):
        # in backward, q/k/v/attn/ffn are recomputed rather than held resident.
        kv_cache = [] if collect_cache else None
        order = [name for (name, _s, _e) in slices]
        cache_transaction = None
        owns_cache_transaction = False
        policy = self.retention_policies.get(cache_name)
        cache_plan = None
        measurements = []
        policy_before = None
        old_mask = None
        rows = None
        if policy is not None:
            if cache_metadata is None:
                raise ValueError("global retention requires cache token metadata")
            first = self.shared_attn[0].attn_caches[cache_name]
            for attention in self.shared_attn[1:]:
                other = attention.attn_caches[cache_name]
                if not torch.equal(first["mask"], other["mask"]) or not torch.equal(first["id"], other["id"]):
                    raise RuntimeError("layer cache slots diverged; cannot apply a global eviction plan")
            _, validity = SharedSelfAttention._normalise_semantic_sequence(
                semantic_index, hidden_states.shape[0], hidden_states.shape[1],
                hidden_states.device, token_valid_mask=token_valid_mask)
            positions = (torch.arange(hidden_states.shape[1], device=hidden_states.device)
                         if validity is None else validity[0].nonzero().flatten())
            rows = {name: value[positions] for name, value in cache_metadata.items()}
            old_mask = first["mask"].clone()
            policy_before = policy.snapshot() if update_cache else None
            cache_plan = policy.plan(old_mask, len(positions), rows)

        # Preflight above is read-only and must not leave an open transaction
        # when metadata validation fails.
        if update_cache != 0 and cache_name is not None:
            cache_transaction = self._active_cache_transactions.get(cache_name)
            if cache_transaction is None:
                cache_transaction = self.begin_cache_transaction(cache_name)
                owns_cache_transaction = True
        cache_entries = None if cache_transaction is None else cache_transaction["entries"]
        cache_entry_start = 0 if cache_entries is None else len(cache_entries)

        def _layer(layer, *streams_in):
            sl_in = {order[i]: streams_in[i] for i in range(len(order))}
            q_chunks, k_chunks, v_chunks, post = [], [], [], []
            for name in order:
                block = self.experts[name].blocks[layer]
                q, k, v, residual, gate, cs, csc, cg = block(
                    sl_in[name], temb=mod[name], rotary_emb=rope_e[name], mot_mode="pre"
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                post.append((name, block, residual, gate, cs, csc, cg))
            q = torch.cat(q_chunks, dim=1)
            k = torch.cat(k_chunks, dim=1)
            v = torch.cat(v_chunks, dim=1)
            if collect_cache:
                kv_cache.append((k, v))  # full-sequence K/V at this layer
            # update_cache/cache_name default to 0/None in training (no streaming cache;
            # SharedSelfAttention falls back to normal attn) and carry the streaming
            # KV-cache args at inference. Captured from the enclosing forward scope.
            attn = self.shared_attn[layer](
                q,
                k,
                v,
                update_cache=update_cache,
                cache_name=cache_name,
                semantic_index=semantic_index,
                token_valid_mask=token_valid_mask,
                cache_transaction=cache_entries,
                cache_plan=cache_plan,
                usage_collector=None if policy is None else (policy, measurements),
                cache_observation_flags=(None if cache_metadata is None
                                         else cache_metadata["observation_flag"]),
            )
            out = {}
            cursor = 0
            for name, block, residual, gate, cs, csc, cg in post:
                sl = attn[:, cursor : cursor + seg_len[name]]
                do_cross = self.experts[name].do_cross_attn and (text[name] is not None)
                out[name] = block(
                    residual,
                    mot_mode="post",
                    mot_kwargs=dict(
                        attn_output=sl,
                        gate_msa=gate,
                        c_shift_msa=cs,
                        c_scale_msa=csc,
                        c_gate_msa=cg,
                        encoder_hidden_states=text[name],
                        do_cross_attn=do_cross,
                        cross_attn_mask=self._cross_masks.get(name),
                    ),
                )
                cursor += seg_len[name]
            return tuple(out[name] for name in order)

        use_ckpt = (
            self.gradient_checkpointing
            and torch.is_grad_enabled()
            and not collect_cache
        )
        try:
            for layer in range(self.num_layers):
                cur = tuple(streams[name] for name in order)
                new = (
                    _ckpt(_layer, layer, *cur, use_reentrant=False)
                    if use_ckpt
                    else _layer(layer, *cur)
                )
                streams = {order[i]: new[i] for i in range(len(order))}

            # 3. widen each expert back to shared_dim and concatenate in slice order
            out = torch.cat(
                [
                    self.experts[name].embed_out(streams[name])
                    for (name, _s, _e) in slices
                ],
                dim=1,
            )
            if policy is not None and update_cache:
                policy.commit(cache_plan[0], rows, old_mask)
                policy.add_usage(measurements)
        except BaseException:
            if policy_before is not None:
                policy.restore(policy_before)
            if cache_transaction is not None:
                if owns_cache_transaction:
                    self.rollback_cache_transaction(cache_transaction)
                else:
                    self._rollback_cache_entries(
                        cache_entries, start=cache_entry_start
                    )
            raise
        else:
            if owns_cache_transaction:
                self.commit_cache_transaction(cache_transaction)
        if collect_cache:
            return out, kv_cache
        return out

    @torch.no_grad()
    def forward_action_cached(
        self,
        a_noisy_stream,
        a_timestep_proj,
        a_temb,
        a_rope,
        kv_cache,
        a_cols,
        rows_mask,
        encoder_hidden_states=None,
        cross_attn_mask=None,
    ):
        """KV-cache fast action denoising: only the action-noisy tokens are
        recomputed each step; the fixed upstream (video / tactile / action-clean)
        K/V come from `kv_cache` (per-layer full-sequence K/V from a prefill forward
        with collect_cache=True). For every layer we recompute the a_noisy q/k/v,
        splice the new a_noisy K/V into the cached full-sequence K/V at columns
        `a_cols`, attend (a_noisy queries × all columns via `rows_mask`), and run the
        action block's post. Equivalent to the full joint forward's a_noisy output
        (verified) but skips the heavy video/tactile expert recompute per step.

        Args:
            a_noisy_stream: [1, S_anoisy, shared_dim] the action-noisy tokens.
            a_*: modulation/time/rope SLICES for the a_noisy tokens.
            kv_cache: list per layer of (k_full, v_full) [1, S_total, heads, dh].
            a_cols: slice for the a_noisy columns within S_total.
            rows_mask: dense bool [S_anoisy, S_total] (a_noisy queries × all cols).
        Returns: [1, S_anoisy, shared_dim] widened a_noisy output.
        """
        ex = self.experts["action"]
        stream = ex.embed_in(a_noisy_stream)
        a_mod = ex.modulation(a_timestep_proj, a_temb)  # per-expert AdaLN (own dim)
        text = ex.text_kv(encoder_hidden_states)  # action's text KV (own dim)
        do_cross = ex.do_cross_attn and (text is not None)
        for layer in range(self.num_layers):
            block = ex.blocks[layer]
            q, k, v, residual, gate, cs, csc, cg = block(
                stream, temb=a_mod, rotary_emb=a_rope, mot_mode="pre"
            )
            k_full, v_full = kv_cache[layer]
            k_full = k_full.clone()
            v_full = v_full.clone()
            k_full[:, a_cols] = k  # splice fresh a_noisy K/V
            v_full[:, a_cols] = v
            attn = custom_sdpa(q, k_full, v_full, attn_mask=rows_mask)
            stream = block(
                residual,
                mot_mode="post",
                mot_kwargs=dict(
                    attn_output=attn,
                    gate_msa=gate,
                    c_shift_msa=cs,
                    c_scale_msa=csc,
                    c_gate_msa=cg,
                    encoder_hidden_states=text,
                    do_cross_attn=do_cross,
                    cross_attn_mask=cross_attn_mask,
                ),
            )
        return ex.embed_out(stream)


# ─────────────────────────── MoT model (3-expert) ───────────────────────────
class WanMoTTransformer3DModel(WanTransformer3DModel):
    """Mixture-of-Transformers variant of WanTransformer3DModel.

    The single shared `blocks` stack is replaced by three per-modality expert
    stacks (video / action / tactile) joined by one shared attention per layer.
    EVERYTHING ELSE is inherited unchanged from the parent — embeddings, text/time
    conditioning, GlobalTactile diffusion head, LocalTactile cross-attn, contact
    gate, output norm, losses — only the backbone execution differs, via the
    overridden `_run_backbone` hook.

    Cascade / ordering is preserved exactly: the concatenated sequence keeps the
    legacy layout [video | action | tactile | pad] and the legacy joint flex mask,
    so "predict visual+tactile, then action" still holds (it lives in the mask's
    frame-id causality, not in the weight split).

    Notes:
      * Experts share num_layers/num_heads/head_dim/dim (required for the shared
        attention); per-expert `mot_expert_ffn_dim` overrides are allowed but then
        warm-starting from a shared checkpoint can only copy the matching-width
        experts (see remap_shared_to_mot_state_dict).
      * Inference uses a streaming KV-cache (implemented below).
    """

    def __init__(
        self,
        *args,
        mot_expert_ffn_dim=None,
        mot_expert_hidden_dim=None,
        mot_cross_attn_experts=("video", "action"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        cfg = self.config
        inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.mot_expert_names = ("video", "action", "tactile")
        self.mot = MoTBackbone(
            num_layers=cfg.num_layers,
            dim=inner_dim,
            ffn_dim=cfg.ffn_dim,
            num_heads=cfg.num_attention_heads,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
            attn_mode="torch",
            expert_names=self.mot_expert_names,
            expert_ffn_dim=mot_expert_ffn_dim,
            expert_hidden_dim=mot_expert_hidden_dim,  # e.g. {"action":1024,"tactile":1024}
            cross_attn_experts=mot_cross_attn_experts,
        )
        # the shared stack is superseded by the per-expert stacks
        del self.blocks
        # Persist the MoT-specific args into config.json so a saved checkpoint can be
        # rebuilt with the SAME expert structure for inference (see load_mot_checkpoint).
        self.register_to_config(
            is_mot=True,
            mot_expert_ffn_dim=mot_expert_ffn_dim,
            mot_expert_hidden_dim=mot_expert_hidden_dim,
            mot_cross_attn_experts=list(mot_cross_attn_experts),
        )

    def cache_transaction(self, cache_name):
        """Return a context that groups multiple model calls into one KV commit."""

        return self.mot.cache_transaction(cache_name)

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        # A normal committed model call is atomic through its output head, not
        # merely through the last shared-attention layer.  Serving can open a
        # wider transaction around its paired video/action calls; in that case
        # reuse the active log instead of nesting another transaction.
        needs_transaction = (
            not train_mode
            and update_cache != 0
            and cache_name is not None
            and not self.mot.has_active_cache_transaction(cache_name)
        )
        if not needs_transaction:
            return super().forward(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
                train_mode=train_mode,
            )
        with self.cache_transaction(cache_name):
            return super().forward(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
                train_mode=train_mode,
            )

    # ───────────────── backbone hook (the only forward change) ─────────────────
    def _run_backbone(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        self_attention_mask,
        cross_attention_mask,
        split_list,
        batch_size,
        temb=None,
    ):
        # split_list = [v_noisy, v_clean, a_noisy, a_clean, t_noisy, t_clean, pad]
        v = split_list[0] + split_list[1]
        a = split_list[2] + split_list[3]
        t = split_list[4] + split_list[5] + split_list[6]  # tactile absorbs the pad
        slices = [
            ("video", 0, v),
            ("action", v, v + a),
            ("tactile", v + a, v + a + t),
        ]
        if v + a + t != hidden_states.shape[1]:
            raise ValueError(
                f"MoT slices sum ({v + a + t}) != sequence length "
                f"({hidden_states.shape[1]}); split_list={split_list}"
            )
        text_len = (
            encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        )
        cross_masks = self._mot_cross_masks(
            split_list, batch_size, text_len, device=hidden_states.device
        )
        self.mot.set_masks(self_block_mask=self_attention_mask, cross_masks=cross_masks)
        return self.mot(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            rotary_emb,
            slices,
        )

    def _mot_cross_masks(self, split_list, batch_size, text_len, device):
        """Dense bool cross-attn masks (1,1,S_q,S_text) enforcing the legacy
        batch-isolation: a query token only attends text of its own (packed) batch.
        For batch_size==1 there is nothing to isolate -> no mask (attend all text),
        matching the legacy single-batch behaviour."""
        if batch_size <= 1 or text_len == 0:
            return {}
        if text_len % batch_size != 0:
            raise ValueError(
                f"text_len ({text_len}) not divisible by batch_size ({batch_size})"
            )
        text_per = text_len // batch_size
        t_batch = torch.arange(batch_size, device=device).repeat_interleave(text_per)

        def q_batch(seg_lens):
            chunks = []
            for seg in seg_lens:
                if seg == 0:
                    continue
                if seg % batch_size != 0:
                    raise ValueError(
                        f"segment length {seg} not divisible by batch_size {batch_size}"
                    )
                chunks.append(
                    torch.arange(batch_size, device=device).repeat_interleave(
                        seg // batch_size
                    )
                )
            if not chunks:
                return torch.empty(0, dtype=torch.long, device=device)
            return torch.cat(chunks)

        masks = {}
        seg_map = {"video": split_list[0:2], "action": split_list[2:4]}
        for name, seg in seg_map.items():
            if name not in self.mot.cross_attn_experts:
                continue
            qb = q_batch(seg)
            masks[name] = (qb[:, None] == t_batch[None, :])[
                None, None
            ]  # (1,1,S_q,S_text)
        return masks

    # ─────────── cache hooks (self.blocks is gone; inference = Phase 2) ───────────
    def _expert_blocks(self):
        for name in self.mot.expert_names:
            for blk in self.mot.experts[name].blocks:
                yield blk

    def clear_cache(self, cache_name):
        for sa in self.mot.shared_attn:
            sa.clear_cache(cache_name)
        getattr(self.mot, "retention_policies", {}).pop(cache_name, None)

    def configure_global_retention(self, cache_name, **config):
        """Enable shared global top-k protection + random remainder eviction.

        Must be configured on an empty cache. No model weights are introduced.
        """
        if self.use_rgb_motion_tokens:
            raise ValueError("global index mode requires full RGB tokens; disable use_rgb_motion_tokens")
        caches = [sa.attn_caches[cache_name] for sa in self.mot.shared_attn]
        if not caches or any(c["mask"].any() for c in caches):
            raise ValueError("configure retention on a new, empty multi-layer KV cache")
        cfg = RetentionConfig(**config)
        self.mot.retention_policies[cache_name] = GlobalKVRetention(
            caches[0]["mask"].numel(), caches[0]["mask"].device, cfg)

    def get_global_retention(self, cache_name):
        """Detached, token-major diagnostics, shared by every expert/layer."""
        policy = self.mot.retention_policies.get(cache_name)
        if policy is None:
            return None
        mask = self.mot.shared_attn[0].attn_caches[cache_name]["mask"]
        slots = mask.nonzero().flatten()
        return {"t0": policy.t0, "slot_indices": slots.clone(),
                **{k: v[slots].detach().clone() for k, v in policy.data.items()},
                "score": policy.scores(slots).detach().clone(),
                "components": {k: v.detach().clone()
                               for k, v in policy.components(slots).items()}}

    def global_cache_cursor(self, cache_name):
        """Capture before a forward to identify its newly committed tokens."""
        return self.mot.retention_policies[cache_name].next_uid

    def video_index_handle(self, cache_name, since_uid):
        policy = self.mot.retention_policies[cache_name]
        mask = self.mot.shared_attn[0].attn_caches[cache_name]['mask']
        return policy.video_handle(mask, since_uid)

    def annotate_video_dino(self, cache_name, handle, features):
        """One metadata write serves all layers. No forward or KV allocation."""
        policy = self.mot.retention_policies[cache_name]
        if handle['owner'] is not policy:
            raise ValueError('video index handle belongs to a different cache generation')
        for attention in self.mot.shared_attn:
            mask = attention.attn_caches[cache_name]['mask']
            if not mask[handle['slots']].all():
                raise ValueError('video index handle addresses removed layer KV')
        return policy.annotate_video_dino(mask, handle, features)

    def get_semantic_cache(self, cache_name, layer=0, *, valid_only=True):
        """Return a read-only snapshot of one layer's semantic KV sidecar.

        The snapshot contains only index metadata -- ``world_time_id``,
        ``dino``, ``neoforce``, ``observation_flag``, modality-presence masks,
        and semantic validity -- plus ``slot_indices`` identifying the physical
        KV slots.  Attention K/V and content embeddings are deliberately not
        exposed by this API.

        Every returned tensor is detached and cloned.  Mutating the returned
        dictionary or its tensors therefore cannot mutate the live cache.
        With ``valid_only=True`` (the default), rows without a semantic index
        -- including unindexed tactile/action KV rows -- are omitted.  ``None``
        is returned when the named KV pool or its semantic sidecar has not been
        initialized.

        Args:
            cache_name: Streaming cache pool name passed to
                :meth:`create_empty_cache`.
            layer: Zero-based shared-attention layer to inspect.
            valid_only: Select only active semantic rows when true; when false,
                return the full sidecar capacity together with its ``valid``
                mask.
        """

        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer must be an integer, got {type(layer).__name__}")
        num_layers = len(self.mot.shared_attn)
        if layer < 0 or layer >= num_layers:
            raise IndexError(
                f"semantic cache layer {layer} is outside [0, {num_layers})"
            )
        if not isinstance(valid_only, bool):
            raise TypeError("valid_only must be bool")

        sidecar = self.mot.shared_attn[layer].semantic_cache(cache_name)
        if sidecar is None:
            return None
        valid = sidecar["valid"]
        slots = (
            torch.nonzero(valid, as_tuple=False).flatten()
            if valid_only
            else torch.arange(valid.numel(), device=valid.device)
        )
        snapshot = {"slot_indices": slots.detach().clone()}
        for name in (
            "valid",
            "world_time_id",
            "dino",
            "neoforce",
            "observation_flag",
            "visual_valid",
            "tactile_valid",
        ):
            snapshot[name] = sidecar[name].index_select(0, slots).detach().clone()
        return snapshot

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
    ):
        # Phase-2 streaming KV-cache: pool lives on each shared cross-expert
        # attention (the actual attention site in MoT), sized like the legacy path.
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        for sa in self.mot.shared_attn:
            sa.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )
        getattr(self.mot, "retention_policies", {}).pop(cache_name, None)

    def _run_main_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        temb,
        rotary_emb,
        update_cache,
        cache_name,
        action_mode,
        main_token_count,
        tactile_token_count,
        semantic_index=None,
        token_valid_mask=None,
        cache_context=None,
    ):
        # MoT streaming inference: the [main, tactile] sequence -> per-modality slices.
        # One of video/action is empty per pass (video pass: action empty; action pass:
        # video empty); the apply_rotary_emb 0-length guard + empty-slice handling cope.
        # Runs the expert cascade with the shared-attn streaming KV cache.
        m, t = int(main_token_count), int(tactile_token_count)
        if action_mode:
            slices = [("video", 0, 0), ("action", 0, m), ("tactile", m, m + t)]
        else:
            slices = [("video", 0, m), ("action", m, m), ("tactile", m, m + t)]
        metadata = None
        policy_enabled = cache_name in getattr(self.mot, "retention_policies", {})
        if cache_context is not None and (policy_enabled or cache_context.get("index") is not None):
            metadata = token_rows(
                cache_context, batch_size=hidden_states.shape[0], length=m+t,
                main_count=m, action_mode=action_mode, update_cache=update_cache,
                device=hidden_states.device)
            # Dense indexing has no sparse gather and no nonzero-presence
            # requirement. Features are kept out of hidden states and Q/K/V.
            if semantic_index is None and not policy_enabled:
                batch = hidden_states.shape[0]
                semantic_index = {
                    name: metadata[name][None].expand(batch, *metadata[name].shape)
                    for name in ("world_time_id", "dino", "neoforce", "observation_flag")}
                semantic_index.update(
                    visual_valid=(semantic_index["dino"] != 0).any(-1),
                    tactile_valid=(semantic_index["neoforce"] != 0).any(-1),
                    valid_mask=torch.ones(batch, m+t, dtype=torch.bool, device=hidden_states.device))
        return self.mot(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            temb,
            rotary_emb,
            slices,
            update_cache=update_cache,
            cache_name=cache_name,
            semantic_index=semantic_index,
            token_valid_mask=token_valid_mask,
            **({"cache_metadata": metadata} if metadata is not None else {}),
        )


def remap_shared_to_mot_state_dict(
    legacy_sd, expert_names=("video", "action", "tactile")
):
    """Warm-start: turn a legacy (shared-backbone) state_dict into a MoT one by
    copying each `blocks.{i}.*` tensor into every expert `mot.experts.{name}.blocks.{i}.*`.
    All non-block keys (embeddings, heads, contact gate, output norm, ...) are kept
    verbatim. NOTE: this assumes the listed experts are FULL-width (== shared dim);
    narrow experts cannot receive the wide WAN blocks and must be left random (use
    the in-place warm-start in utils.load_mot_transformer, which skips them). Load
    with strict=False (shared_attn / narrow-expert projections have no source)."""
    new_sd = {}
    prefix = "blocks."
    for key, value in legacy_sd.items():
        if key.startswith(prefix):
            rest = key[len(prefix) :]  # "{layer}.{...}"
            for name in expert_names:
                new_sd[f"mot.experts.{name}.blocks.{rest}"] = value.clone()
        else:
            new_sd[key] = value
    return new_sd
