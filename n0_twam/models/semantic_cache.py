# Copyright 2025-2026 NeoteAI Team. All rights reserved.
"""Semantic metadata and selection for a streaming KV cache.

The attention key/value tensors in this module are deliberately kept separate
from the semantic index

    i = {world_time_id, DINO, NeoForce, observation_flag}.

``visual_valid`` and ``tactile_valid`` are auxiliary presence masks.  They say
whether DINO/NeoForce exists for a token; they are not additional index fields.
No index feature is projected into, concatenated with, or added to K/V here.

The ``update_cache`` modes mirror N0-TWAM's streaming convention:

* ``0``: append a temporary current chunk and later roll it back;
* ``1``: commit predicted tokens (``observation_flag == 0``);
* ``2``: commit observed tokens (``observation_flag == 1``), replacing
  semantically matched predictions at the same world time when possible.

The class is intentionally independent of ``mot.py`` so it can first be tested
as a metadata/selection component and then wired into shared attention without
changing the meaning of the original content embeddings.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional, Union

import torch
import torch.nn.functional as F


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class SemanticIndex:
    """Per-token semantic index, plus modality-presence metadata.

    Shapes are token-major and do not include the K/V batch dimension:

    * ``world_time_id``: ``[N]`` integer world/model time (not diffusion time);
    * ``dino``: ``[N, dino_dim]`` or ``None``;
    * ``neoforce``: ``[N, neoforce_dim]`` or ``None``;
    * ``observation_flag``: ``[N]``, where 1/True means observed and 0/False
      means predicted;
    * presence masks: ``[N]`` booleans.  If omitted, a provided feature is
      considered present for every token and a missing feature absent for every
      token.

    At least one of DINO and NeoForce must be present for every token.
    """

    world_time_id: torch.Tensor
    dino: Optional[torch.Tensor]
    neoforce: Optional[torch.Tensor]
    observation_flag: torch.Tensor
    visual_valid: Optional[torch.Tensor] = None
    tactile_valid: Optional[torch.Tensor] = None

    @property
    def num_tokens(self) -> int:
        if self.world_time_id.ndim != 1:
            raise ValueError("world_time_id must have shape [N]")
        return int(self.world_time_id.shape[0])


@dataclass(frozen=True)
class PredictionMatchConfig:
    """Similarity rule used when an observation replaces a prediction."""

    dino_weight: float = 1.0
    neoforce_weight: float = 1.0
    min_similarity: float = 0.8

    def __post_init__(self) -> None:
        if self.dino_weight < 0 or self.neoforce_weight < 0:
            raise ValueError("prediction-match weights must be non-negative")
        if self.dino_weight == 0 and self.neoforce_weight == 0:
            raise ValueError("at least one prediction-match weight must be positive")
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must lie in [-1, 1]")


@dataclass(frozen=True)
class ImportanceWeights:
    """Independent weights for semantic cache selection."""

    time: float = 1.0
    dino: float = 1.0
    neoforce: float = 1.0
    observation: float = 1.0

    def __post_init__(self) -> None:
        values = (self.time, self.dino, self.neoforce, self.observation)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("importance weights must be finite")


@dataclass(frozen=True)
class ImportanceComponents:
    """Unmixed importance terms for physical cache ``slots``.

    ``time`` is exponential closeness to the nearest query time. ``dino`` and
    ``neoforce`` are the maximum cosine similarity to a compatible query token;
    they are zero when that modality cannot be compared. ``observation`` is 1
    for an observed cache token and 0 for a predicted one.
    """

    slots: torch.Tensor
    time: torch.Tensor
    dino: torch.Tensor
    neoforce: torch.Tensor
    observation: torch.Tensor

    def weighted(
        self, weights: Union[ImportanceWeights, Mapping[str, float]]
    ) -> torch.Tensor:
        weights = _coerce_weights(weights)
        return (
            self.time * weights.time
            + self.dino * weights.dino
            + self.neoforce * weights.neoforce
            + self.observation * weights.observation
        )

    def take(self, positions: torch.Tensor) -> "ImportanceComponents":
        """Select positions within this component collection."""

        return ImportanceComponents(
            slots=self.slots[positions],
            time=self.time[positions],
            dino=self.dino[positions],
            neoforce=self.neoforce[positions],
            observation=self.observation[positions],
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "time": self.time,
            "dino": self.dino,
            "neoforce": self.neoforce,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class SemanticKVView:
    """A gathered, ordered view; content K/V and index remain separate."""

    key: torch.Tensor
    value: torch.Tensor
    index: SemanticIndex
    slots: torch.Tensor
    insertion_ids: torch.Tensor


@dataclass(frozen=True)
class SemanticKVSelection:
    """Top-k cache selection and the scores that selected it."""

    key: torch.Tensor
    value: torch.Tensor
    index: SemanticIndex
    slots: torch.Tensor
    insertion_ids: torch.Tensor
    scores: torch.Tensor
    components: ImportanceComponents


@dataclass(frozen=True)
class _NormalisedIndex:
    world_time_id: torch.Tensor
    dino: torch.Tensor
    neoforce: torch.Tensor
    observation_flag: torch.Tensor
    visual_valid: torch.Tensor
    tactile_valid: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return int(self.world_time_id.shape[0])


@dataclass(frozen=True)
class _SlotSnapshot:
    slots: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor
    insertion_ids: torch.Tensor
    world_time_id: torch.Tensor
    dino: torch.Tensor
    neoforce: torch.Tensor
    observation_flag: torch.Tensor
    visual_valid: torch.Tensor
    tactile_valid: torch.Tensor


@dataclass(frozen=True)
class CacheUpdate:
    """Receipt returned by :meth:`SemanticKVCache.append`.

    ``slots`` follows input-token order. ``replaced_slots`` contains committed
    prediction slots reused by observed tokens, while ``evicted_slots`` contains
    old FIFO slots displaced for capacity.  A mode-0 receipt must be passed to
    :meth:`rollback`, unless :meth:`temporary_append` is used as a context
    manager.
    """

    slots: torch.Tensor
    replaced_slots: torch.Tensor
    evicted_slots: torch.Tensor
    update_cache: int
    _snapshot: _SlotSnapshot = field(repr=False)
    _next_insertion_id_before: int = field(repr=False)
    _revision_after: int = field(repr=False)

    @property
    def temporary(self) -> bool:
        return self.update_cache == SemanticKVCache.TEMPORARY


def _coerce_weights(
    weights: Union[ImportanceWeights, Mapping[str, float]],
) -> ImportanceWeights:
    if isinstance(weights, ImportanceWeights):
        return weights
    unknown = set(weights) - {"time", "dino", "neoforce", "observation"}
    if unknown:
        raise ValueError(f"unknown importance components: {sorted(unknown)}")
    return ImportanceWeights(**dict(weights))


class SemanticKVCache:
    """Fixed-capacity K/V pool with a separate semantic index.

    K/V have shape ``[B, N, ...]``.  One semantic index row is associated with
    token position ``N`` and is shared across the batch, matching the current
    streaming N0-TWAM use case (normally ``B == 1``).  K/V storage is allocated
    lazily from the first append, while semantic storage is fixed at
    construction.

    Capacity pressure uses FIFO eviction for compatibility with the existing
    cache.  Semantic compression is explicit through :meth:`select_topk`, so
    eviction policy and research-time importance scoring are not conflated.
    """

    TEMPORARY = 0
    PREDICTED = 1
    OBSERVED = 2

    def __init__(
        self,
        capacity: int,
        dino_dim: int,
        neoforce_dim: int,
        *,
        device: Union[str, torch.device] = "cpu",
        feature_dtype: torch.dtype = torch.float32,
        prediction_match: Optional[PredictionMatchConfig] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if dino_dim < 0 or neoforce_dim < 0:
            raise ValueError("dino_dim and neoforce_dim must be non-negative")
        if dino_dim == 0 and neoforce_dim == 0:
            raise ValueError("DINO and NeoForce dimensions cannot both be zero")
        if not feature_dtype.is_floating_point:
            raise ValueError("feature_dtype must be floating point")

        self.capacity = int(capacity)
        self.dino_dim = int(dino_dim)
        self.neoforce_dim = int(neoforce_dim)
        self.device = torch.device(device)
        self.feature_dtype = feature_dtype
        self.prediction_match = prediction_match or PredictionMatchConfig()

        self.mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.insertion_ids = torch.full(
            (self.capacity,), -1, dtype=torch.long, device=self.device
        )
        self.world_time_id = torch.full(
            (self.capacity,), -1, dtype=torch.long, device=self.device
        )
        self.dino = torch.zeros(
            self.capacity,
            self.dino_dim,
            dtype=self.feature_dtype,
            device=self.device,
        )
        self.neoforce = torch.zeros(
            self.capacity,
            self.neoforce_dim,
            dtype=self.feature_dtype,
            device=self.device,
        )
        self.observation_flag = torch.zeros(
            self.capacity, dtype=torch.bool, device=self.device
        )
        self.visual_valid = torch.zeros(
            self.capacity, dtype=torch.bool, device=self.device
        )
        self.tactile_valid = torch.zeros(
            self.capacity, dtype=torch.bool, device=self.device
        )

        self._key: Optional[torch.Tensor] = None
        self._value: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None
        self._next_insertion_id = 0
        self._revision = 0
        self._active_temporary: Optional[CacheUpdate] = None

    def __len__(self) -> int:
        return int(self.mask.sum().item())

    @property
    def initialized(self) -> bool:
        return self._key is not None

    @property
    def key_buffer(self) -> Optional[torch.Tensor]:
        """Raw K storage.  It never contains semantic index features."""

        return self._key

    @property
    def value_buffer(self) -> Optional[torch.Tensor]:
        """Raw V storage.  It never contains semantic index features."""

        return self._value

    def clear(self) -> None:
        self._require_no_temporary_update()
        self.mask.zero_()
        self.insertion_ids.fill_(-1)
        self.world_time_id.fill_(-1)
        self.observation_flag.zero_()
        self.visual_valid.zero_()
        self.tactile_valid.zero_()
        self.dino.zero_()
        self.neoforce.zero_()
        self._next_insertion_id = 0
        self._revision += 1

    def clear_predictions(self) -> torch.Tensor:
        """Remove committed predicted entries and return their physical slots."""

        self._require_no_temporary_update()
        slots = (self.mask & ~self.observation_flag).nonzero(as_tuple=False).flatten()
        self.mask[slots] = False
        self.insertion_ids[slots] = -1
        self.world_time_id[slots] = -1
        self.visual_valid[slots] = False
        self.tactile_valid[slots] = False
        if slots.numel():
            self._revision += 1
        return slots

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index: SemanticIndex,
        *,
        update_cache: int,
        replace_predictions: bool = True,
    ) -> CacheUpdate:
        """Append temporary, predicted, or observed K/V plus separate metadata.

        Mode 0 is transactional: no other mutation is allowed before rollback.
        Mode 1 requires every ``observation_flag`` to be zero. Mode 2 requires
        every flag to be one and, by default, reuses same-time prediction slots
        selected by DINO/NeoForce similarity.
        """

        self._require_no_temporary_update()
        if update_cache not in (self.TEMPORARY, self.PREDICTED, self.OBSERVED):
            raise ValueError(
                "update_cache must be 0 (temporary), 1 (predicted), or 2 (observed)"
            )

        num_tokens = self._validate_kv(key, value)
        normalised = self._normalise_index(index, expected_tokens=num_tokens)
        if num_tokens > self.capacity:
            raise ValueError(
                f"cannot append {num_tokens} tokens to capacity {self.capacity}"
            )
        if update_cache == self.PREDICTED and normalised.observation_flag.any():
            raise ValueError("update_cache=1 requires observation_flag=0")
        if update_cache == self.OBSERVED and (~normalised.observation_flag).any():
            raise ValueError("update_cache=2 requires observation_flag=1")

        self._ensure_kv_storage(key, value)
        assert self._key is not None and self._value is not None

        match_slots = torch.full(
            (num_tokens,), -1, dtype=torch.long, device=self.device
        )
        if update_cache == self.OBSERVED and replace_predictions and num_tokens > 0:
            match_slots = self._match_prediction_slots(normalised)

        unmatched_rows = (match_slots < 0).nonzero(as_tuple=False).flatten()
        protected = match_slots[match_slots >= 0]
        allocated, evicted = self._allocate_slots(
            int(unmatched_rows.numel()), protected=protected
        )
        slots = match_slots.clone()
        slots[unmatched_rows] = allocated
        if len(set(slots.tolist())) != num_tokens:
            raise RuntimeError("internal cache allocation produced duplicate slots")

        snapshot = self._snapshot(slots)
        next_id_before = self._next_insertion_id
        insertion_ids = torch.arange(
            self._next_insertion_id,
            self._next_insertion_id + num_tokens,
            dtype=torch.long,
            device=self.device,
        )
        self._next_insertion_id += num_tokens

        # Advanced-index assignment is intentional.  Index data is written to
        # independent buffers and never mixed into key/value.
        self._key[:, slots] = key.detach()
        self._value[:, slots] = value.detach()
        self.mask[slots] = True
        self.insertion_ids[slots] = insertion_ids
        self.world_time_id[slots] = normalised.world_time_id
        self.dino[slots] = normalised.dino
        self.neoforce[slots] = normalised.neoforce
        self.observation_flag[slots] = normalised.observation_flag
        self.visual_valid[slots] = normalised.visual_valid
        self.tactile_valid[slots] = normalised.tactile_valid
        self._revision += 1

        result = CacheUpdate(
            slots=slots.clone(),
            replaced_slots=protected.clone(),
            evicted_slots=evicted.clone(),
            update_cache=update_cache,
            _snapshot=snapshot,
            _next_insertion_id_before=next_id_before,
            _revision_after=self._revision,
        )
        if update_cache == self.TEMPORARY:
            self._active_temporary = result
        return result

    def rollback(self, update: CacheUpdate) -> None:
        """Undo the active mode-0 append, including any temporary FIFO eviction."""

        if not update.temporary:
            raise ValueError("only update_cache=0 appends can be rolled back")
        if self._active_temporary is not update:
            raise RuntimeError("the supplied update is not the active temporary append")
        if self._revision != update._revision_after:
            raise RuntimeError(
                "cache changed after temporary append; rollback is unsafe"
            )
        if self._key is None or self._value is None:
            raise RuntimeError("cache storage disappeared before rollback")

        snapshot = update._snapshot
        slots = snapshot.slots
        self._key[:, slots] = snapshot.key
        self._value[:, slots] = snapshot.value
        self.mask[slots] = snapshot.mask
        self.insertion_ids[slots] = snapshot.insertion_ids
        self.world_time_id[slots] = snapshot.world_time_id
        self.dino[slots] = snapshot.dino
        self.neoforce[slots] = snapshot.neoforce
        self.observation_flag[slots] = snapshot.observation_flag
        self.visual_valid[slots] = snapshot.visual_valid
        self.tactile_valid[slots] = snapshot.tactile_valid
        self._next_insertion_id = update._next_insertion_id_before
        self._active_temporary = None
        self._revision += 1

    @contextmanager
    def temporary_append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        index: SemanticIndex,
    ) -> Iterator[CacheUpdate]:
        """Append with mode 0 and restore the exact prior state on context exit."""

        update = self.append(key, value, index, update_cache=self.TEMPORARY)
        try:
            yield update
        finally:
            if self._active_temporary is update:
                self.rollback(update)

    def valid_slots(self, *, order: str = "insertion") -> torch.Tensor:
        slots = self.mask.nonzero(as_tuple=False).flatten()
        if order == "slot":
            return slots
        if order != "insertion":
            raise ValueError("order must be 'insertion' or 'slot'")
        if slots.numel() == 0:
            return slots
        positions = torch.argsort(
            self.insertion_ids[slots], descending=False, stable=True
        )
        return slots[positions]

    def view(
        self,
        slots: Optional[torch.Tensor] = None,
        *,
        order: str = "insertion",
    ) -> SemanticKVView:
        """Gather K/V and their separate semantic metadata."""

        self._require_initialized()
        if slots is None:
            slots = self.valid_slots(order=order)
        else:
            slots = self._validate_slots(slots, require_valid=True)
        assert self._key is not None and self._value is not None
        return SemanticKVView(
            key=self._key[:, slots],
            value=self._value[:, slots],
            index=self._index_at(slots),
            slots=slots.clone(),
            insertion_ids=self.insertion_ids[slots].clone(),
        )

    def importance_components(
        self,
        query: SemanticIndex,
        *,
        time_scale: float = 1.0,
        slots: Optional[torch.Tensor] = None,
    ) -> ImportanceComponents:
        """Compute each index contribution without mixing the components.

        For a multi-token query, time uses the nearest query time and semantic
        similarity uses the most similar modality-compatible query token.
        """

        if not math.isfinite(float(time_scale)) or time_scale <= 0:
            raise ValueError("time_scale must be a finite positive number")
        if slots is None:
            slots = self.valid_slots(order="insertion")
        else:
            slots = self._validate_slots(slots, require_valid=True)
        query_index = self._normalise_index(query)
        if query_index.num_tokens == 0:
            raise ValueError("importance query must contain at least one token")

        cache_time = self.world_time_id[slots].to(torch.float32)
        query_time = query_index.world_time_id.to(torch.float32)
        time_distance = (cache_time[:, None] - query_time[None, :]).abs().amin(dim=1)
        time_score = torch.exp(-time_distance / float(time_scale))

        dino_score = self._max_modality_similarity(
            self.dino[slots],
            self.visual_valid[slots],
            query_index.dino,
            query_index.visual_valid,
        )
        neoforce_score = self._max_modality_similarity(
            self.neoforce[slots],
            self.tactile_valid[slots],
            query_index.neoforce,
            query_index.tactile_valid,
        )
        observation_score = self.observation_flag[slots].to(torch.float32)
        return ImportanceComponents(
            slots=slots.clone(),
            time=time_score,
            dino=dino_score,
            neoforce=neoforce_score,
            observation=observation_score,
        )

    def select_topk(
        self,
        query: SemanticIndex,
        k: int,
        *,
        weights: Union[ImportanceWeights, Mapping[str, float]] = ImportanceWeights(),
        time_scale: float = 1.0,
    ) -> SemanticKVSelection:
        """Return top-k K/V using a transparent weighted semantic score."""

        if k < 0:
            raise ValueError("k must be non-negative")
        self._require_initialized()
        components = self.importance_components(query, time_scale=time_scale)
        all_scores = components.weighted(weights)
        count = min(int(k), int(all_scores.numel()))
        if count:
            order = torch.argsort(all_scores, descending=True, stable=True)[:count]
        else:
            order = torch.empty(0, dtype=torch.long, device=self.device)
        selected_components = components.take(order)
        slots = selected_components.slots
        view = self.view(slots)
        return SemanticKVSelection(
            key=view.key,
            value=view.value,
            index=view.index,
            slots=view.slots,
            insertion_ids=view.insertion_ids,
            scores=all_scores[order],
            components=selected_components,
        )

    def _validate_kv(self, key: torch.Tensor, value: torch.Tensor) -> int:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("key and value must be tensors")
        if key.ndim < 3 or value.ndim < 3:
            raise ValueError("key and value must have shape [B, N, ...]")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("key and value must have equal batch/token dimensions")
        if key.device != self.device or value.device != self.device:
            raise ValueError(f"key and value must be on cache device {self.device}")
        if self._batch_size is not None and key.shape[0] != self._batch_size:
            raise ValueError(
                f"batch size changed from {self._batch_size} to {key.shape[0]}"
            )
        if self._key is not None:
            assert self._value is not None
            expected_key = (self._key.shape[0], *self._key.shape[2:])
            expected_value = (self._value.shape[0], *self._value.shape[2:])
            actual_key = (key.shape[0], *key.shape[2:])
            actual_value = (value.shape[0], *value.shape[2:])
            if actual_key != expected_key or actual_value != expected_value:
                raise ValueError(
                    "K/V batch size or trailing shape changed after initialization"
                )
            if key.dtype != self._key.dtype or value.dtype != self._value.dtype:
                raise ValueError("K/V dtype changed after initialization")
        return int(key.shape[1])

    def _ensure_kv_storage(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self._key is not None:
            return
        self._batch_size = int(key.shape[0])
        self._key = torch.empty(
            (key.shape[0], self.capacity, *key.shape[2:]),
            dtype=key.dtype,
            device=self.device,
        )
        self._value = torch.empty(
            (value.shape[0], self.capacity, *value.shape[2:]),
            dtype=value.dtype,
            device=self.device,
        )

    def _normalise_index(
        self,
        index: SemanticIndex,
        *,
        expected_tokens: Optional[int] = None,
    ) -> _NormalisedIndex:
        if not isinstance(index, SemanticIndex):
            raise TypeError("index must be a SemanticIndex")
        time = index.world_time_id
        observation = index.observation_flag
        if not isinstance(time, torch.Tensor) or time.ndim != 1:
            raise ValueError("world_time_id must be a tensor with shape [N]")
        if time.dtype not in _INTEGER_DTYPES or time.dtype == torch.bool:
            raise ValueError("world_time_id must use an integer dtype")
        num_tokens = int(time.shape[0])
        if expected_tokens is not None and num_tokens != expected_tokens:
            raise ValueError(
                f"index has {num_tokens} tokens, but K/V have {expected_tokens}"
            )
        if time.device != self.device:
            raise ValueError(f"index tensors must be on cache device {self.device}")

        if not isinstance(observation, torch.Tensor) or observation.shape != (
            num_tokens,
        ):
            raise ValueError("observation_flag must have shape [N]")
        if observation.device != self.device:
            raise ValueError(f"index tensors must be on cache device {self.device}")
        if observation.dtype != torch.bool:
            if observation.dtype not in _INTEGER_DTYPES:
                raise ValueError("observation_flag must be boolean or integer 0/1")
            if not torch.all((observation == 0) | (observation == 1)):
                raise ValueError("observation_flag values must be 0 or 1")
        observation = observation.to(torch.bool)

        dino, visual = self._normalise_modality(
            index.dino,
            index.visual_valid,
            num_tokens=num_tokens,
            feature_dim=self.dino_dim,
            feature_name="dino",
            mask_name="visual_valid",
        )
        neo, tactile = self._normalise_modality(
            index.neoforce,
            index.tactile_valid,
            num_tokens=num_tokens,
            feature_dim=self.neoforce_dim,
            feature_name="neoforce",
            mask_name="tactile_valid",
        )
        if torch.any(~(visual | tactile)):
            bad = (~(visual | tactile)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "every token needs DINO or NeoForce; neither is valid at rows " f"{bad}"
            )
        return _NormalisedIndex(
            world_time_id=time.to(torch.long),
            dino=dino,
            neoforce=neo,
            observation_flag=observation,
            visual_valid=visual,
            tactile_valid=tactile,
        )

    def _normalise_modality(
        self,
        feature: Optional[torch.Tensor],
        presence: Optional[torch.Tensor],
        *,
        num_tokens: int,
        feature_dim: int,
        feature_name: str,
        mask_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A globally disabled modality has a real, fixed [N, 0] representation,
        # but it can never be marked present for an individual token.
        if feature_dim == 0:
            if feature is not None:
                if not isinstance(feature, torch.Tensor) or feature.shape != (
                    num_tokens,
                    0,
                ):
                    raise ValueError(f"{feature_name} must have shape [N, 0]")
                if feature.device != self.device:
                    raise ValueError(
                        f"index tensors must be on cache device {self.device}"
                    )
            if presence is not None:
                if (
                    not isinstance(presence, torch.Tensor)
                    or presence.shape != (num_tokens,)
                    or presence.dtype != torch.bool
                ):
                    raise ValueError(f"{mask_name} must be boolean with shape [N]")
                if presence.device != self.device:
                    raise ValueError(
                        f"index tensors must be on cache device {self.device}"
                    )
                if presence.any():
                    raise ValueError(
                        f"{mask_name} cannot be true when {feature_name}_dim is zero"
                    )
            return (
                torch.empty(
                    num_tokens,
                    0,
                    dtype=self.feature_dtype,
                    device=self.device,
                ),
                torch.zeros(num_tokens, dtype=torch.bool, device=self.device),
            )

        if presence is None:
            present = torch.full(
                (num_tokens,),
                feature is not None,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            if not isinstance(presence, torch.Tensor) or presence.shape != (
                num_tokens,
            ):
                raise ValueError(f"{mask_name} must have shape [N]")
            if presence.dtype != torch.bool:
                raise ValueError(f"{mask_name} must be boolean")
            if presence.device != self.device:
                raise ValueError(f"index tensors must be on cache device {self.device}")
            present = presence

        if feature is None:
            if present.any():
                raise ValueError(f"{feature_name} is missing where {mask_name} is true")
            values = torch.zeros(
                num_tokens,
                feature_dim,
                dtype=self.feature_dtype,
                device=self.device,
            )
        else:
            if not isinstance(feature, torch.Tensor) or feature.shape != (
                num_tokens,
                feature_dim,
            ):
                raise ValueError(f"{feature_name} must have shape [N, {feature_dim}]")
            if feature.device != self.device:
                raise ValueError(f"index tensors must be on cache device {self.device}")
            if not feature.dtype.is_floating_point:
                raise ValueError(f"{feature_name} must be floating point")
            if not torch.isfinite(feature).all():
                raise ValueError(f"{feature_name} must contain only finite values")
            values = feature.to(dtype=self.feature_dtype).clone()
            values[~present] = 0
        return values, present.clone()

    def _allocate_slots(
        self, count: int, *, protected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if count == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        protected_mask = torch.zeros_like(self.mask)
        protected_mask[protected] = True
        free = (~self.mask & ~protected_mask).nonzero(as_tuple=False).flatten()
        evicted = torch.empty(0, dtype=torch.long, device=self.device)
        if free.numel() < count:
            candidates = (self.mask & ~protected_mask).nonzero(as_tuple=False).flatten()
            order = torch.argsort(
                self.insertion_ids[candidates], descending=False, stable=True
            )
            need = count - int(free.numel())
            evicted = candidates[order[:need]]
            free = torch.cat((free, evicted))
        if free.numel() < count:
            raise RuntimeError("not enough cache slots after allocation")
        return free[:count], evicted

    def _match_prediction_slots(self, observed: _NormalisedIndex) -> torch.Tensor:
        result = torch.full(
            (observed.num_tokens,), -1, dtype=torch.long, device=self.device
        )
        predicted_slots = (
            (self.mask & ~self.observation_flag).nonzero(as_tuple=False).flatten()
        )
        if observed.num_tokens == 0 or predicted_slots.numel() == 0:
            return result

        same_time = (
            observed.world_time_id[:, None]
            == self.world_time_id[predicted_slots][None, :]
        )
        dino_available = (
            observed.visual_valid[:, None] & self.visual_valid[predicted_slots][None, :]
        )
        neo_available = (
            observed.tactile_valid[:, None]
            & self.tactile_valid[predicted_slots][None, :]
        )
        dino_similarity = self._pairwise_cosine(
            observed.dino, self.dino[predicted_slots]
        )
        neo_similarity = self._pairwise_cosine(
            observed.neoforce, self.neoforce[predicted_slots]
        )
        denominator = (
            dino_available.to(torch.float32) * self.prediction_match.dino_weight
            + neo_available.to(torch.float32) * self.prediction_match.neoforce_weight
        )
        score = (
            dino_similarity
            * dino_available.to(torch.float32)
            * self.prediction_match.dino_weight
            + neo_similarity
            * neo_available.to(torch.float32)
            * self.prediction_match.neoforce_weight
        ) / denominator.clamp_min(torch.finfo(torch.float32).eps)
        eligible = same_time & (denominator > 0)
        score = score.masked_fill(~eligible, -torch.inf)

        # Global greedy one-to-one matching avoids assigning one prediction to
        # several observations.  Sorting all candidate pairs also lets NeoForce
        # disambiguate visually identical patches (and vice versa).
        flat_order = torch.argsort(score.flatten(), descending=True, stable=True)
        used_observed = torch.zeros(
            observed.num_tokens, dtype=torch.bool, device=self.device
        )
        used_predicted = torch.zeros(
            predicted_slots.numel(), dtype=torch.bool, device=self.device
        )
        pred_count = int(predicted_slots.numel())
        for flat_position in flat_order.tolist():
            obs_position = flat_position // pred_count
            pred_position = flat_position % pred_count
            pair_score = float(score[obs_position, pred_position].item())
            if pair_score < self.prediction_match.min_similarity:
                break
            if used_observed[obs_position] or used_predicted[pred_position]:
                continue
            result[obs_position] = predicted_slots[pred_position]
            used_observed[obs_position] = True
            used_predicted[pred_position] = True
        return result

    @staticmethod
    def _pairwise_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape[-1] == 0:
            return torch.zeros(
                left.shape[0], right.shape[0], dtype=torch.float32, device=left.device
            )
        left = F.normalize(left.to(torch.float32), dim=-1)
        right = F.normalize(right.to(torch.float32), dim=-1)
        return left @ right.transpose(0, 1)

    @classmethod
    def _max_modality_similarity(
        cls,
        cached_feature: torch.Tensor,
        cached_valid: torch.Tensor,
        query_feature: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros(
            cached_feature.shape[0],
            dtype=torch.float32,
            device=cached_feature.device,
        )
        cache_rows = cached_valid.nonzero(as_tuple=False).flatten()
        query_rows = query_valid.nonzero(as_tuple=False).flatten()
        if cache_rows.numel() == 0 or query_rows.numel() == 0:
            return result
        similarities = cls._pairwise_cosine(
            cached_feature[cache_rows], query_feature[query_rows]
        )
        result[cache_rows] = similarities.amax(dim=1)
        return result

    def _snapshot(self, slots: torch.Tensor) -> _SlotSnapshot:
        assert self._key is not None and self._value is not None
        return _SlotSnapshot(
            slots=slots.clone(),
            key=self._key[:, slots].clone(),
            value=self._value[:, slots].clone(),
            mask=self.mask[slots].clone(),
            insertion_ids=self.insertion_ids[slots].clone(),
            world_time_id=self.world_time_id[slots].clone(),
            dino=self.dino[slots].clone(),
            neoforce=self.neoforce[slots].clone(),
            observation_flag=self.observation_flag[slots].clone(),
            visual_valid=self.visual_valid[slots].clone(),
            tactile_valid=self.tactile_valid[slots].clone(),
        )

    def _index_at(self, slots: torch.Tensor) -> SemanticIndex:
        return SemanticIndex(
            world_time_id=self.world_time_id[slots].clone(),
            dino=self.dino[slots].clone(),
            neoforce=self.neoforce[slots].clone(),
            observation_flag=self.observation_flag[slots].clone(),
            visual_valid=self.visual_valid[slots].clone(),
            tactile_valid=self.tactile_valid[slots].clone(),
        )

    def _validate_slots(
        self, slots: torch.Tensor, *, require_valid: bool
    ) -> torch.Tensor:
        if not isinstance(slots, torch.Tensor) or slots.ndim != 1:
            raise ValueError("slots must be a one-dimensional tensor")
        if slots.dtype not in _INTEGER_DTYPES or slots.dtype == torch.bool:
            raise ValueError("slots must have integer dtype")
        if slots.device != self.device:
            raise ValueError(f"slots must be on cache device {self.device}")
        slots = slots.to(torch.long)
        if slots.numel() and ((slots < 0).any() or (slots >= self.capacity).any()):
            raise ValueError("slot is outside cache capacity")
        if require_valid and slots.numel() and not self.mask[slots].all():
            raise ValueError("all requested slots must be valid")
        return slots

    def _require_initialized(self) -> None:
        if self._key is None or self._value is None:
            raise RuntimeError("cache has not received its first K/V append")

    def _require_no_temporary_update(self) -> None:
        if self._active_temporary is not None:
            raise RuntimeError(
                "rollback the active temporary append before mutating cache"
            )


__all__ = [
    "CacheUpdate",
    "ImportanceComponents",
    "ImportanceWeights",
    "PredictionMatchConfig",
    "SemanticIndex",
    "SemanticKVCache",
    "SemanticKVSelection",
    "SemanticKVView",
]
