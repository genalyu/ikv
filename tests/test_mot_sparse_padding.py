"""Regression tests for sparse-video padding in MoT shared attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from n0_twam.models.model import custom_sdpa
from n0_twam.models.mot import SharedSelfAttention, WanMoTTransformer3DModel


def _semantic_index(
    valid: torch.Tensor, observed: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Create a complete RGB-only semantic prefix for a Q/K/V sequence."""

    if valid.dim() == 1:
        valid = valid[None]
    batch, tokens = valid.shape
    if observed is None:
        observed = torch.ones(batch, tokens, dtype=torch.long)
    elif observed.dim() == 1:
        observed = observed[None]
    return {
        "world_time_id": torch.arange(tokens)[None].expand(batch, -1),
        "dino": torch.arange(batch * tokens * 2, dtype=torch.float32).reshape(
            batch, tokens, 2
        ),
        "neoforce": torch.empty(batch, tokens, 0),
        "observation_flag": observed.expand(batch, -1).clone(),
        "visual_valid": valid.clone(),
        "tactile_valid": torch.zeros_like(valid),
        "valid_mask": valid,
    }


def _cache_only_mot_model(num_layers: int = 2) -> WanMoTTransformer3DModel:
    """Build only the module tree needed to exercise the public cache API."""

    model = WanMoTTransformer3DModel.__new__(WanMoTTransformer3DModel)
    nn.Module.__init__(model)
    model.mot = nn.Module()
    model.mot.shared_attn = nn.ModuleList(
        SharedSelfAttention() for _ in range(num_layers)
    )
    return model


def _tiny_mot_model(*, require_index: bool = False):
    return WanMoTTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=2,
        out_channels=2,
        action_dim=3,
        text_dim=8,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        rope_max_seq_len=16,
        attn_mode="torch",
        use_local_tactile=False,
        use_rgb_motion_tokens=True,
        rgb_motion_require_index=require_index,
    ).eval()


def test_streaming_cache_physically_masks_video_padding_but_keeps_tail() -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "stream",
        total_tolen=8,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )

    # Tokens 0..1 are the indexed sparse-video prefix. Token 1 is padding.
    # Tokens 2..3 are an unindexed tactile/action tail and must remain real.
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[-1.0, 1.0]]]])
    k = torch.tensor([[[[0.5, 0.0]], [[100.0, 100.0]], [[0.0, 0.5]], [[0.5, 0.5]]]])
    v = torch.tensor(
        [[[[1.0, 2.0]], [[50_000.0, -50_000.0]], [[3.0, 4.0]], [[5.0, 6.0]]]]
    )
    semantic = _semantic_index(torch.tensor([True, False]))

    actual = attention(
        q,
        k,
        v,
        update_cache=2,
        cache_name="stream",
        semantic_index=semantic,
    )

    kept = torch.tensor([0, 2, 3])
    expected = custom_sdpa(q, k[:, kept], v[:, kept])
    expected[:, 1] = 0  # padded video queries have no physical output
    torch.testing.assert_close(actual, expected)

    cache = attention.attn_caches["stream"]
    assert cache["mask"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert cache["id"].tolist() == [0, 0, 0, -1, -1, -1, -1, -1]
    assert cache["is_pred"].tolist() == [False] * 8

    # Semantic validity applies only to the video prefix; physical tail slots
    # are intentionally valid K/V without pretending to carry a visual index.
    assert cache["semantic"]["valid"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_physical_padding_mask_works_without_optional_semantic_index() -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "no-semantic",
        total_tolen=8,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    q = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[-1.0, 1.0]]]]
    )
    k = torch.tensor(
        [[[[0.5, 0.0]], [[100.0, 100.0]], [[0.0, 0.5]], [[0.5, 0.5]]]]
    )
    v = torch.tensor(
        [[[[1.0, 2.0]], [[50_000.0, -50_000.0]], [[3.0, 4.0]], [[5.0, 6.0]]]]
    )
    # First two rows are the sparse-video prefix; row 1 is padding. The two
    # unindexed tactile/action tail rows remain physical tokens.
    physical_prefix = torch.tensor([[True, False]])
    actual = attention(
        q,
        k,
        v,
        update_cache=2,
        cache_name="no-semantic",
        semantic_index=None,
        token_valid_mask=physical_prefix,
    )

    kept = torch.tensor([0, 2, 3])
    expected = custom_sdpa(q, k[:, kept], v[:, kept])
    expected[:, 1] = 0
    torch.testing.assert_close(actual, expected)
    cache = attention.attn_caches["no-semantic"]
    assert cache["mask"].sum().item() == 3
    assert cache["semantic"] is None


class _RecordingMoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_kwargs = None
        self.mask_kwargs = None

    def forward(self, hidden_states, *args, **kwargs):
        self.forward_kwargs = kwargs
        return hidden_states

    def set_masks(self, **kwargs):
        self.mask_kwargs = kwargs


def test_mot_model_forwards_physical_validity_independently_of_semantics() -> None:
    model = WanMoTTransformer3DModel.__new__(WanMoTTransformer3DModel)
    nn.Module.__init__(model)
    recorder = _RecordingMoT()
    model.mot = recorder
    hidden = torch.zeros(1, 4, 2)
    physical_prefix = torch.tensor([[True, False]])

    output = model._run_main_blocks(
        hidden,
        encoder_hidden_states=None,
        timestep_proj=torch.zeros(1, 4, 6, 2),
        temb=None,
        rotary_emb=torch.zeros(1, 4, 1, 1, dtype=torch.complex64),
        update_cache=2,
        cache_name="stream",
        action_mode=False,
        main_token_count=2,
        tactile_token_count=2,
        semantic_index=None,
        token_valid_mask=physical_prefix,
    )

    assert output is hidden
    assert recorder.forward_kwargs["semantic_index"] is None
    assert recorder.forward_kwargs["token_valid_mask"] is physical_prefix


def test_mot_inference_clears_masks_left_by_training() -> None:
    model = WanMoTTransformer3DModel.__new__(WanMoTTransformer3DModel)
    nn.Module.__init__(model)
    recorder = _RecordingMoT()
    model.mot = recorder

    model._clear_inference_attention_masks()

    assert recorder.mask_kwargs == {
        "self_block_mask": None,
        "dense_self_mask": None,
        "cross_masks": None,
    }


def test_optional_semantic_index_does_not_cache_sparse_padding_end_to_end() -> None:
    model = _tiny_mot_model(require_index=False)
    model.create_empty_cache(
        "optional-index",
        4,
        20,
        20,
        torch.device("cpu"),
        torch.float32,
        1,
    )
    indices = torch.tensor([[[0, -1]]])
    inference_input = {
        "noisy_latents": torch.randn(1, 2, 1, 4, 4),
        "timesteps": torch.zeros(1, 1),
        "grid_id": torch.zeros(1, 4, 4, dtype=torch.long),
        "text_emb": torch.randn(1, 3, 8),
        "tactile_global_latent": torch.randn(1, 1, 48, 1, 4, 4),
        "tactile_sensor_ids": torch.zeros(1, 1, dtype=torch.long),
        "motion_indices": indices,
        "motion_valid_mask": indices >= 0,
    }

    with torch.no_grad():
        model(
            inference_input,
            update_cache=2,
            cache_name="optional-index",
        )

    cache = model.mot.shared_attn[0].attn_caches["optional-index"]
    # One real RGB token plus four tactile patches; the padded RGB row has no
    # physical cache slot even though semantic metadata is optional.
    assert int(cache["mask"].sum()) == 5
    assert cache["semantic"] is None


def test_inference_forward_clears_real_mot_training_masks() -> None:
    model = _tiny_mot_model(require_index=False)
    model.use_rgb_motion_tokens = False
    stale_train_mask = object()
    with patch.object(
        model.mot,
        "forward",
        side_effect=lambda hidden_states, *args, **kwargs: hidden_states,
    ):
        model._run_backbone(
            torch.randn(1, 4, 4),
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 6, 4),
            torch.ones(1, 4, 1, 2, dtype=torch.complex64),
            stale_train_mask,
            None,
            [1, 1, 1, 1, 0, 0, 0],
            1,
            torch.randn(1, 4, 4),
        )
    attention = model.mot.shared_attn[0]
    assert attention.flex.block_mask is stale_train_mask

    inference_input = {
        "noisy_latents": torch.randn(1, 2, 1, 4, 4),
        "timesteps": torch.zeros(1, 1),
        "grid_id": torch.zeros(1, 4, 4, dtype=torch.long),
        "text_emb": torch.randn(1, 3, 8),
        "tactile_global_latent": torch.randn(1, 1, 48, 1, 4, 4),
        "tactile_sensor_ids": torch.zeros(1, 1, dtype=torch.long),
    }
    with torch.no_grad():
        output = model(inference_input, update_cache=0, cache_name="no-cache")

    assert output.shape == (1, 16, 2)
    assert attention.flex.block_mask is None
    assert attention._dense_mask is None
    assert model.mot._cross_masks == {}


def test_streaming_cache_uses_per_token_source_for_semantic_prefix() -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "mixed-source",
        total_tolen=8,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    q = k = v = torch.randn(1, 5, 1, 2)
    semantic = _semantic_index(
        torch.tensor([True, True, False]),
        observed=torch.tensor([1, 0, 0]),
    )

    # Coarse mode says this is a predicted update. The semantic prefix says
    # token 0 is actually observed and token 1 predicted; token 2 is padding.
    # The two unindexed tail tokens continue to inherit the coarse mode.
    attention(
        q,
        k,
        v,
        update_cache=1,
        cache_name="mixed-source",
        semantic_index=semantic,
    )

    cache = attention.attn_caches["mixed-source"]
    assert cache["mask"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert cache["is_pred"].tolist() == [
        False,  # observed semantic video token
        True,  # predicted semantic video token
        True,  # unindexed tail inherits update_cache=1
        True,
        False,
        False,
        False,
        False,
    ]
    assert cache["semantic"]["observation_flag"][:3].tolist() == [
        True,
        False,
        False,
    ]


def test_sparse_padding_does_not_evict_committed_cache_rows() -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "full",
        total_tolen=6,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )

    initial = torch.arange(12, dtype=torch.float32).reshape(1, 6, 1, 2)
    attention.update_cache(
        "full", initial, initial, is_pred=False,
        semantic_index=_semantic_index(torch.ones(6, dtype=torch.bool)),
    )

    # Only source row 0 is real. Rows 1..3 are sparse padding and therefore
    # must neither occupy physical slots nor force three extra FIFO evictions.
    update = torch.full((1, 4, 1, 2), 100.0)
    slots = attention.update_cache(
        "full", update, update, is_pred=False,
        semantic_index=_semantic_index(
            torch.tensor([True, False, False, False])
        ),
    )

    cache = attention.attn_caches["full"]
    assert slots.numel() == 1
    assert cache["mask"].all()
    assert (cache["id"] == 0).sum().item() == 5
    assert (cache["id"] == 1).sum().item() == 1
    assert cache["semantic"]["valid"].all()
    assert sorted(cache["semantic"]["world_time_id"].tolist()) == [0, 1, 2, 3, 4, 5]


def test_temporary_full_cache_updates_restore_complete_transaction() -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "denoise",
        total_tolen=5,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )

    # Fill the window in two commits so the oldest rows include predicted state
    # and carry distinct ids/semantic metadata.  A temporary two-token append
    # must borrow precisely those full-cache slots without evicting them for the
    # next denoising iteration.
    predicted = torch.arange(4, dtype=torch.float32).reshape(1, 2, 1, 2)
    predicted_index = _semantic_index(
        torch.ones(2, dtype=torch.bool),
        observed=torch.zeros(2, dtype=torch.long),
    )
    attention(
        predicted,
        predicted,
        predicted + 20,
        update_cache=1,
        cache_name="denoise",
        semantic_index=predicted_index,
    )

    observed = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2) + 100
    observed_index = _semantic_index(torch.ones(3, dtype=torch.bool))
    observed_index["world_time_id"] += 10
    observed_index["dino"] += 50
    attention(
        observed,
        observed,
        observed + 20,
        update_cache=2,
        cache_name="denoise",
        semantic_index=observed_index,
    )

    cache = attention.attn_caches["denoise"]
    assert cache["mask"].all()
    assert cache["id"].tolist() == [0, 0, 1, 1, 1]
    assert cache["is_pred"].tolist() == [True, True, False, False, False]

    snapshot = {
        name: cache[name].clone() for name in ("k", "v", "id", "mask", "is_pred")
    }
    semantic_object = cache["semantic"]
    semantic_snapshot = {
        name: value.clone() for name, value in semantic_object.items()
    }

    # Simulate repeated non-final denoising steps.  Values and semantic metadata
    # deliberately change on every pass so a partial mask-only rollback cannot
    # accidentally satisfy the equality checks.
    for step in range(3):
        temporary = torch.full((1, 2, 1, 2), 1_000.0 + step)
        temporary_index = _semantic_index(
            torch.ones(2, dtype=torch.bool),
            observed=torch.zeros(2, dtype=torch.long),
        )
        temporary_index["world_time_id"] += 100 + 10 * step
        temporary_index["dino"] += 1_000 + 100 * step

        output = attention(
            temporary,
            temporary + 10,
            temporary + 20,
            update_cache=0,
            cache_name="denoise",
            semantic_index=temporary_index,
        )
        assert output.shape == temporary.shape

        for name, expected in snapshot.items():
            torch.testing.assert_close(cache[name], expected, rtol=0, atol=0)
        assert cache["semantic"] is semantic_object
        for name, expected in semantic_snapshot.items():
            torch.testing.assert_close(
                cache["semantic"][name], expected, rtol=0, atol=0
            )


@pytest.mark.parametrize(
    ("capacity", "initial_tokens", "update_mode", "bad_observed"),
    [
        (4, 4, 1, False),  # full cache: failed prediction would evict history
        (5, 2, 2, True),  # non-full cache: failed observation would dirty free slots
    ],
)
def test_failed_committed_update_is_atomic_for_full_and_nonfull_cache(
    capacity: int,
    initial_tokens: int,
    update_mode: int,
    bad_observed: bool,
) -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "atomic",
        total_tolen=capacity,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    cache = attention.attn_caches["atomic"]
    # Make even currently-free K/V rows deterministic: atomicity covers the
    # complete backing pool, not only rows selected by the validity mask.
    cache["k"].copy_(
        torch.arange(capacity * 2, dtype=torch.float32).reshape(1, capacity, 1, 2)
        - 100
    )
    cache["v"].copy_(cache["k"] - 1_000)

    initial = torch.arange(initial_tokens * 2, dtype=torch.float32).reshape(
        1, initial_tokens, 1, 2
    )
    attention(
        initial,
        initial + 10,
        initial + 20,
        update_cache=2,
        cache_name="atomic",
        semantic_index=_semantic_index(
            torch.ones(initial_tokens, dtype=torch.bool)
        ),
    )
    assert cache["semantic"]["dino"].shape[-1] == 2

    snapshot = {
        name: cache[name].clone() for name in ("k", "v", "id", "mask", "is_pred")
    }
    semantic_object = cache["semantic"]
    semantic_snapshot = {
        name: value.clone() for name, value in semantic_object.items()
    }

    bad = torch.full((1, 2, 1, 2), 9_999.0)
    bad_index = _semantic_index(
        torch.ones(2, dtype=torch.bool),
        observed=torch.full((2,), int(bad_observed), dtype=torch.long),
    )
    # Existing sidecar width is 2. The incompatible width is diagnosed only at
    # the semantic write boundary, after slot planning, so this exercises actual
    # rollback for both eviction and free-slot paths.
    bad_index["dino"] = torch.randn(1, 2, 3)

    with pytest.raises(ValueError, match="feature dimensions changed"):
        attention(
            bad,
            bad + 10,
            bad + 20,
            update_cache=update_mode,
            cache_name="atomic",
            semantic_index=bad_index,
        )

    for name, expected in snapshot.items():
        torch.testing.assert_close(cache[name], expected, rtol=0, atol=0)
    assert cache["semantic"] is semantic_object
    for name, expected in semantic_snapshot.items():
        torch.testing.assert_close(cache["semantic"][name], expected, rtol=0, atol=0)


def test_committed_cache_rolls_back_when_attention_backend_fails() -> None:
    attention = SharedSelfAttention()
    attention.init_kv_cache(
        "backend-failure",
        total_tolen=4,
        num_head=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    cache = attention.attn_caches["backend-failure"]
    cache["k"].copy_(torch.arange(8, dtype=torch.float32).reshape(1, 4, 1, 2))
    cache["v"].copy_(cache["k"] + 100)
    before = {
        name: cache[name].clone()
        for name in ("k", "v", "id", "mask", "is_pred")
    }
    assert cache["semantic"] is None

    qkv = torch.randn(1, 1, 1, 2)
    with patch(
        "n0_twam.models.mot.custom_sdpa",
        side_effect=RuntimeError("attention backend failed"),
    ):
        with pytest.raises(RuntimeError, match="attention backend failed"):
            attention(
                qkv,
                qkv,
                qkv,
                update_cache=2,
                cache_name="backend-failure",
                semantic_index=_semantic_index(torch.ones(1, dtype=torch.bool)),
            )

    for name, expected in before.items():
        torch.testing.assert_close(cache[name], expected, rtol=0, atol=0)
    # The failed write lazily allocated a semantic sidecar; rollback removes it.
    assert cache["semantic"] is None


def test_failed_grounding_append_preserves_existing_prediction() -> None:
    model = WanMoTTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=2,
        out_channels=2,
        action_dim=3,
        text_dim=8,
        freq_dim=4,
        ffn_dim=8,
        num_layers=1,
        rope_max_seq_len=16,
        attn_mode="torch",
        use_local_tactile=False,
        use_rgb_motion_tokens=True,
        rgb_motion_require_index=True,
    ).eval()
    model.create_empty_cache(
        "clear-rollback",
        attn_window=2,
        latent_token_per_chunk=8,
        action_token_per_chunk=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )
    attention = model.mot.shared_attn[0]
    predicted = torch.arange(12, dtype=torch.float32).reshape(1, 3, 1, 4)
    attention(
        predicted,
        predicted + 10,
        predicted + 20,
        update_cache=1,
        cache_name="clear-rollback",
        semantic_index=_semantic_index(
            torch.ones(3, dtype=torch.bool),
            observed=torch.zeros(3, dtype=torch.long),
        ),
    )
    cache = attention.attn_caches["clear-rollback"]
    before = {
        name: cache[name].clone()
        for name in ("k", "v", "id", "mask", "is_pred")
    }
    semantic_object = cache["semantic"]
    semantic_before = {
        name: value.clone() for name, value in semantic_object.items()
    }

    with pytest.raises(RuntimeError, match="later grounding failure"):
        with model.cache_transaction("clear-rollback"):
            attention(
                predicted, predicted + 30, predicted + 40,
                update_cache=2, cache_name="clear-rollback",
                semantic_index=_semantic_index(torch.ones(3, dtype=torch.bool)),
                cache_transaction=model.mot.active_cache_transaction_entries("clear-rollback"),
            )
            assert cache["mask"].sum() == 6
            assert (cache["mask"] & cache["is_pred"]).sum() == 3
            raise RuntimeError("later grounding failure")

    for name, expected in before.items():
        torch.testing.assert_close(cache[name], expected, rtol=0, atol=0)
    assert cache["semantic"] is semantic_object
    for name, expected in semantic_before.items():
        torch.testing.assert_close(cache["semantic"][name], expected, rtol=0, atol=0)


def test_committed_model_forward_rolls_back_all_layers_on_later_failure() -> None:
    model = WanMoTTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=4,
        in_channels=2,
        out_channels=2,
        action_dim=3,
        text_dim=8,
        freq_dim=4,
        ffn_dim=8,
        num_layers=2,
        rope_max_seq_len=16,
        attn_mode="torch",
        use_local_tactile=False,
        use_rgb_motion_tokens=True,
        rgb_motion_require_index=True,
    ).eval()
    model.create_empty_cache(
        "layer-failure",
        attn_window=4,
        latent_token_per_chunk=20,
        action_token_per_chunk=20,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )

    indices = torch.tensor([[[0, -1]]])
    valid = indices >= 0
    input_dict = {
        "noisy_latents": torch.randn(1, 2, 1, 4, 4),
        "timesteps": torch.zeros(1, 1),
        "grid_id": torch.zeros(1, 4, 4, dtype=torch.long),
        "text_emb": torch.randn(1, 3, 8),
        "tactile_global_latent": torch.randn(1, 1, 48, 1, 4, 4),
        "tactile_sensor_ids": torch.zeros(1, 1, dtype=torch.long),
        "motion_indices": indices,
        "motion_valid_mask": valid,
        "world_time_id": torch.tensor([[[3, -1]]]),
        "dino_features": torch.randn(1, 1, 2, 5),
        "neoforce_features": torch.empty(1, 1, 2, 0),
        "observation_flag": torch.tensor([[[1, 0]]]),
        "visual_valid": valid,
        "tactile_valid": torch.zeros_like(valid),
    }
    caches = [
        attention.attn_caches["layer-failure"]
        for attention in model.mot.shared_attn
    ]
    for cache in caches:
        cache["k"].zero_()
        cache["v"].zero_()
    before = [
        {
            name: cache[name].clone()
            for name in ("k", "v", "id", "mask", "is_pred")
        }
        for cache in caches
    ]

    with patch.object(
        model.mot.shared_attn[1],
        "forward",
        side_effect=RuntimeError("forced layer-1 failure"),
    ):
        with pytest.raises(RuntimeError, match="forced layer-1 failure"):
            with torch.no_grad():
                model(input_dict, update_cache=2, cache_name="layer-failure")

    assert [int(cache["mask"].sum()) for cache in caches] == [0, 0]
    for cache, snapshot in zip(caches, before):
        for name, expected in snapshot.items():
            torch.testing.assert_close(cache[name], expected, rtol=0, atol=0)
        assert cache["semantic"] is None


def test_current_sdpa_combines_existing_mask_with_semantic_padding_mask() -> None:
    attention = SharedSelfAttention()
    # Preserve an existing causal constraint while adding semantic validity.
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    attention.set_dense_mask(causal)

    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[-1.0, 1.0]]]])
    k = torch.tensor([[[[0.5, 0.0]], [[100.0, 100.0]], [[0.0, 0.5]], [[0.5, 0.5]]]])
    v = torch.tensor(
        [[[[1.0, 2.0]], [[50_000.0, -50_000.0]], [[3.0, 4.0]], [[5.0, 6.0]]]]
    )
    semantic = _semantic_index(torch.tensor([True, False]))

    actual = attention(q, k, v, semantic_index=semantic)

    physical_valid = torch.tensor([[True, False, True, True]])
    allowed = (
        causal[None, None]
        & physical_valid[:, None, :, None]
        & physical_valid[:, None, None, :]
    )
    expected = custom_sdpa(q, k, v, attn_mask=allowed)
    expected[:, 1] = 0
    torch.testing.assert_close(actual, expected)

    # Both tail queries remain live, and the enormous padded value cannot leak.
    assert actual[:, 2:].abs().sum() > 0
    assert actual[:, 2:].abs().max() < 100.0


def test_model_semantic_cache_accessor_is_layered_and_read_only() -> None:
    model = _cache_only_mot_model()
    for attention in model.mot.shared_attn:
        attention.init_kv_cache(
            "stream",
            total_tolen=5,
            num_head=1,
            head_dim=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            batch_size=1,
        )

    qkv = torch.randn(1, 3, 1, 2)
    first = _semantic_index(torch.tensor([True, False]))
    second = _semantic_index(torch.tensor([True, True]))
    second["world_time_id"] += 10
    second["dino"] += 100
    model.mot.shared_attn[0](
        qkv, qkv, qkv, update_cache=2, cache_name="stream", semantic_index=first
    )
    model.mot.shared_attn[1](
        qkv, qkv, qkv, update_cache=2, cache_name="stream", semantic_index=second
    )

    live_cache = model.mot.shared_attn[1].attn_caches["stream"]
    key_before = live_cache["k"].clone()
    value_before = live_cache["v"].clone()
    snapshot = model.get_semantic_cache("stream", layer=1)

    assert snapshot is not None
    assert set(snapshot) == {
        "slot_indices",
        "valid",
        "world_time_id",
        "dino",
        "neoforce",
        "observation_flag",
        "visual_valid",
        "tactile_valid",
    }
    assert snapshot["slot_indices"].tolist() == [0, 1]
    assert snapshot["valid"].tolist() == [True, True]
    assert snapshot["world_time_id"].tolist() == [10, 11]
    assert snapshot["dino"].shape == (2, 2)
    assert snapshot["neoforce"].shape == (2, 0)

    # This is a clone-based snapshot, not an alias into semantic K/V storage.
    snapshot["world_time_id"][0] = 999
    snapshot["dino"].zero_()
    snapshot["valid"].zero_()
    fresh = model.get_semantic_cache("stream", layer=1)
    assert fresh is not None
    assert fresh["world_time_id"].tolist() == [10, 11]
    assert fresh["dino"].abs().sum() > 0
    assert fresh["valid"].all()
    torch.testing.assert_close(live_cache["k"], key_before)
    torch.testing.assert_close(live_cache["v"], value_before)

    # Layer 0 has its own independent semantic rows and full-capacity inspection
    # retains physical slot positions without exposing K/V tensors.
    layer_zero = model.get_semantic_cache("stream", layer=0, valid_only=False)
    assert layer_zero is not None
    assert layer_zero["slot_indices"].tolist() == [0, 1, 2, 3, 4]
    assert layer_zero["valid"].tolist() == [True, False, False, False, False]
    assert layer_zero["world_time_id"][0].item() == 0


def test_model_semantic_cache_accessor_validates_layer_and_missing_sidecar() -> None:
    model = _cache_only_mot_model(num_layers=1)
    assert model.get_semantic_cache("missing", layer=0) is None
    with pytest.raises(IndexError, match="outside"):
        model.get_semantic_cache("missing", layer=1)
    with pytest.raises(TypeError, match="integer"):
        model.get_semantic_cache("missing", layer=True)
    with pytest.raises(TypeError, match="valid_only"):
        model.get_semantic_cache("missing", layer=0, valid_only=1)
