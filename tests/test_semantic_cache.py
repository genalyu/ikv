import math

import pytest
import torch

from n0_twam.models.semantic_cache import (
    ImportanceWeights,
    PredictionMatchConfig,
    SemanticIndex,
    SemanticKVCache,
)


def _kv(values, *, width=2):
    values = torch.as_tensor(values, dtype=torch.float32)
    key = values.view(1, -1, 1).expand(-1, -1, width).clone()
    value = (values + 100).view(1, -1, 1).expand(-1, -1, width).clone()
    return key, value


def _index(
    times,
    *,
    dino,
    neo,
    observed,
    visual_valid=None,
    tactile_valid=None,
):
    return SemanticIndex(
        world_time_id=torch.tensor(times, dtype=torch.long),
        dino=None if dino is None else torch.tensor(dino, dtype=torch.float32),
        neoforce=None if neo is None else torch.tensor(neo, dtype=torch.float32),
        observation_flag=torch.tensor(observed, dtype=torch.bool),
        visual_valid=(
            None
            if visual_valid is None
            else torch.tensor(visual_valid, dtype=torch.bool)
        ),
        tactile_valid=(
            None
            if tactile_valid is None
            else torch.tensor(tactile_valid, dtype=torch.bool)
        ),
    )


def test_kv_content_and_semantic_index_are_stored_separately():
    cache = SemanticKVCache(capacity=4, dino_dim=3, neoforce_dim=2)
    key, value = _kv([1, 2])
    index = _index(
        [4, 4],
        dino=[[1, 0, 0], [0, 1, 0]],
        neo=[[9, 9], [8, 8]],
        observed=[0, 0],
        visual_valid=[1, 0],
        tactile_valid=[0, 1],
    )

    cache.append(key, value, index, update_cache=SemanticKVCache.PREDICTED)
    view = cache.view()

    # Index dimensions never enlarge or alter the original attention K/V.
    assert view.key.shape == (1, 2, 2)
    assert view.value.shape == (1, 2, 2)
    torch.testing.assert_close(view.key, key)
    torch.testing.assert_close(view.value, value)
    assert view.index.dino.shape == (2, 3)
    assert view.index.neoforce.shape == (2, 2)
    assert view.index.visual_valid.tolist() == [True, False]
    assert view.index.tactile_valid.tolist() == [False, True]
    # Invalid feature rows are canonicalized to zero, not mistaken for real data.
    torch.testing.assert_close(view.index.dino[1], torch.zeros(3))
    torch.testing.assert_close(view.index.neoforce[0], torch.zeros(2))


def test_presence_validation_requires_at_least_one_modality():
    cache = SemanticKVCache(capacity=2, dino_dim=2, neoforce_dim=2)
    key, value = _kv([1])
    bad = _index(
        [0],
        dino=[[1, 0]],
        neo=None,
        observed=[0],
        visual_valid=[0],
        tactile_valid=[0],
    )
    with pytest.raises(ValueError, match="needs DINO or NeoForce"):
        cache.append(key, value, bad, update_cache=1)

    wrong_mode = _index([0], dino=[[1, 0]], neo=None, observed=[1], visual_valid=[1])
    with pytest.raises(ValueError, match="update_cache=1"):
        cache.append(key, value, wrong_mode, update_cache=1)


def test_rgb_only_cache_supports_globally_absent_neoforce():
    cache = SemanticKVCache(capacity=2, dino_dim=2, neoforce_dim=0)
    key, value = _kv([1, 2])
    rgb_only = _index([0, 1], dino=[[1, 0], [0, 1]], neo=None, observed=[0, 0])
    cache.append(key, value, rgb_only, update_cache=1)

    view = cache.view()
    assert view.index.neoforce.shape == (2, 0)
    assert view.index.tactile_valid.tolist() == [False, False]
    query = _index([1], dino=[[0, 1]], neo=None, observed=[1])
    components = cache.importance_components(query)
    torch.testing.assert_close(components.neoforce, torch.zeros(2))
    assert (
        cache.select_topk(
            query,
            1,
            weights={"time": 0, "dino": 1, "neoforce": 0, "observation": 0},
        ).index.world_time_id.item()
        == 1
    )

    with pytest.raises(ValueError, match="cannot both be zero"):
        SemanticKVCache(capacity=1, dino_dim=0, neoforce_dim=0)


def test_temporary_append_restores_evicted_content_and_metadata():
    cache = SemanticKVCache(capacity=2, dino_dim=2, neoforce_dim=2)
    key, value = _kv([1, 2])
    committed = _index(
        [1, 2],
        dino=[[1, 0], [0, 1]],
        neo=None,
        observed=[0, 0],
        visual_valid=[1, 1],
    )
    cache.append(key, value, committed, update_cache=1)
    before = cache.view()

    temp_key, temp_value = _kv([99])
    temporary = _index([3], dino=None, neo=[[1, 1]], observed=[0], tactile_valid=[1])
    with cache.temporary_append(temp_key, temp_value, temporary) as update:
        assert update.temporary
        assert update.evicted_slots.numel() == 1
        assert 99 in cache.view().key[0, :, 0].tolist()
        with pytest.raises(RuntimeError, match="rollback"):
            cache.clear_predictions()

    after = cache.view()
    torch.testing.assert_close(after.key, before.key)
    torch.testing.assert_close(after.value, before.value)
    torch.testing.assert_close(after.index.world_time_id, before.index.world_time_id)
    torch.testing.assert_close(after.index.dino, before.index.dino)
    torch.testing.assert_close(after.insertion_ids, before.insertion_ids)


def test_fixed_capacity_uses_fifo_eviction_for_committed_appends():
    cache = SemanticKVCache(capacity=2, dino_dim=2, neoforce_dim=2)
    key, value = _kv([1, 2])
    index = _index([1, 2], dino=[[1, 0], [0, 1]], neo=None, observed=[0, 0])
    cache.append(key, value, index, update_cache=1)

    key3, value3 = _kv([3])
    index3 = _index([3], dino=[[1, 1]], neo=None, observed=[0])
    result = cache.append(key3, value3, index3, update_cache=1)

    assert result.evicted_slots.tolist() == [0]
    assert cache.view().key[0, :, 0].tolist() == [2.0, 3.0]
    assert cache.view().index.world_time_id.tolist() == [2, 3]


def test_observation_replaces_same_time_prediction_using_both_modalities():
    cache = SemanticKVCache(
        capacity=4,
        dino_dim=2,
        neoforce_dim=2,
        prediction_match=PredictionMatchConfig(min_similarity=0.8),
    )
    key, value = _kv([10, 20, 30])
    predicted = _index(
        [5, 5, 6],
        dino=[[1, 0], [1, 0], [0, 1]],
        neo=[[1, 0], [0, 1], [1, 0]],
        observed=[0, 0, 0],
    )
    initial = cache.append(key, value, predicted, update_cache=1)

    observed_key, observed_value = _kv([200])
    observed = _index([5], dino=[[1, 0]], neo=[[0, 1]], observed=[1])
    result = cache.append(observed_key, observed_value, observed, update_cache=2)

    # DINO ties; NeoForce selects the second prediction. It is replaced in place.
    assert result.replaced_slots.tolist() == [initial.slots[1].item()]
    assert result.evicted_slots.numel() == 0
    assert len(cache) == 3
    replaced_slot = result.replaced_slots[0]
    assert cache.key_buffer[0, replaced_slot, 0].item() == 200
    assert cache.observation_flag[replaced_slot].item() is True

    # Semantic similarity alone is insufficient: world time must also match.
    later_key, later_value = _kv([700])
    later = _index([7], dino=[[1, 0]], neo=[[0, 1]], observed=[1])
    later_result = cache.append(later_key, later_value, later, update_cache=2)
    assert later_result.replaced_slots.numel() == 0
    assert len(cache) == 4


def test_importance_components_are_independent_and_drive_weighted_topk():
    cache = SemanticKVCache(capacity=4, dino_dim=2, neoforce_dim=2)
    key, value = _kv([1, 2, 3, 4])
    # Mixed observed/predicted rows are legal for temporary/query-independent
    # setup; commit them in their two protocol-valid groups.
    first_key, first_value = key[:, [0, 2]], value[:, [0, 2]]
    first = _index(
        [7, 10],
        dino=[[0, 1], [-1, 0]],
        neo=[[1, 0], [0, 1]],
        observed=[1, 1],
        visual_valid=[1, 0],
        tactile_valid=[0, 1],
    )
    cache.append(first_key, first_value, first, update_cache=2)
    second_key, second_value = key[:, [1, 3]], value[:, [1, 3]]
    second = _index(
        [9, 10],
        dino=[[1, 0], [0.6, 0.8]],
        neo=[[0, 1], [1, 0]],
        observed=[0, 0],
        visual_valid=[1, 1],
        tactile_valid=[1, 0],
    )
    cache.append(second_key, second_value, second, update_cache=1)

    query = _index([10], dino=[[1, 0]], neo=[[0, 1]], observed=[1])
    components = cache.importance_components(query, time_scale=1.0)

    # Cache insertion order is observed rows first, then predicted rows.
    torch.testing.assert_close(
        components.time,
        torch.tensor([math.exp(-3), 1.0, math.exp(-1), 1.0]),
    )
    torch.testing.assert_close(components.dino, torch.tensor([0.0, 0.0, 1.0, 0.6]))
    torch.testing.assert_close(components.neoforce, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(
        components.observation, torch.tensor([1.0, 1.0, 0.0, 0.0])
    )

    dino_only = cache.select_topk(
        query,
        2,
        weights=ImportanceWeights(time=0, dino=1, neoforce=0, observation=0),
    )
    assert dino_only.index.world_time_id.tolist() == [9, 10]
    torch.testing.assert_close(dino_only.scores, torch.tensor([1.0, 0.6]))
    assert dino_only.components.as_dict().keys() == {
        "time",
        "dino",
        "neoforce",
        "observation",
    }

    observed_only = cache.select_topk(
        query,
        2,
        weights={"time": 0, "dino": 0, "neoforce": 0, "observation": 1},
    )
    assert observed_only.index.observation_flag.tolist() == [True, True]
