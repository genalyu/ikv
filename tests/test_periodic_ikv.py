"""Periodic management must preserve FIFO, archival and transaction semantics."""
from unittest.mock import patch
import pytest
import torch
from test_cpu_kv_archive import model, policy, grounded
from test_global_kv_retention import rows, append
from n0_twam.models.global_kv_retention import RetentionConfig

@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_interval(value):
    with pytest.raises(ValueError, match="positive integer"):
        RetentionConfig(ikv_interval_frames=value)

@torch.no_grad()
def test_archive_management_frequency_and_no_duplicate_action_count():
    m = model(ikv_interval_frames=3)
    p = policy(m)
    with patch.object(p.archive, "select", wraps=p.archive.select) as select:
        for t in range(7):
            grounded(m, t)
            grounded(m, t, action=True)
        assert select.call_count == 3  # frames 0, 3, 6
    assert p.archive.stats["archived_tokens"] > 7 * 8
    assert p._frame_clock == 6

@torch.no_grad()
def test_intermediate_fifo_skips_scores_and_preserves_all_cpu_history():
    m = model(ikv_interval_frames=4)
    p = policy(m)
    grounded(m, 0)
    with patch.object(p, "scores", side_effect=AssertionError("intermediate scoring")):
        for t in (1, 2, 3):
            grounded(m, t)
    active = m.get_global_retention("test")
    assert active["world_time_id"][active["kind"] == 0].min() == 2
    assert p.archive.stats["archived_tokens"] == 32
    assert any((r["metadata"]["world_time_id"] == 0).any() for r in p.archive.records)

@torch.no_grad()
def test_failed_management_retries_and_counter_rolls_back():
    m = model(ikv_interval_frames=2)
    p = policy(m)
    grounded(m, 0); grounded(m, 1)
    before = p.snapshot()[3]
    count = p.archive.stats["archived_tokens"]
    with pytest.raises(RuntimeError, match="abort"):
        with m.cache_transaction("test"):
            grounded(m, 2)
            raise RuntimeError("abort")
    assert p.snapshot()[3] == before
    assert p.archive.stats["archived_tokens"] == count
    with patch.object(p.archive, "select", wraps=p.archive.select) as select:
        grounded(m, 2)
        assert select.call_count == 1

@torch.no_grad()
def test_chunk_threshold_and_duplicate_frames():
    p = policy(model(ikv_interval_frames=3))
    p._commit_cadence(rows([0, 0]))
    assert not p.management_due(rows([0, 0]))
    p._commit_cadence(rows([1, 1, 2, 2]))
    assert not p.management_due(rows([2]))
    assert p.management_due(rows([3, 4]))
    p._commit_cadence(rows([3, 4]))
    assert p._frames_since_management == 0
    assert not p.management_due(rows([5]))

@torch.no_grad()
def test_evict_mode_intermediate_fifo_and_full_score_event():
    p = policy(model(history_storage="evict", ikv_interval_frames=3))
    mask = torch.zeros(p.capacity, dtype=torch.bool)
    first, _ = append(p, mask, rows([0] * 8))
    with patch.object(p, "scores", side_effect=AssertionError("intermediate scoring")):
        _, victims = append(p, mask, rows([1] * 2))
        assert victims.tolist() == first[:2].tolist()
        append(p, mask, rows([2] * 2))
    with patch.object(p, "scores", wraps=p.scores) as score:
        append(p, mask, rows([3] * 2))
        assert score.call_count == 1
