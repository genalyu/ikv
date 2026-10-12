"""Archive integration tests with the real two-layer MoT and tiny weights."""
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from test_global_kv_retention import tiny_model, model_input, cache_snapshot, assert_cache_equal


def model(**kwargs):
    result = tiny_model(False)
    config = dict(version=2, video_capacity=8, action_capacity=4, tactile_capacity=12,
                  query_weight=0, action_query_weight=0, tactile_query_weight=0,
                  visual_weight=0, persistence_weight=0, contact_weight=0,
                  history_storage="cpu_archive", archive_max_bytes=1024**2)
    config.update(kwargs)
    result.configure_global_retention("test", **config)
    return result


def policy(m):
    return m.mot.retention_policies["test"]


def grounded(m, time, action=False):
    return m(model_input(time, action), update_cache=2, cache_name="test", action_mode=action)


@torch.no_grad()
def test_current_observation_is_reserved_and_all_history_is_archived():
    m = model()
    for t in range(6):
        grounded(m, t)
    p = policy(m)
    assert p.archive.stats["archived_tokens"] == 48
    active = m.get_global_retention("test")
    assert (active["world_time_id"] == 5).sum() == 8
    assert (active["kind"] == 0).sum() == 8
    assert (active["kind"] == 2).sum() == 12
    assert len(torch.unique(torch.cat([r["metadata"]["token_uid"] for r in p.archive.records]))) == 48


@torch.no_grad()
def test_predictions_do_not_evict_or_archive_history_and_clear_rolls_back():
    m = model()
    grounded(m, 0)
    before = cache_snapshot(m)
    p = policy(m)
    uid, count = p.next_uid, p.archive.stats["archived_tokens"]
    m(model_input(1), update_cache=1, cache_name="test")
    m(model_input(1, True), update_cache=1, cache_name="test", action_mode=True)
    assert_cache_equal(before, cache_snapshot(m))
    assert p.next_uid == uid and p.archive.stats["archived_tokens"] == count
    assert len(m.mot.shared_attn[0].attn_caches["test"]["predicted_kv"]) == 2
    with pytest.raises(RuntimeError):
        with m.cache_transaction("test"):
            m.clear_pred_cache("test")
            raise RuntimeError("rollback clear")
    assert len(m.mot.shared_attn[0].attn_caches["test"]["predicted_kv"]) == 2
    m.clear_pred_cache("test")
    assert not m.mot.shared_attn[0].attn_caches["test"]["predicted_kv"]
    assert_cache_equal(before, cache_snapshot(m))


@torch.no_grad()
def test_old_cpu_kv_can_be_recalled_without_recomputation():
    m = model()
    for t in range(4): grounded(m, t)
    p = policy(m)
    record = p.archive.records[0]
    old_uid = record["metadata"]["token_uid"][0].item()
    assert old_uid not in m.get_global_retention("test")["token_uid"].tolist()
    # Change the ranking function, without touching archived tensors.
    original = p.scores
    def prefer_old(slots, **kwargs):
        score = torch.zeros(len(slots), device=p.data["kind"].device)
        return score + (p.data["token_uid"][slots] == old_uid) * 100
    # The CPU scorer is a shallow copy, so a class patch binds to its CPU data.
    def ranked(self, slots, **kwargs):
        return (self.data["token_uid"][slots] == old_uid).float() * 100
    with patch.object(type(p), "scores", ranked): grounded(m, 5)
    active = m.get_global_retention("test")
    slot = active["slot_indices"][active["token_uid"] == old_uid]
    assert len(slot) == 1
    for layer, attention in enumerate(m.mot.shared_attn):
        torch.testing.assert_close(attention.attn_caches["test"]["k"][:, slot], record["kv"][layer][0][:, :1])
        torch.testing.assert_close(attention.attn_caches["test"]["v"][:, slot], record["kv"][layer][1][:, :1])
    assert p.archive.stats["recalled_tokens"] >= 1


@torch.no_grad()
def test_failure_restores_archive_gpu_and_uid():
    m = model()
    for t in range(3): grounded(m, t)
    before, p = cache_snapshot(m), policy(m)
    stats, uid = dict(p.archive.stats), p.next_uid
    with patch.object(m.mot.experts["video"], "embed_out", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError): grounded(m, 4)
    assert_cache_equal(before, cache_snapshot(m))
    assert p.archive.stats == stats and p.next_uid == uid


@torch.no_grad()
def test_limit_error_rolls_back_without_deleting_history():
    m = model(archive_max_bytes=1)
    before = cache_snapshot(m)
    with pytest.raises(MemoryError, match="limit exceeded"): grounded(m, 0)
    assert_cache_equal(before, cache_snapshot(m))
    assert policy(m).archive.stats["archived_tokens"] == 0
    assert policy(m).next_uid == 0


@torch.no_grad()
def test_disk_spill_recall_and_episode_cleanup(tmp_path):
    m = model(archive_max_bytes=1, archive_overflow="disk", archive_directory=str(tmp_path))
    for t in range(4): grounded(m, t)
    p = policy(m)
    assert p.archive.stats["resident_bytes"] <= 1
    assert p.archive.stats["spilled_bytes"] > 0
    old_uid = p.archive.records[0]["metadata"]["token_uid"][0].item()
    def ranked(self, slots, **kwargs):
        return (self.data["token_uid"][slots] == old_uid).float() * 100
    with patch.object(type(p), "scores", ranked): grounded(m, 5)
    assert old_uid in m.get_global_retention("test")["token_uid"].tolist()
    assert list(tmp_path.rglob("*.pt"))
    m.clear_cache("test")
    assert not list(tmp_path.rglob("*.pt"))


@torch.no_grad()
def test_request_transaction_rolls_back_committed_archive_append(tmp_path):
    m = model(archive_max_bytes=1, archive_overflow="disk", archive_directory=str(tmp_path))
    grounded(m, 0)
    before, p = cache_snapshot(m), policy(m)
    files, stats = set(p.archive.files), dict(p.archive.stats)
    with pytest.raises(RuntimeError):
        with m.cache_transaction("test"):
            grounded(m, 1)
            raise RuntimeError("late request failure")
    assert_cache_equal(before, cache_snapshot(m))
    assert p.archive.files == files and p.archive.stats == stats
    assert set(str(x) for x in tmp_path.rglob("*.pt")) == files


@torch.no_grad()
def test_archive_rejects_fake_grounding_without_breaking_outer_transaction():
    m = model()
    grounded(m, 0)
    before = cache_snapshot(m)
    data = model_input(1)
    data["kv_index"] = {"observation_flag": 0}
    with m.cache_transaction("test"):
        with pytest.raises(ValueError, match="real observation"):
            m(data, update_cache=2, cache_name="test")
        assert m.mot.has_active_cache_transaction("test")
    assert_cache_equal(before, cache_snapshot(m))


@torch.no_grad()
def test_caught_forward_failure_restores_recall_inside_outer_transaction():
    m = model()
    for t in range(4): grounded(m, t)
    before, p = cache_snapshot(m), policy(m)
    stats = dict(p.archive.stats)
    uid = p.archive.records[0]["metadata"]["token_uid"][0].item()
    def ranked(self, slots, **kwargs):
        return (self.data["token_uid"][slots] == uid).float() * 100
    with m.cache_transaction("test"):
        with patch.object(type(p), "scores", ranked), patch.object(
                m.mot.experts["video"], "embed_out", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError): grounded(m, 5)
        assert_cache_equal(before, cache_snapshot(m))
        assert p.archive.stats == stats
    assert_cache_equal(before, cache_snapshot(m))
