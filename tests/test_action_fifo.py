"""Action cache must follow baseline FIFO, independently of IKV sensory history."""
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from test_cpu_kv_archive import model, policy, grounded
from test_global_kv_retention import tiny_model, model_input, rows, append, cache_snapshot, assert_cache_equal


def test_action_fifo_ignores_scores_evidence_and_observation_time():
    m=model();p=policy(m)
    mask=torch.zeros(p.capacity,dtype=torch.bool)
    actions,_=append(p,mask,rows([100,101,0,1],kind=1))
    with patch.object(p,"scores",side_effect=AssertionError("action was scored")), patch.object(p,"evidence",side_effect=AssertionError("action used sensory evidence")):
        _,victims=p.plan(mask,2,rows([2,3],kind=1))
    assert victims.tolist()==actions[:2].tolist()


@torch.no_grad()
def test_action_is_not_archived_or_affected_by_sensory_selection():
    m=model();p=policy(m)
    for t in range(3):grounded(m,t,action=True)
    assert all(not (r["metadata"]["kind"]==1).any() for r in p.archive.records)
    c=m.mot.shared_attn[0].attn_caches["test"]
    before=p.data["token_uid"][c["mask"] & (p.data["kind"]==1)].clone()
    grounded(m,8)
    after=p.data["token_uid"][c["mask"] & (p.data["kind"]==1)]
    assert set(before.tolist())==set(after.tolist())


@torch.no_grad()
def test_action_only_attention_matches_baseline_fifo_through_prediction_clear():
    from n0_twam.models.mot import SharedSelfAttention
    torch.manual_seed(42)
    indexed,baseline=SharedSelfAttention(),SharedSelfAttention()
    for attention in (indexed,baseline):
        attention.init_kv_cache("test",4,1,4,torch.device("cpu"),torch.float32,1)
    indexed.attn_caches["test"].update(cpu_archive=True,action_fifo_capacity=4)
    for mode in (2,2,0,1,0,1):
        q,k,v=[torch.randn(1,2,1,4) for _ in range(3)]
        expected=baseline(q,k,v,update_cache=mode,cache_name="test")
        actual=indexed(q,k,v,update_cache=mode,cache_name="test",
                       cache_token_kinds=torch.ones(2,dtype=torch.long),
                       history_token_kinds=torch.ones(4,dtype=torch.long),
                       history_token_uids=indexed.attn_caches["test"]["id"])
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    baseline.clear_pred_cache("test");indexed.clear_pred_cache("test")
    assert not baseline.attn_caches["test"]["mask"].any()
    assert not indexed.attn_caches["test"]["mask"].any()


@torch.no_grad()
def test_predicted_action_fifo_eviction_is_transactional():
    m=model()
    grounded(m,0,action=True);grounded(m,1,action=True)
    before=cache_snapshot(m)
    with pytest.raises(RuntimeError,match="abort"):
        with m.cache_transaction("test"):
            m(model_input(2,True),update_cache=1,cache_name="test",action_mode=True)
            raise RuntimeError("abort")
    assert_cache_equal(before,cache_snapshot(m))
    for attention in m.mot.shared_attn:
        cache=attention.attn_caches["test"]
        assert not cache.get("predicted_kv") and not cache.get("predicted_kinds")
