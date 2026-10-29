import torch
import pytest
from training.memory.selection import replay
from training.memory.serving_policy import FrozenMemoryRetention, native_ids
from n0_twam.models.global_kv_retention import token_rows
from n0_twam.utils.utils import get_mesh_id

def rows(start,width,kind,update=2):
    grid=get_mesh_id(width,12 if kind else 8,1 if kind else 24,kind,1,start,action=bool(kind))[None]
    return token_rows(dict(grid_id=grid),batch_size=1,length=grid.shape[-1],
        main_count=grid.shape[-1],action_mode=bool(kind),update_cache=update,device="cpu")

@pytest.mark.parametrize("frames",[7,8,23])
def test_actual_policy_matches_offline_plan_with_denoise_retries(frames):
    budgets=(768,48)
    scorer=lambda n:torch.sin(torch.arange(n*192).float()*.19)
    expected=replay(frames,scorer,budgets=budgets,chunk_size=2)
    policy=FrozenMemoryRetention("cpu",scorer,budgets)
    mask=torch.zeros(sum(budgets),dtype=torch.bool)
    for stage,wanted in enumerate(expected):
        start=(stage//2)*2;kind=stage%2;width=min(2,frames-start)
        r=rows(start,width,kind)
        before=policy.snapshot()
        first=policy.plan(mask,len(r["kind"]),r)
        # Diffusion retries share identical history; planning has no mutation.
        for _ in range(2):
            again=policy.plan(mask,len(r["kind"]),r)
            assert all(torch.equal(a,b) for a,b in zip(first,again))
        assert policy.next_uid==before[0][1]
        slots,victims=first
        kept=mask.clone();kept[victims]=False
        live=policy.data["memory_id"][kept]
        canonical=torch.where(live%2==0,live//2,frames*192+live//2)
        assert sorted(canonical.tolist())==sorted(wanted[wanted>=0].tolist())
        old=mask.clone();mask[victims]=False;mask[slots]=True
        policy.commit(slots,r,old)
        assert mask.sum()<=sum(budgets)

def test_native_policy_rejects_duplicate_and_wrong_grid():
    r=rows(0,2,0);r["grid_position"][0,1]=24
    with pytest.raises(ValueError,match="8x24"):native_ids(r)


def commit_rows(policy, mask, r):
    old=mask.clone()
    slots,victims=policy.plan(mask,len(r["kind"]),r)
    mask[victims]=False
    mask[slots]=True
    policy.commit(slots,r,old)
    return victims


def test_history_context_protects_current_in_prediction_and_scores_grounding():
    # Fill four real frames; only one old frame can survive a two-frame
    # prediction once the latest real frame is protected.
    observed=[4];calls=[]
    def score(history, context):
        calls.append((history,context))
        return -torch.arange(history*192).float()
    policy=FrozenMemoryRetention("cpu",lambda n:pytest.fail("legacy called"),
        budgets=(768,48),score_history=score,observed_prefix=lambda:observed[0])
    mask=torch.zeros(816,dtype=torch.bool)
    commit_rows(policy,mask,rows(0,2,0))
    commit_rows(policy,mask,rows(2,2,0))
    # Lowest scores intentionally favor old IDs; protection must override them.
    before=policy.data["memory_id"].clone()
    incoming=rows(4,2,0,update=0)
    slots,victims=policy.plan(mask,384,incoming)
    assert calls[-1]==(3,4)
    kept=before[mask & ~torch.isin(torch.arange(816),victims)]//2
    assert set(kept.tolist())==set(range(192))|set(range(3*192,4*192))
    assert torch.equal(before,policy.data["memory_id"])
    assert all(torch.equal(a,b) for a,b in zip((slots,victims),policy.plan(mask,384,incoming)))
    # New real observations are already available at grounding time. Both
    # incoming frames remain reserved; the last old frame is now history.
    observed[0]=6
    commit_rows(policy,mask,rows(4,2,0))
    assert calls[-1]==(4,6)
    live=policy.data["memory_id"][mask]//2
    assert set(live.tolist())==set(range(384))|set(range(4*192,6*192))
    assert not torch.isin(torch.arange(2*192,4*192),live).any()


def test_history_context_action_recency_and_snapshot_restore():
    observed=[6]
    policy=FrozenMemoryRetention("cpu",None,budgets=(768,24),
        score_history=lambda h,o:-torch.arange(h*192).float(),
        observed_prefix=lambda:observed[0])
    mask=torch.zeros(792,dtype=torch.bool)
    commit_rows(policy,mask,rows(0,2,0))
    commit_rows(policy,mask,rows(2,2,0))
    commit_rows(policy,mask,rows(0,2,1))
    commit_rows(policy,mask,rows(2,2,1))
    ids=policy.data["memory_id"][mask]
    assert set(ids[ids%2==1].tolist())==set((2*torch.arange(24,48)+1).tolist())
    snapshot=policy.snapshot();oldmask=mask.clone();oldids=ids.clone()
    commit_rows(policy,mask,rows(4,2,0))
    policy.restore(snapshot);mask.copy_(oldmask)
    assert torch.equal(policy.data["memory_id"][mask],oldids)


@pytest.mark.parametrize("bad", [0,3,4.0])
def test_history_context_rejects_unavailable_or_noninteger_observations(bad):
    policy=FrozenMemoryRetention("cpu",None,budgets=(768,48),
        score_history=lambda h,o:torch.zeros(h*192),observed_prefix=lambda:bad)
    mask=torch.zeros(816,dtype=torch.bool)
    commit_rows(policy,mask,rows(0,2,0))
    commit_rows(policy,mask,rows(2,2,0))
    with pytest.raises(ValueError,match="observed prefix"):
        policy.plan(mask,384,rows(4,2,0))


def test_history_context_rejects_partial_configuration_and_malformed_scores():
    with pytest.raises(ValueError,match="provider"):
        FrozenMemoryRetention("cpu",None,score_history=lambda h,o:[])
    policy=FrozenMemoryRetention("cpu",None,budgets=(768,48),
        score_history=lambda h,o:torch.zeros(o*192),observed_prefix=lambda:4)
    mask=torch.zeros(816,dtype=torch.bool)
    commit_rows(policy,mask,rows(0,2,0));commit_rows(policy,mask,rows(2,2,0))
    with pytest.raises(ValueError,match="malformed"):
        policy.plan(mask,384,rows(4,2,0))


def test_serving_history_context_wires_observed_count_dynamically():
    from types import SimpleNamespace
    from training.memory.serving import frozen_policy
    class Scorer:
        next_frame=4
        def score(self,n):raise AssertionError("legacy called")
        def score_history(self,h,o):return torch.zeros(h*192)
    server=SimpleNamespace(memory_scorer=Scorer(),
        job_config=SimpleNamespace(memory_history_context=True))
    p=frozen_policy(server,"cpu")
    assert p.observed_prefix()==4
    server.memory_scorer.next_frame=6
    assert p.observed_prefix()==6
    server.job_config.memory_history_context=False
    assert frozen_policy(server,"cpu").score_history is None
