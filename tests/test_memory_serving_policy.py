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
