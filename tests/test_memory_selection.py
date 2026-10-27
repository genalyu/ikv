import torch
import pytest
from training.memory.selection import evict, replay, scorer_to_native, support_from_plan


def test_camera_row_mapping():
    source=torch.arange(192).reshape(1,3,8,8)
    native=scorer_to_native(source)
    assert native[0,:24].tolist()==list(range(8))+list(range(64,72))+list(range(128,136))
    assert native[0,24:32].tolist()==list(range(8,16))


def test_streaming_causality_and_no_resurrection():
    calls=[]
    def scorer(n):
        calls.append(n)
        # Old tokens become highly relevant again, but cannot be resurrected.
        return torch.arange(n*192).float() * (-1 if n>=4 else 1)
    p=replay(6,scorer,budgets=(384,24),chunk_size=1)
    assert calls==[2,3,4,5]  # no current target frame, even for action phases
    V=6*192
    live=set()
    for stage,row in enumerate(p):
        kept=set(row[row>=0].tolist())
        assert kept<=live
        f,k=divmod(stage,2)
        assert all((x//192*2 if x<V else (x-V)//12*2+1)<stage for x in kept)
        start=f*192 if not k else V+f*12
        count=192 if not k else 12
        live=kept|set(range(start,start+count))
        assert sum(i<V for i in live)<=384
        assert sum(i>=V for i in live)<=24


def test_ties_use_native_ids_and_overflow_is_error():
    assert evict([9,2,7],[0,0,0],[1,1,1],(0,0),(2,0)).tolist()==[2,7]
    with pytest.raises(ValueError,match="exceeds"):
        evict([],[],[],(193,0),(192,0))


def test_support_rejects_future():
    layout=dict(seq=torch.zeros(8,dtype=torch.long),phase=torch.tensor([0,2,0,2,1,3,1,3]),
        clean=torch.tensor([0,0,1,1,0,0,1,1],dtype=torch.bool),kind=torch.tensor([0,0,0,0,1,1,1,1]))
    p=torch.full((4,2),-1);p[2]=torch.tensor([0,2])
    support=support_from_plan(layout,p,2)
    assert support["noisy"][2].nonzero().flatten().tolist()==[2,6]
    p[0,0]=1
    with pytest.raises(ValueError,match="leakage"):
        support_from_plan(layout,p,2)


def test_native_two_frame_groups_never_consult_group_targets():
    calls=[]
    p=replay(7,lambda n:(calls.append(n) or torch.arange(n*192).float()),
             budgets=(768,48),chunk_size=2)
    assert calls==[4,6]
    assert p.shape==(8,816)
    V=7*192
    live=set()
    for stage,row in enumerate(p):
        kept=set(row[row>=0].tolist())
        assert kept<=live
        frame=(stage//2)*2
        kind=stage%2
        width=min(2,7-frame)
        assert all((x//192//2*2 if x<V else (x-V)//12//2*2+1)<stage for x in kept)
        start=frame*192 if kind==0 else V+frame*12
        count=width*(192 if kind==0 else 12)
        live=kept|set(range(start,start+count))
        assert sum(x<V for x in live)<=768
        assert sum(x>=V for x in live)<=48
