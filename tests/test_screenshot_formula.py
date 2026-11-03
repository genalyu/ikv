"""Screenshot formula and deterministic random-mixture regressions."""
import pytest
import torch
from n0_twam.models.global_kv_retention import RetentionConfig
from n0_twam.models.multimodal_kv_retention import make_retention_policy
from test_global_kv_retention import rows, append

def policy(**kw):
    cfg=dict(version=2,video_capacity=4,action_capacity=2,tactile_capacity=4,
             formula_variant="screenshot_v1",random_weight=0.,seed=42)
    cfg.update(kw)
    return make_retention_policy(10,"cpu",RetentionConfig(**cfg))

def test_visual_screenshot_exact_and_contact_decay():
    p=policy();mask=torch.zeros(10,dtype=torch.bool)
    slots,_=append(p,mask,rows([0,8],dino=torch.eye(2)))
    p.data["contact_duration"][slots]=torch.tensor([0.,8.])
    p.observe_dense(torch.eye(2),torch.tensor([0.,8.]),torch.ones(2))
    expected=torch.tensor([(torch.exp(torch.tensor(-1.))+1+1)/3,
                           (1+torch.exp(torch.tensor(-1.))+1)/3])
    torch.testing.assert_close(p.scores(slots,t0=8),expected)
    p.data["contact_duration"][slots[0]]=16
    assert p.scores(slots,t0=8)[0]<expected[0]

def test_tactile_has_independent_class_recency():
    p=policy();mask=torch.zeros(10,dtype=torch.bool)
    first=rows([0,0],kind=2);first["neoforce"]=torch.eye(2)
    slots,_=append(p,mask,first)
    later=rows([8],kind=2);later["neoforce"]=torch.tensor([[0.,1.]])
    append(p,mask,later)
    e=torch.exp(torch.tensor(-1.))
    torch.testing.assert_close(p.scores(slots,t0=8),torch.tensor([(e+1)/2,e]))
    assert p.history.clock is None

def test_predictions_and_missing_features_have_zero_content_bonus():
    p=policy();mask=torch.zeros(10,dtype=torch.bool)
    incoming=rows([0],kind=2);incoming["neoforce"]=torch.tensor([[1.,0.]])
    incoming["observation_flag"][:]=False
    slots,_=append(p,mask,incoming)
    assert p.tactile_history.clock is None
    torch.testing.assert_close(p.scores(slots,t0=8),torch.exp(torch.tensor([-1.]))/2)

def test_random_mix_repeats_under_permutation_and_does_not_use_global_rng():
    p=policy(random_weight=.1);base=policy()
    mask=torch.zeros(10,dtype=torch.bool);other=torch.zeros_like(mask)
    incoming=rows([0,1,2],dino=torch.eye(3))
    slots,_=append(p,mask,incoming);append(base,other,incoming)
    state=torch.random.get_rng_state().clone()
    a=p.scores(slots,t0=8);b=base.scores(slots,t0=8)
    assert torch.equal(state,torch.random.get_rng_state())
    torch.testing.assert_close(a,p.scores(slots,t0=8))
    torch.testing.assert_close(a.flip(0),p.scores(slots.flip(0),t0=8))
    u=(a-.9*b)/.1
    assert ((u>=0)&(u<=1)).all()
    assert not torch.equal(a,p.scores(slots,t0=9))

def test_snapshot_restores_tactile_prototypes():
    p=policy();snapshot=p.snapshot();mask=torch.zeros(10,dtype=torch.bool)
    incoming=rows([0],kind=2);incoming["neoforce"]=torch.ones(1,2)
    append(p,mask,incoming);assert p.tactile_history.clock==0
    p.restore(snapshot);assert p.tactile_history.clock is None

@pytest.mark.parametrize("kw",[dict(random_weight=-.1),dict(random_weight=1.1),
    dict(formula_variant="invalid"),dict(formula_variant="legacy",random_weight=.1),
    dict(class_recency_scale=4),dict(content_threshold=.8)])
def test_invalid_screenshot_config_rejected(kw):
    with pytest.raises(ValueError):policy(**kw)
