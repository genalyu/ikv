"""Stage-1 raw heuristic and train/inference separation regressions."""
import pytest
import torch
from n0_twam.models.global_kv_retention import RetentionConfig
from n0_twam.models.multimodal_kv_retention import make_retention_policy
from test_global_kv_retention import rows, append

def policy(**kw):
    cfg=dict(version=2,video_capacity=4,action_capacity=2,tactile_capacity=4,
             seed=42)
    cfg.update(kw)
    return make_retention_policy(10,"cpu",RetentionConfig(**cfg))

def test_visual_screenshot_exact_and_contact_decay():
    p=policy();mask=torch.zeros(10,dtype=torch.bool)
    slots,_=append(p,mask,rows([0,8],dino=torch.eye(2)))
    p.data["contact_duration"][slots]=torch.tensor([0.,8.])
    p.observe_dense(torch.eye(2),torch.tensor([0.,8.]),torch.ones(2))
    expected=torch.tensor([(torch.exp(torch.tensor(-1.))+1+1),
                           (1+torch.exp(torch.tensor(-1.))+1)])
    torch.testing.assert_close(p.heuristic_scores(slots,t0=8),expected)
    p.data["contact_duration"][slots[0]]=16
    assert p.heuristic_scores(slots,t0=8)[0]<expected[0]

def test_tactile_has_independent_class_recency():
    p=policy();mask=torch.zeros(10,dtype=torch.bool)
    first=rows([0,0],kind=2);first["neoforce"]=torch.eye(2)
    slots,_=append(p,mask,first)
    later=rows([8],kind=2);later["neoforce"]=torch.tensor([[0.,1.]])
    append(p,mask,later)
    e=torch.exp(torch.tensor(-1.))
    torch.testing.assert_close(p.heuristic_scores(slots,t0=8),torch.tensor([e+1,2*e]))
    assert p.history.clock is None

def test_predictions_and_missing_features_have_zero_content_bonus():
    p=policy();mask=torch.zeros(10,dtype=torch.bool)
    incoming=rows([0],kind=2);incoming["neoforce"]=torch.tensor([[1.,0.]])
    incoming["observation_flag"][:]=False
    slots,_=append(p,mask,incoming)
    assert p.tactile_history.clock is None
    torch.testing.assert_close(p.heuristic_scores(slots,t0=8),torch.exp(torch.tensor([-1.])))

def test_inference_has_no_mix_and_does_not_use_global_rng():
    p=policy()
    mask=torch.zeros(10,dtype=torch.bool)
    slots,_=append(p,mask,rows([0,1,2],dino=torch.eye(3)))
    state=torch.random.get_rng_state().clone()
    a=p.scores(slots,t0=8)
    torch.testing.assert_close(a,p.heuristic_scores(slots,t0=8))
    torch.testing.assert_close(a.flip(0),p.scores(slots.flip(0),t0=8))
    assert torch.equal(state,torch.random.get_rng_state())
    p.config=__import__("dataclasses").replace(p.config,seed=997)
    torch.testing.assert_close(a,p.scores(slots,t0=8))

def test_prototype_threshold_point_75_and_clamped_age():
    from n0_twam.models.multimodal_kv_retention import ContentHistory
    h=ContentHistory(policy().config,"cpu")
    h.observe(torch.tensor([[1.,0.]]),torch.tensor([8.]),torch.ones(1))
    x=torch.tensor([[.75,(1-.75**2)**.5],[.749,(1-.749**2)**.5],[1.,0.]])
    got=h.class_recency(x,torch.tensor([0.,0.,12.]))
    torch.testing.assert_close(got,torch.tensor([torch.exp(torch.tensor(-1.)),0.,1.]))

def test_training_weights_random_reproducible_and_inference_reset():
    from n0_twam.models.ikv_training import set_training_heuristic
    p=policy(); cfg={"sample_heuristic_weights":True}
    torch.manual_seed(7); set_training_heuristic(p,cfg,"cpu",0,0)
    first=p.training_weights.clone()
    assert ((first>=0)&(first<2)).all() and not torch.equal(first,torch.ones(3))
    set_training_heuristic(p,cfg,"cpu",0,1)
    assert not torch.equal(first,p.training_weights)
    torch.manual_seed(7);set_training_heuristic(p,cfg,"cpu",0,0)
    torch.testing.assert_close(first,p.training_weights)
    set_training_heuristic(p,{},"cpu",0,0)
    assert p.training_weights is None

def test_raw_weighted_formula():
    from n0_twam.models.heuristic_score import heuristic_scores
    times=torch.tensor([0.,8.,16.])
    got=heuristic_scores(times,8,torch.tensor([1,1,0]),torch.tensor([0.,8.,50.]),
                         torch.tensor([1.,0.,1.]),weights=(2,3,4))
    e=torch.exp(torch.tensor(-1.))
    torch.testing.assert_close(got,torch.tensor([2*e+3+4,2+3*e,6.]))

def test_snapshot_restores_tactile_prototypes():
    p=policy();snapshot=p.snapshot();mask=torch.zeros(10,dtype=torch.bool)
    incoming=rows([0],kind=2);incoming["neoforce"]=torch.ones(1,2)
    append(p,mask,incoming);assert p.tactile_history.clock==0
    p.restore(snapshot);assert p.tactile_history.clock is None

@pytest.mark.parametrize("kw",[dict(random_weight=-.1),dict(random_weight=1.1),
    dict(formula_variant="screenshot_v1"),dict(formula_variant="legacy",random_weight=.1),
    dict(random_weight=.1),dict(content_threshold=.9),
    dict(class_recency_scale=4),dict(content_threshold=.8)])
def test_old_formula_config_rejected(kw):
    with pytest.raises(ValueError):policy(**kw)

def training_memory():
    # Six query/observation phases, two resident visual slots. Older untouched
    # patches compete against newer contacted patches, so eta changes rankings.
    phase=torch.arange(6).repeat_interleave(2)
    clean=torch.tensor([True,False]*6)
    metadata=rows(phase.tolist(),dino=torch.tensor([[1.,0.]]*12))
    metadata["contact_present"]=phase>0
    metadata["contact_duration"]=phase.float()*16
    return dict(config=dict(capacity=4, sample_heuristic_weights=True,
                    history_cache_key=123456,
                    retention=dict(video_capacity=2,action_capacity=1,tactile_capacity=1,
                        query_weight=0.,action_query_weight=0.,tactile_query_weight=0.,
                        persistence_weight=0.,class_recency_weight=1.)),
                layout=dict(seq=torch.zeros(12,dtype=torch.long),phase=phase,
                            clean=clean,kind=torch.zeros(12,dtype=torch.long)),
                rows=metadata,dense_dino_features=torch.tensor([[[[1.,0.]]]*6]))

def test_real_training_support_plan_samples_and_replays_with_cache():
    from n0_twam.models.ikv_training import build_ikv_support_plan
    from copy import deepcopy
    memory=training_memory()
    selected=set()
    for seed in range(12):
        a=deepcopy(memory);b=deepcopy(memory)
        torch.manual_seed(seed)
        plan=build_ikv_support_plan(a,"cpu")
        torch.manual_seed(seed)
        repeated=build_ikv_support_plan(b,"cpu")
        torch.testing.assert_close(plan["clean"],repeated["clean"])
        torch.testing.assert_close(plan["noisy"],repeated["noisy"])
        assert a["config"]["sampled_heuristic_weights"]==b["config"]["sampled_heuristic_weights"]
        assert len(a["config"]["sampled_heuristic_weights"])==6
        selected.add(tuple(plan["noisy"][5].tolist()))
    assert len(selected)>1, "sampled term weights must change actual sparse masks"

def test_training_sample_does_not_modify_inference_policy():
    from n0_twam.models.ikv_training import build_ikv_support_plan
    p=policy(); mask=torch.zeros(10,dtype=torch.bool)
    slots,_=append(p,mask,rows([0,1,2],dino=torch.eye(3)))
    expected=p.scores(slots,t0=8).clone()
    build_ikv_support_plan(training_memory(),"cpu")
    assert p.training_weights is None
    torch.testing.assert_close(expected,p.scores(slots,t0=8))

@pytest.mark.parametrize("checkpoint",[False,True])
def test_sampled_eta_real_mot_recurrent_masked_parity_and_gradients(checkpoint):
    from copy import deepcopy
    from test_masked_ikv_training import prepared_case
    from test_global_kv_retention import tiny_model
    from n0_twam.models.ikv_training import run_ikv_training,run_ikv_masked_training
    left=tiny_model(False).mot;right=deepcopy(left)
    left.gradient_checkpointing=right.gradient_checkpointing=checkpoint
    hidden,text,timestep,temb,rope,memory=prepared_case(4,2)
    memory["config"]["sample_heuristic_weights"]=True
    second=deepcopy(memory)
    a=hidden.detach().clone().requires_grad_()
    b=hidden.detach().clone().requires_grad_()
    torch.manual_seed(194)
    expected=run_ikv_training(left,a,text,timestep,temb,rope,memory)
    torch.manual_seed(194)
    actual=run_ikv_masked_training(right,b,text,timestep,temb,rope,second)
    assert memory["config"]["sampled_heuristic_weights"]==second["config"]["sampled_heuristic_weights"]
    weights=deepcopy(second["config"]["sampled_heuristic_weights"])
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(b.grad,a.grad,rtol=3e-4,atol=3e-6)
    assert weights==second["config"]["sampled_heuristic_weights"]
    for (_,p),(_,q) in zip(left.named_parameters(),right.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad,p.grad,rtol=3e-4,atol=3e-6)

def test_snapshot_restores_training_eta():
    p=policy();p.training_weights=torch.tensor([.2,1.,1.8])
    state=p.snapshot()
    p.training_weights=torch.ones(3)
    p.restore(state)
    torch.testing.assert_close(p.training_weights,torch.tensor([.2,1.,1.8]))
