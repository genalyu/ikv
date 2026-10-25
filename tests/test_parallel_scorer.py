"""CPU correctness tests for exact all-history scoring and native selection."""
import itertools
from copy import deepcopy
import pytest
import torch
from training.parallel_scorer.model import ScorerConfig,ParallelScorer,select_indices,save_scorer
from training.parallel_scorer.data import regression_loss,check_disjoint,load_packet
from training.parallel_scorer.runtime import FeatureBank,RuntimeScorer
from training.parallel_scorer.rl import ordered_logprob,sample_modalities,path_logprob,group_loss
from training.parallel_scorer.train_rl import native_domain,native_scores

torch.set_num_threads(2)
def setup(n=13):
    torch.manual_seed(3)
    c=ScorerConfig(vision_dim=12,text_dim=10,width=16,layers=2,heads=4,
                   query_chunk=4,ff_chunk=5,checkpoint_layers=False)
    model=ParallelScorer(c)
    args=dict(features=torch.randn(n,12),text=torch.randn(5,10),
              times=torch.arange(n).float(),xy=torch.rand(n,2),
              modalities=torch.arange(n)%2,sensors=torch.arange(n)%3,query_time=float(n-1))
    return model,args

def test_full_vs_tiled_outputs_and_gradients():
    m,a=setup();other=deepcopy(m)
    for block in other.blocks:
        block.self_attn.chunk=1000;block.task_attn.chunk=1000;block.chunk=1000
    x=m(**a);y=other(**a)
    torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
    x.square().sum().backward();y.square().sum().backward()
    for p,q in zip(m.parameters(),other.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-6)

def test_permutation_equivariance_and_cross_modal_bidirectionality():
    m,a=setup();m.eval();base=m(**a)
    order=torch.randperm(len(base))
    perm={k:v[order] if k not in ("text","query_time") else v for k,v in a.items()}
    torch.testing.assert_close(m(**perm),base[order],rtol=1e-5,atol=1e-6)
    changed={**a,"features":a["features"].clone()}
    changed["features"][-2]+=5 # late tactile candidate affects early visual candidate
    assert abs(float((m(**changed)[0]-base[0]).detach()))>1e-7
    changed={**a,"text":a["text"]+2}
    assert not torch.allclose(m(**changed),base)

def test_checkpoint_preserves_trainable_input_gradients():
    m,a=setup();other=deepcopy(m);other.config.checkpoint_layers=True
    a["features"].requires_grad_()
    b={**a,"features":a["features"].detach().clone().requires_grad_()}
    m(**a).sum().backward();other(**b).sum().backward()
    torch.testing.assert_close(a["features"].grad,b["features"].grad)

def test_variable_sizes_and_future_rejection():
    for n in (1,7,31):
        m,a=setup(n);assert m(**a).shape==(n,)
    a["query_time"]=-1
    with pytest.raises(ValueError,match="Future"):m(**a)

def test_protected_budget_and_mapping(tmp_path):
    m,a=setup(4)
    packet={**a,"ids":["a","b","c","d"],"encoder_revision":"rev"}
    save_scorer(m,tmp_path/"m.pt",encoder_revision="rev")
    runtime=RuntimeScorer(tmp_path/"m.pt","cpu")
    out=runtime.score(packet,[2,1],kv_candidate_ids=["a","a","b","c","d"],
                      protected=[1,0,1,0,0])
    assert len(out["scores"])==5 and len(out["selected_indices"])==3
    assert {0,2}.issubset(set(out["selected_indices"].tolist()))
    with pytest.raises(ValueError):runtime.score(packet,[0,0],protected=[1,0,0,0])
    packet["encoder_revision"]="other"
    with pytest.raises(ValueError,match="mismatch"):runtime.score(packet,[1,1])

def test_feature_bank_revision_duplicates_and_no_eviction(tmp_path):
    m,a=setup(4);bank=FeatureBank("r")
    bank.append(["d","b","c","a"],*[a[k] for k in ("features","times","xy","modalities","sensors")],"r")
    p=bank.packet(a["text"],a["query_time"]);assert p["ids"]==["a","b","c","d"]
    with pytest.raises(ValueError,match="Duplicate"):
        bank.append(["d","b","c","a"],*[a[k] for k in ("features","times","xy","modalities","sensors")],"r")
    bank.save(tmp_path/"b.pt");assert len(FeatureBank.load(tmp_path/"b.pt","r").rows)==4
    with pytest.raises(ValueError,match="Stale"):FeatureBank.load(tmp_path/"b.pt","new")

def test_ties_do_not_make_false_pairwise_targets():
    z=torch.zeros(30,requires_grad=True);y=torch.full((30,),50.)
    loss=regression_loss(z,y);loss.backward()
    assert loss==0 and z.grad.eq(0).all()

def test_plackett_luce_matches_explicit_and_normalizes():
    z=torch.tensor([.2,-.4,.8],requires_grad=True)
    all_p=[]
    for seq in itertools.permutations(range(3),2):
        order=torch.tensor(seq);lp=ordered_logprob(z,order)
        remaining=[0,1,2];manual=z.sum()*0
        for j in seq:
            manual=manual+z[j]-torch.logsumexp(z[remaining],0);remaining.remove(j)
        torch.testing.assert_close(lp,manual)
        all_p.append(lp.exp())
    torch.testing.assert_close(sum(all_p),torch.tensor(1.))
    ordered_logprob(z,torch.tensor([2,0])).backward();assert torch.isfinite(z.grad).all()

def test_group_policy_gradients_and_budget():
    z=torch.linspace(-1,1,10,requires_grad=True);mods=torch.arange(10)%2
    paths=[sample_modalities(z.detach(),mods,[2,3]) for _ in range(4)]
    assert all(len(p[0])==2 and len(p[1])==3 for p in paths)
    old=torch.stack([path_logprob(z.detach(),mods,p) for p in paths])
    loss=group_loss(z,mods,paths,old,[1.,2.,3.,4.]);loss.backward()
    assert torch.isfinite(z.grad).all()
    assert group_loss(z,mods,paths,old,[1.,1.,1.,1.])==0

def test_native_map_and_sampled_set_match():
    packet=dict(features=torch.zeros(4,2),modalities=torch.tensor([0,0,1,1]))
    record=dict(query_frame=1,target_raw_frame=2,alignment_provenance="verified",
                video_candidate_map=[0,1,-1],tactile_candidate_map=[2,3,-1])
    prepared=dict(latent_dict=dict(targets=torch.zeros(1,1,3,1,1)),
                  action_dict=dict(tactile_global_targets=torch.zeros(1,1,1,3,1,1)))
    index,mods,maps=native_domain(packet,record,prepared,(1,1,1))
    assert index.tolist()==[0,1,2,3]
    values=native_scores([torch.tensor([1]),torch.tensor([2])],maps)
    assert values[0].tolist()==[0,1,0] and values[1].tolist()==[1,0,0]
    record["video_candidate_map"]=[0,-1,-1]
    with pytest.raises(ValueError):native_domain(packet,record,prepared,(1,1,1))

def test_episode_split_guard():
    with pytest.raises(ValueError):check_disjoint([dict(episode_id="x")],[dict(episode_id="x")])

def test_explicit_tactile_selection_preserves_action_and_early_phases():
    from n0_twam.models.scorer_support import apply_scorer_support
    layout=dict(seq=torch.zeros(12,dtype=torch.long),
       phase=torch.tensor([0,0,0,0,0,0,2,2,2,2,3,3]),
       clean=torch.tensor([1,1,1,1,1,1,1,1,1,1,0,0],dtype=torch.bool),
       kind=torch.tensor([0,0,1,1,2,2,0,0,2,2,0,1]))
    memory=dict(layout=layout,rows=dict(world_time_id=torch.arange(12)),config=dict(capacity=3))
    plan=dict(noisy=torch.zeros(4,12,dtype=torch.bool),clean=torch.ones(4,12,dtype=torch.bool))
    spec=dict(target_phase=2,video_scores=[2.,1.,0.,0.],tactile_scores=[9.,1.,0.,0.],budgets=[1,1,1])
    result=apply_scorer_support(plan,memory,spec)
    assert result["noisy"][2].nonzero().flatten().tolist()==[0,3,4]
    assert not result["noisy"][:2].any()
    assert torch.equal(result["clean"],plan["clean"])
