import torch
import pytest
from training.parallel_scorer.memory_objective import history_mask,soft_topk,memory_loss,selection_metrics

def test_current_is_context_only_and_supervision_mask_is_respected():
    p=dict(times=torch.tensor([0.,1.,2.]),query_time=2.,supervised=torch.tensor([True,False,True]))
    m=history_mask(p);assert m.tolist()==[True,False,False]
    z=torch.tensor([.2,20.,-20.],requires_grad=True)
    loss,_=memory_loss(z,torch.tensor([90.,0.,100.]),m,k=1,pairs=16)
    loss.backward();assert z.grad[0]!=0 and z.grad[1:].eq(0).all()

def test_soft_topk_mass_shift_invariance_and_implicit_gradient():
    z=torch.tensor([-1.,-.2,.5,1.2],dtype=torch.double,requires_grad=True)
    p=soft_topk(z,2);assert abs(float(p.detach().sum())-2)<1e-7
    assert torch.allclose(p,soft_topk(z+20,2),atol=1e-7)
    assert torch.autograd.gradcheck(lambda x:soft_topk(x,2),(z,),eps=1e-4,atol=1e-4,rtol=1e-3)
    gradient=torch.autograd.grad((p*torch.tensor([.1,.3,.8,1.])).sum(),z)[0]
    assert abs(float(gradient.sum()))<1e-8

def test_boundary_ties_are_supervised_and_equal_labels_do_not_rank():
    y=torch.tensor([90.,50.,50.,10.]);z=torch.zeros(4,requires_grad=True);mask=torch.ones(4,dtype=torch.bool)
    loss,_=memory_loss(z,y,mask,k=2,pairs=256,rank_weight=0,selection_weight=0)
    loss.backward();assert z.grad[0]<0 and z.grad[-1]>0
    a=torch.tensor([0.,1.,0.,0.],requires_grad=True)
    changed,_=memory_loss(a,y,mask,k=2,rank_weight=0,selection_weight=0);assert changed>loss
    same=torch.full((8,),50.);uniform=torch.randn(8,requires_grad=True)
    total,p=memory_loss(uniform,same,torch.ones(8,dtype=torch.bool),k=3)
    assert p["ranking"]==0;assert abs(float(p["selection"]))<1e-5
    total.backward();assert torch.isfinite(uniform.grad).all()

def test_tie_aware_metrics_and_history_eligibility():
    y=torch.tensor([90.,50.,50.,10.,100.]);mask=torch.tensor([True,True,True,True,False])
    m=selection_metrics(torch.tensor([4.,1.,2.,0.,100.]),y,mask,k=2)
    assert m["tie_aware_overlap"]==1 and abs(m["ndcg"]-1)<1e-6
    assert m["candidates"]==4 and m["regret"]==0
    bad=selection_metrics(torch.tensor([0.,1.,2.,4.,100.]),y,mask,k=2)
    assert bad["regret"]>0 and bad["tie_aware_overlap"]<1
    with pytest.raises(ValueError):memory_loss(torch.ones(2),torch.ones(2),torch.zeros(2,dtype=torch.bool))
