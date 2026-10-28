import torch
from training.parallel_scorer.memory_objective_v3 import memory_loss

def test_dense_background_does_not_dilute_strict_positive_regression():
    def positive_gradient(background):
        y=torch.tensor([90.,50.,50.]+[10.]*background);z=torch.zeros_like(y,requires_grad=True)
        loss,_=memory_loss(z,y,torch.ones_like(y,dtype=torch.bool),k=2,rank_weight=0,selection_weight=0)
        loss.backward();return z.grad[0]
    assert torch.allclose(positive_gradient(8),positive_gradient(8000))

def test_boundary_is_supervised_and_current_is_excluded():
    y=torch.tensor([90.,50.,50.,10.,100.]);z=torch.tensor([0.,1.,1.,0.,-10.],requires_grad=True);mask=torch.tensor([True,True,True,True,False])
    loss,_=memory_loss(z,y,mask,k=2,rank_weight=0,selection_weight=0);loss.backward()
    assert z.grad[0]<0 and z.grad[1:3].gt(0).all() and z.grad[3]>0 and z.grad[4]==0

def test_equal_labels_and_missing_strata_have_finite_gradients():
    for y,k in [(torch.full((20,),50.),4),(torch.tensor([90.,50.,50.]),3)]:
        z=torch.randn(len(y),requires_grad=True);loss,p=memory_loss(z,y,torch.ones_like(y,dtype=torch.bool),k=k);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
        if y.unique().numel()==1:assert p["ranking"]==0

def test_pair_and_selection_ordering_is_improved_by_correct_scores():
    y=torch.tensor([90.,80.,50.,50.,10.,0.]);mask=torch.ones(6,dtype=torch.bool)
    good=torch.tensor([2.,1.5,0.,0.,-1.5,-2.]);torch.manual_seed(42);a,_=memory_loss(good,y,mask,k=3)
    torch.manual_seed(42);b,_=memory_loss(-good,y,mask,k=3);assert a<b
