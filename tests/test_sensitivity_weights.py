import torch
import pytest
from training.parallel_scorer.rl import SensitivityWeights

def test_high_spread_more_weight_and_detached():
    w=SensitivityWeights()
    f,d,raw=w.weights(torch.tensor([[1.,1.01,1.02,1.],[0.,1.,2.,3.]],requires_grad=True))
    assert f[1]>f[0] and not f.requires_grad
    assert torch.allclose(f.sum(),torch.tensor(1.))
    assert raw.min()>=.25 and raw.max()<=4

def test_equal_zero_and_partial_batch():
    w=SensitivityWeights()
    f,_,_=w.weights([[1.,1.,1.,1.],[9.,9.,9.,9.]])
    assert torch.equal(f,torch.tensor([.5,.5]))
    f,_,_=w.weights([[1.,2.,3.,4.]])
    assert f.item()==1

def test_batch_order_invariant_and_ema():
    a,b=SensitivityWeights(),SensitivityWeights()
    x=[[0.,1.,2.,3.],[0.,.1,.2,.3]]
    f,_,_=a.weights(x);g,_,_=b.weights(x[::-1])
    assert torch.equal(f,g.flip(0)) and a.reference==b.reference
    before=a.reference;a.weights([[0.,0.,0.,0.]])
    assert abs(a.reference-before*.95)<1e-7

def test_weighted_accumulation_keeps_anchor_uniform():
    f,_,_=SensitivityWeights().weights([[0.,1.,2.,3.],[0.,.1,.2,.3]])
    x=torch.tensor(2.,requires_grad=True)
    for i in range(2): (f[i]*(i+1)*x+x*x/2).backward()
    assert torch.allclose(x.grad,(f*torch.tensor([1.,2.])).sum()+4)
