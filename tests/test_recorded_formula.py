import math
import pytest
import torch
from training.parallel_scorer.recorded_formula import recorded_visual_formula

def test_recorded_formula_clocks_contact_and_class_age():
    v=torch.tensor([0.,8.,16.])
    r=recorded_visual_formula(v,16.,torch.tensor([0.,8.,0.]),torch.ones(3,dtype=torch.bool),
                             torch.tensor([.9,.95,1.]),torch.full((3,),16.),torch.ones(3,dtype=torch.bool))
    expected=torch.tensor([2*math.exp(-2)+1,3*math.exp(-1),3.])
    torch.testing.assert_close(r['score'],expected)

def test_recorded_formula_gates_and_no_direct_similarity_reward():
    z=torch.zeros(4);ones=torch.ones(4)
    r=recorded_visual_formula(z,0.,z,torch.tensor([1,1,0,1]),
        torch.tensor([.89,.9,1.,1.]),z,torch.tensor([1,1,1,0]))
    torch.testing.assert_close(r['score'],torch.tensor([2.,3.,2.,2.]))
    a=recorded_visual_formula(z,0.,z,ones,torch.full((4,),.91),z,ones)
    b=recorded_visual_formula(z,0.,z,ones,ones,z,ones)
    torch.testing.assert_close(a['score'],b['score'])

def test_negative_contact_rejected():
    with pytest.raises(ValueError,match='Negative'):
        recorded_visual_formula(torch.zeros(1),0.,torch.tensor([-1.]),torch.ones(1),
                                torch.ones(1),torch.zeros(1),torch.ones(1))

