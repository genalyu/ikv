import numpy as np
import pytest
import torch
from training.memory.online_scorer import ObservedPrefix

class Encoder:
    def task(self,prompt):return torch.ones(1,2)
    def images(self,images):
        return torch.cat([torch.full((64,2),float(np.asarray(x)[0,0,0])) for x in images])
class Scorer:
    info={"encoder_revision":"test"}
    def score(self,packet,budgets):
        assert packet["times"].max()<=packet["query_time"]
        return {"scores":packet["features"][:,0]}

def test_online_observation_prefix_is_causal_and_native_ordered():
    bank=ObservedPrefix(Encoder(),Scorer(),"task")
    images=np.zeros((2,3,256,256,3),dtype=np.uint8)
    images[0,0]=1;images[0,1]=2;images[0,2]=3
    images[1]=99
    bank.append(0,images)
    score=bank.score(1)
    assert score[:24].tolist()==[1.]*8+[2.]*8+[3.]*8
    assert score.max()==3
    assert bank.score(2)[192:].eq(99).all()
    bank.append(0,images)  # same request after a lost response is idempotent
    assert len(bank.frames)==2
    images[0]=42
    with pytest.raises(ValueError,match="changed"):bank.append(0,images)
    with pytest.raises(ValueError,match="unavailable"):bank.score(3)
