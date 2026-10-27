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

class ContextScorer:
    info={"encoder_revision":"test"}
    def score(self,packet,budgets):
        return {"scores":packet["features"][:,0]+packet["features"][:,0].mean()}

class CountingEncoder(Encoder):
    calls=0
    def images(self,images):
        self.calls+=1
        return super().images(images)

def test_history_conditions_on_current_without_returning_current_or_future():
    encoder=CountingEncoder()
    bank=ObservedPrefix(encoder,ContextScorer(),"task")
    images=np.zeros((2,3,256,256,3),dtype=np.uint8)
    images[0]=2;images[1]=10
    bank.append(0,images)
    before=bank.score(1).clone()
    conditioned=bank.score_history(1,2)
    assert conditioned.shape==(192,)
    assert not torch.equal(before,conditioned)
    assert conditioned.eq(8).all()
    bank.append(2,np.full((1,3,256,256,3),255,dtype=np.uint8))
    assert torch.equal(bank.score_history(1,2),conditioned)  # explicit boundary excludes later observations
    assert bank.score_history(0,1).numel()==0
    bank.score_history(1,2)
    assert encoder.calls==3  # no repeated encoder calls when rescoring context
    for history,context in [(1,1),(2,1),(1,4),(-1,2)]:
        with pytest.raises(ValueError,match="History must precede"):
            bank.score_history(history,context)
