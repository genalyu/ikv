"""Long-stream differential coverage: indexing alone must not change computation."""
from copy import deepcopy
import pytest
import torch
from test_global_kv_retention import tiny_model, model_input

@pytest.mark.parametrize("batch",[1,2])
@torch.no_grad()
def test_indexed_stream_matches_fifo_through_repeated_grounding_without_eviction(batch):
    indexed=tiny_model()
    fifo=tiny_model(False)
    fifo.load_state_dict(indexed.state_dict())
    for model in (indexed,fifo):
        model.create_empty_cache("test",72,16,8,torch.device("cpu"),torch.float32,batch)
        for sa in model.mot.shared_attn:
            sa.attn_caches["test"]["k"].zero_()
            sa.attn_caches["test"]["v"].zero_()
    indexed.configure_global_retention("test",top_k=3,seed=17)
    def compare_step(t,action,update,cold=False):
        original=model_input(t,action)
        if batch==2:
            original={k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in original.items()}
        decorated=deepcopy(original)
        count=2 if action else 4
        decorated["kv_index"]={"dino":torch.randn(count,5),
                               "neoforce":torch.randn(count,2),
                               "observation_flag":int(update==2 or cold)}
        expected=fifo(deepcopy(original),update_cache=update,cache_name="test",action_mode=action)
        actual=indexed(decorated,update_cache=update,cache_name="test",action_mode=action)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,y in zip(indexed.mot.shared_attn,fifo.mot.shared_attn):
            x,y=x.attn_caches["test"],y.attn_caches["test"]
            for key in ("k","v","mask","id"):
                torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
    for t in range(24):
        for action in (False,True):
            compare_step(t,action,0,cold=t==0)
            compare_step(t,action,1,cold=t==0)
        with indexed.cache_transaction("test"),fifo.cache_transaction("test"):
            indexed.clear_pred_cache("test",include_observed=t==0)
            fifo.clear_pred_cache("test")
            for action in (False,True):
                compare_step(t,action,2)
        assert indexed.mot.shared_attn[0].attn_caches["test"]["mask"].sum()==14*(t+1)
