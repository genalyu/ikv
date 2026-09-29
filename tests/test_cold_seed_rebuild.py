"""Cold seed reconstruction must match legacy grounding and remain transactional."""
import pytest
import torch
from test_global_kv_retention import (tiny_model, model_input, cache_snapshot,
    assert_cache_equal, assert_snapshot_equal)
from test_rgb_motion_server_helpers import _kv_lifecycle_server
from test_contact_pair_server import inputs

@pytest.mark.parametrize("global_policy",[False,True])
@torch.no_grad()
def test_including_observed_clear_rolls_back_all_layers(global_policy):
    model=tiny_model(global_policy)
    model(model_input(0),update_cache=2,cache_name="test")
    model(model_input(1),update_cache=1,cache_name="test")
    before=cache_snapshot(model)
    policy=model.mot.retention_policies.get("test")
    state=None if policy is None else policy.snapshot()
    with pytest.raises(RuntimeError,match="abort"):
        with model.cache_transaction("test"):
            model.clear_pred_cache("test",include_observed=True)
            assert all(not a.attn_caches["test"]["mask"].any() for a in model.mot.shared_attn)
            model(model_input(0),update_cache=2,cache_name="test")
            raise RuntimeError("abort")
    assert_cache_equal(before,cache_snapshot(model))
    if policy is not None: assert_snapshot_equal(state,policy.snapshot())

def test_dense_cold_rebuilds_seed_at_zero_then_preserves_history():
    server,captured=_kv_lifecycle_server(sparse=False)
    server.job_config.kv_cache_policy="global"
    calls=[]
    server.transformer.clear_pred_cache=lambda name,**kw: calls.append(kw)
    server._init_kv_index=None
    server._compute_kv_cache({"obs":[],"state":torch.zeros(1)})
    assert calls==[{"include_observed":True}]
    assert captured["latent"].shape[2]==3
    assert captured["action"].shape[2]==3
    assert captured["kwargs"]["frame_st_id"]==0
    assert server.frame_st_id==3
    server._compute_kv_cache({"obs":[],"state":torch.zeros(1)})
    assert calls==[{"include_observed":True},{}]
    assert captured["kwargs"]["frame_st_id"]==3
    assert server.frame_st_id==5

def test_cold_mixed_layout_retains_tactile_without_false_visual_pair():
    server,forward,index,visual,tactile,pairs=inputs()
    tactile["tactile_global_latent"]=tactile["tactile_global_latent"][:,:,:,0:1]
    pairs={key:value[[0,2]] for key,value in pairs.items()}
    server._attach_observed_contact_pairs(
        {"contact_pairs":pairs},forward,index,visual,tactile,cold_seed_frames=1)
    tail=forward["latent_res_lst"]["tactile_kv_index"]
    torch.testing.assert_close(tail["neoforce"],pairs["neoforce"])
    assert tail["dino"].eq(0).all()
    assert pairs["visual_rows"].tolist()==[0,1]
    assert forward["action_res_lst"]["tactile_kv_index"] is tail
