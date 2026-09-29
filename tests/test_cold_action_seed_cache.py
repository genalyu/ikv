"""Cold clamped actions must survive the first real-observation grounding."""
import pytest
import torch
from test_dense_kv_index_server import dense_server, TWAMServer
from test_global_kv_retention import tiny_model
from n0_twam.utils.utils import get_mesh_id
from n0_twam.models.global_kv_retention import token_rows

def prepared(monkeypatch, *, cfg=False, clamp=True, policy="global"):
    server = dense_server()
    server.job_config.kv_cache_policy = policy
    server.action_mask = torch.ones(3, dtype=torch.bool)
    server.prompt_embeds = torch.randn(1, 3, 8)
    server.negative_prompt_embeds = torch.zeros(1, 3, 8)
    server.use_cfg = cfg
    monkeypatch.setitem(TWAMServer._prepare_latent_input.__globals__, "get_mesh_id", get_mesh_id)
    actions = torch.randn(1, 3, 2, 2, 1)
    data = server._prepare_latent_input(
        None, actions, action_t=5,
        action_cond=torch.zeros(1,3,1,2,1) if clamp else None,
        frame_st_id=0)["action_res_lst"]
    return server._repeat_input_for_cfg(data)

@pytest.mark.parametrize("cfg", [False, True])
def test_only_clamped_first_action_frame_is_observed(monkeypatch, cfg):
    data = prepared(monkeypatch, cfg=cfg)
    batch = 2 if cfg else 1
    rows = token_rows({"grid_id":data["grid_id"], "index":data.get("kv_index"),
                      "actions":data["noisy_latents"]},
                     batch_size=batch, length=4, main_count=4, action_mode=True,
                     update_cache=1, device="cpu")
    assert rows["observation_flag"].tolist() == [True,True,False,False]
    assert data["noisy_latents"][:,:,0].eq(0).all()
    assert data["timesteps"][:,0].eq(0).all()

@pytest.mark.parametrize("policy,clamp", [("fifo",True),("global",False)])
def test_unclamped_or_legacy_actions_keep_original_contract(monkeypatch,policy,clamp):
    data=prepared(monkeypatch,policy=policy,clamp=clamp)
    assert "kv_index" not in data

@torch.no_grad()
def test_prediction_clear_preserves_exactly_seed_actions(monkeypatch):
    model=tiny_model()
    data=prepared(monkeypatch)
    data["tactile_global_latent"] = torch.randn(1,1,48,2,4,4)
    data["tactile_sensor_ids"] = torch.zeros(1,1,dtype=torch.long)
    model(data, update_cache=1, cache_name="test", action_mode=True)
    before=model.get_global_retention("test")
    assert before["observation_flag"].sum()==2
    model.clear_pred_cache("test")
    for attention in model.mot.shared_attn:
        cache=attention.attn_caches["test"]
        assert cache["mask"].sum()==2
        assert not cache["is_pred"][cache["mask"]].any()
    after=model.get_global_retention("test")
    assert len(after["token_uid"])==2
    assert after["kind"].eq(1).all()
    assert after["observation_flag"].all()

@torch.no_grad()
def test_seed_metadata_does_not_change_current_action_prediction(monkeypatch):
    from copy import deepcopy
    reference=tiny_model()
    changed=tiny_model()
    changed.load_state_dict(reference.state_dict())
    data=prepared(monkeypatch)
    data["tactile_global_latent"]=torch.randn(1,1,48,2,4,4)
    data["tactile_sensor_ids"]=torch.zeros(1,1,dtype=torch.long)
    old=deepcopy(data)
    old.pop("kv_index")
    expected=reference(old,update_cache=1,cache_name="test",action_mode=True)
    actual=changed(data,update_cache=1,cache_name="test",action_mode=True)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(reference.mot.shared_attn,changed.mot.shared_attn):
        x,y=a.attn_caches["test"],b.attn_caches["test"]
        for key in ("k","v","mask","id"):
            torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
