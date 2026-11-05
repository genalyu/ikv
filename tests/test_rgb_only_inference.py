"""Explicit RGB-only serving never constructs tactile tokens."""
import pytest
import torch
from unittest.mock import patch
from n0_twam.models.mot import WanMoTTransformer3DModel

def _tiny_mot_model():
    return WanMoTTransformer3DModel(patch_size=(1,2,2),num_attention_heads=1,attention_head_dim=4,in_channels=2,out_channels=2,action_dim=3,text_dim=8,freq_dim=4,ffn_dim=8,num_layers=1,rope_max_seq_len=16,attn_mode="torch",use_local_tactile=False).cpu().eval()

@pytest.mark.parametrize("action_mode", [False, True])
def test_rgb_only_video_and_action(action_mode):
    model = _tiny_mot_model()
    channels = 3 if action_mode else 2
    x = torch.randn(1, channels, 1, 2 if action_mode else 4, 2 if action_mode else 4)
    inp = dict(noisy_latents=x, timesteps=torch.zeros(1,1),
               grid_id=torch.zeros(1,4,4,dtype=torch.long),
               text_emb=torch.randn(1,3,8), rgb_only=True)
    with patch.object(model.mot.experts["tactile"], "embed_in", side_effect=AssertionError("inactive expert invoked")), patch.object(model, "_encode_tactile_condition", side_effect=AssertionError("tactile invoked")):
        with torch.no_grad():
            out = model(inp, action_mode=action_mode, cache_name=None)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4 if action_mode else 16, channels)
    del inp["rgb_only"]
    with pytest.raises(KeyError, match="tactile_global_latent"):
        model(inp, action_mode=action_mode, cache_name=None)

def test_rgb_only_rejects_tactile_payload():
    model = _tiny_mot_model()
    inp = dict(noisy_latents=torch.randn(1,2,1,4,4),
               timesteps=torch.zeros(1,1), grid_id=torch.zeros(1,4,4,dtype=torch.long),
               text_emb=torch.randn(1,3,8), rgb_only=True, tactile_sensor_ids=torch.zeros(1,1))
    with pytest.raises(ValueError, match="cannot carry"):
        model(inp, cache_name=None)
