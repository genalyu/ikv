"""Removed sparse-motion requests fail at the public model/recipe boundary."""
from types import SimpleNamespace
import pytest
import torch
from n0_twam.input_contract import validate_dense_rgb_config
from n0_twam.models.mot import WanMoTTransformer3DModel
from n0_twam.task_pipeline.config import MODES, training_config
@pytest.mark.parametrize("mode", ["motion","motion_ikv"])
def test_removed_modes_rejected(mode):
    assert set(MODES)=={"baseline","ikv"}
    with pytest.raises(ValueError,match="baseline and ikv"):
        training_config({},mode,8,require_ready=False)
@pytest.mark.parametrize("flag", ["use_rgb_motion_tokens","rgb_motion_online_preprocess"])
def test_removed_config_cannot_silently_change_recipe(flag):
    with pytest.raises(ValueError,match="removed"):
        validate_dense_rgb_config(SimpleNamespace(**{flag:True}))
@pytest.mark.parametrize("key", ["rgb_motion_patch_indices","motion_indices","condition_motion"])
def test_model_rejects_removed_payload_before_computation(key):
    model=WanMoTTransformer3DModel(patch_size=(1,2,2),num_attention_heads=1,attention_head_dim=4,
        in_channels=2,out_channels=2,action_dim=3,text_dim=8,freq_dim=4,ffn_dim=8,
        num_layers=0,rope_max_seq_len=16,attn_mode="torch",use_local_tactile=False)
    with pytest.raises(ValueError,match="removed"):
        model({key:torch.zeros(1)},cache_name=None)
    with pytest.raises(ValueError,match="removed"):
        model._prepare_train_inputs({"latent_dict":{key:torch.zeros(1)}})

def test_sparse_checkpoint_is_rejected_before_loading_weights(tmp_path):
    import json
    from n0_twam.models.utils import load_mot_checkpoint
    (tmp_path/"config.json").write_text(json.dumps({"is_mot":True,"use_rgb_motion_tokens":True}))
    with pytest.raises(ValueError,match="removed"):
        load_mot_checkpoint(str(tmp_path))
